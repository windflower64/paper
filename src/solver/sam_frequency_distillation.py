"""FreeKD风格的训练期频率蒸馏适配器。

首版只迁移公开公式：冻结A00教师、官方频率提示、同层高频L1蒸馏。
它不修改学生前向，不进入验证/部署，也不把SAM掩码直接写入损失。
项目专属SAM门控必须等公开公式单独通过后再启用。
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from pytorch_wavelets import DWTForward

from ..core import YAMLConfig


def _checkpoint_weights(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("ema", {}).get("module") is not None:
        return state["ema"]["module"], "ema.module"
    if "model" in state:
        return state["model"], "model"
    return state, "raw"


def _load_prompt(checkpoint_path, module_index, level_index):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", state)
    key = f"mask_modules.{int(module_index)}.mask_token"
    if key not in state_dict:
        raise KeyError(f"missing FreeKD prompt tensor: {key}")
    tokens = state_dict[key].float()
    if tuple(tokens.shape) != (4, 6, 256):
        raise ValueError(f"unexpected FreeKD prompt shape for {key}: {tuple(tokens.shape)}")
    if not 0 <= int(level_index) < 3:
        raise ValueError("frequency level_index must be 0, 1, or 2")
    # mask_token[0]对应低频；三个高频分解层级使用mask_token[1:4]。
    return tokens[int(level_index) + 1].contiguous(), key


def _disable_cached_experimental_branches(yaml_cfg):
    """抵御仓库可变YAML include缓存造成的旧实验字段泄漏。"""
    backbone = yaml_cfg.get("HGNetv2", {})
    for key in (
        "btrd_stage",
        "pdbr_stage",
        "tndp_stage",
        "bpc_stage",
        "qrl_source_stage",
        "mdqa_source_stage",
        "pat_stage",
        "pat_sf_stage",
        "rpc_stage",
        "srtod_stage",
        "spatial_aux_stage",
        "lad_stage",
        "preserve_stage",
        "blur_stage",
    ):
        if key in backbone:
            backbone[key] = -1
    for key in (
        "spar_enabled",
        "sabr_enabled",
        "sbrd_enabled",
        "qcsr_enabled",
    ):
        if key in backbone:
            backbone[key] = False
    backbone["pretrained"] = False

    criterion = yaml_cfg.get("DFINECriterion", {})
    for key in tuple(criterion):
        if key.endswith("_aux_weight") or key in {
            "spar_aux_weight",
            "sbrd_aux_weight",
            "qcsr_teacher_weight",
            "qcsr_transfer_weight",
        }:
            criterion[key] = 0.0


class SAMFrequencyDistiller:
    """冻结教师和官方提示均不注册进学生优化器或训练checkpoint。"""

    def __init__(self, student, spec, device):
        self.device = device
        self.loss_weight = float(spec.get("loss_weight", 1.0))
        if self.loss_weight <= 0:
            raise ValueError("frequency_distillation.loss_weight must be positive")
        self.level_index = int(spec.get("level_index", 0))
        self.module_index = int(spec.get("prompt_module_index", 4))
        self.sam_gate_mode = str(spec.get("sam_gate_mode", "none"))
        if self.sam_gate_mode != "none":
            raise ValueError(
                "S-FREQ2 public reproduction requires sam_gate_mode=none; "
                "SAM gating is reserved for the next isolated experiment"
            )

        teacher_config = Path(spec["teacher_config"])
        teacher_checkpoint = Path(spec["teacher_checkpoint"])
        prompt_checkpoint = Path(spec["prompt_checkpoint"])
        for path in (teacher_config, teacher_checkpoint, prompt_checkpoint):
            if not path.is_file():
                raise FileNotFoundError(path)

        # 构造冻结教师不能改变学生训练的CPU/CUDA随机序列。
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            teacher_cfg = YAMLConfig(str(teacher_config))
            _disable_cached_experimental_branches(teacher_cfg.yaml_cfg)
            teacher = teacher_cfg.model.to(device).eval()
            weights, weight_source = _checkpoint_weights(teacher_checkpoint)
            teacher.load_state_dict(weights, strict=True)
        finally:
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher = teacher
        self.teacher_weight_source = weight_source
        self.teacher_checkpoint = str(teacher_checkpoint)

        prompt, prompt_key = _load_prompt(
            prompt_checkpoint, self.module_index, self.level_index
        )
        self.prompt = prompt.to(device)
        self.prompt_key = prompt_key
        self.prompt_checkpoint = str(prompt_checkpoint)
        self.transform = DWTForward(J=3, mode="zero", wave="haar").to(device)

        self.student_cache = {}
        self.teacher_cache = {}
        student_backbone = student.backbone
        if len(student_backbone.stages) <= 2 or len(teacher.backbone.stages) <= 2:
            raise ValueError("frequency distillation requires the real S8->S16 stage")

        def capture_student(_module, inputs):
            self.student_cache["s8"] = inputs[0]

        def capture_teacher(_module, inputs):
            self.teacher_cache["s8"] = inputs[0].detach()

        self.student_hook = student_backbone.stages[2].register_forward_pre_hook(
            capture_student
        )
        self.teacher_hook = teacher.backbone.stages[2].register_forward_pre_hook(
            capture_teacher
        )
        self.last_stats = {}

    @staticmethod
    def _prompt_attention(coefficients, prompt):
        # 严格复刻FreeKD：当前分解层级先对三个方向求和，再与6个提示点积。
        frequency_slice = coefficients.sum(dim=2)
        return torch.einsum("tc,bchw->bthw", prompt, frequency_slice).sigmoid()

    def __call__(self, samples, targets=None):
        if "s8" not in self.student_cache:
            raise RuntimeError("student S8 was not captured before frequency loss")
        with torch.no_grad(), torch.autocast(
            device_type=str(self.device), dtype=torch.float16, enabled=self.device.type == "cuda"
        ):
            self.teacher.backbone(samples)
        if "s8" not in self.teacher_cache:
            raise RuntimeError("teacher S8 was not captured")

        student_s8 = self.student_cache["s8"].float()
        teacher_s8 = self.teacher_cache["s8"].float()
        if student_s8.shape != teacher_s8.shape:
            raise RuntimeError(
                f"student/teacher S8 mismatch: {student_s8.shape} vs {teacher_s8.shape}"
            )
        if student_s8.shape[1] != self.prompt.shape[1]:
            raise RuntimeError(
                f"prompt channel mismatch: S8={student_s8.shape[1]}, prompt={self.prompt.shape[1]}"
            )

        _student_low, student_levels = self.transform(student_s8)
        with torch.no_grad():
            _teacher_low, teacher_levels = self.transform(teacher_s8)
            teacher_high = teacher_levels[self.level_index]
            attention = self._prompt_attention(teacher_high, self.prompt)
        student_high = student_levels[self.level_index]

        mask = attention.unsqueeze(2).unsqueeze(2)
        teacher_selected = teacher_high.unsqueeze(1) * mask
        student_selected = student_high.unsqueeze(1) * mask
        raw_loss = F.l1_loss(student_selected, teacher_selected)
        weighted_loss = self.loss_weight * raw_loss
        with torch.no_grad():
            self.last_stats = {
                "raw_loss": float(raw_loss),
                "weighted_loss": float(weighted_loss),
                "attention_mean": float(attention.mean()),
                "attention_std": float(attention.std()),
                "student_high_rms": float(student_high.square().mean().sqrt()),
                "teacher_high_rms": float(teacher_high.square().mean().sqrt()),
            }
        return weighted_loss

    def describe(self):
        return {
            "teacher_checkpoint": self.teacher_checkpoint,
            "teacher_weight_source": self.teacher_weight_source,
            "prompt_checkpoint": self.prompt_checkpoint,
            "prompt_key": self.prompt_key,
            "prompt_module_index": self.module_index,
            "frequency_level_index": self.level_index,
            "loss_weight": self.loss_weight,
            "sam_gate_mode": self.sam_gate_mode,
            "student_feature": "real internal S8 before standard S8->S16 stage",
            "inference_change": False,
        }

    def close(self):
        self.student_hook.remove()
        self.teacher_hook.remove()
