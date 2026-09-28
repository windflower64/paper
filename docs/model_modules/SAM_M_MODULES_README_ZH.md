# SAM 与 M 模块代码入口

本目录对应论文方法中的两个核心模块。数据集、训练权重、日志和临时实验脚本不属于发布代码。

## 1. SAM 形状指导模块

SAM 只作为训练阶段的外部形状教师，不进入推理路径。

- `src/zoo/dfine/sam_group_contrast.py`：SGC2，利用 SAM 掩码与框区域构造分组形状约束。
- `src/zoo/dfine/sam_query_evidence_reader.py`：SQER2，为匹配到的检测查询读取局部形状证据。
- `src/zoo/dfine/sam_mask_aggregation.py`：掩码聚合与基础损失函数。
- `src/zoo/dfine/sam_query_mask_init.py`：查询级掩码初始化辅助模块。
- `src/zoo/dfine/sam_scale_supervision.py`：多尺度形状监督。
- `src/zoo/dfine/sam_support_shape.py`：候选区域支持形状聚合。
- `src/data/dataset/coco_dataset.py`：读取单目标和多目标 SAM 实例掩码，并检查掩码数量与标注框数量一致。

## 2. M 红外内容融合模块

M 模块使用红外分支提供目标相关内容证据，同时将 RGB 保留为检测位置与形状的主要参照。

- `src/zoo/dfine/target_evidence_thermal_fusion.py`：目标证据驱动的红外内容融合。
- `src/zoo/dfine/qdmf.py`：查询条件动态模态融合组件。
- `src/zoo/dfine/dfine_decoder.py`：M 模块在 D-FINE 解码器中的接入口，以及红外候选证据读取和分类校准逻辑。

## 3. 运行边界

- SAM 掩码只用于训练监督；推理时不需要加载 SAM 模型或掩码文件。
- M 模块是 RGB-T 双模态路径；单模态控制实验不属于发布接口。
- 远程仓库不包含数据集和 checkpoint，使用者需要自行配置数据路径与权重。
