# 位置搜索与对应 SAM 监督方案的完整研究快照

本目录同时保存 SBR1、SBR2、SBR3，不仅保存最佳 GT-only 结果。`中文记录` 包含设计、失败分析及后续 SAM 多尺度计划；`实验记录` 保持每组独立协议、逐轮日志、状态及小型精修头权重；两个 ANALYSIS 目录保存结果核验。

代码入口：`tools/train_sbr1.py`、`train_sbr2.py`、`train_sbr3.py`；模块位于 `src/zoo/dfine/sam_boundary_refiner.py`、`sam_boundary_refiner_v2.py`、`sam_side_search.py`。相应启动脚本、测试和分析脚本同在 tools。

SBR3 的 R1_GT_ONLY 与 R2_SAM 使用同一位置搜索结构，后者额外接受 SAM 形状监督。两组都有最佳及末轮精修头权重；它们并非完整检测器权重。SBR2 主动中止，不应视为完成的双组比较。

恢复原运行目录时，把实验记录的各 SBR 子目录放回项目父目录 outputs，并保持代码目录名 D-FINE。启动脚本使用本机路径与环境，移机需检查并调整；勿直接覆盖现有实验。

外部必需资源未上传：RGB-T 数据、SAM 掩码缓存、冻结检测器权重。冻结基座原路径为 E:/two_paper/outputs/M_LAYOUT_RECOVERY_20E_TESTDEV/R2_OTE2_LAYOUT/seed0/best_stg1.pth，SHA256 为 d7fc60568887e12ba66fbe5b474d09be5cfcc5dd4fe14bc102556b5e5b5aa952。详细输入配置见训练源码及各组 protocol.json。缺少这些资源不能独立重跑。本次保留现有 SAM 多实例有效性修复作为运行依赖，不把它计为 SBR 创新。

位置搜索暂列第二篇候选；当前主线继续研究 SAM。两条路线均保留，不删除未获增益的 SAM 监督结果。
