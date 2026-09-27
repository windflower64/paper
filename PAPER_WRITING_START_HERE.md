# 论文写作入口

这个公开仓库同时包含 D-FINE 代码快照与论文研究资料。写论文请先读以下两份中文材料：

1. [核心定位备忘录：SAM形状指导与红外内容补充](docs/paper-ai/20260926_论文研究定位与核心论证备忘录_SAM形状指导与红外内容补充_ZH.md)
2. [完整研究资料包：方法、实验结果、机制边界与历史记录](docs/paper-ai/Research_Handoff_Public_ZH.md)

建议先让 AI 总结研究问题、方法信息流和已有证据，再起草摘要、引言与方法。核心表述为“以形状知识约束目标表征，以内容关联获取跨模态补充”。资料包已明确区分设计动机与实验证据；请勿把“保护融合后的空间结构”写成已经证明的机制。

代码入口包括 `src/zoo/dfine/dfine.py`、`src/zoo/dfine/target_evidence_thermal_fusion.py`、`src/zoo/dfine/sam_group_contrast.py`、`src/zoo/dfine/sam_query_evidence_reader.py`。结果和结论以资料包中对应的实验记录为准。
