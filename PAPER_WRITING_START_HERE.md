# 论文写作入口

这个公开仓库同时包含 D-FINE 代码快照与按项目知识库层级整理的研究文档。先从 [知识库总索引](knowledge/README.md) 开始。

1. [核心定位备忘录：SAM形状指导与红外内容补充](knowledge/论文写作前置准备/20260926_论文研究定位与核心论证备忘录_SAM形状指导与红外内容补充_ZH.md)
2. [当前SAM贡献与空间结构假设裁决](knowledge/论文写作前置准备/20260927_R2中SAM指导的贡献与空间结构假设裁决_ZH.md)
3. [SAM与M机制关联审计结果](knowledge/论文写作前置准备/20260927_SAM与M机制关联_固定权重干预结果与Astra交接_ZH.md)
4. [红外融合结果与机制边界](knowledge/M系列知识库/10_目标证据驱动的红外融合/12_普通双模态融合对照_结果与机制边界_ZH.md)

建议先让 AI 总结研究问题、方法信息流和已有证据，再起草摘要、引言与方法。核心表述为“以形状知识约束目标表征，以内容关联获取跨模态补充”。资料包已明确区分设计动机与实验证据；请勿把“保护融合后的空间结构”写成已经证明的机制。

代码入口包括 `src/zoo/dfine/dfine.py`、`src/zoo/dfine/target_evidence_thermal_fusion.py`、`src/zoo/dfine/sam_group_contrast.py`、`src/zoo/dfine/sam_query_evidence_reader.py`。结果和结论以知识库中的完成实验记录为准。
