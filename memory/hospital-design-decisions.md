---
name: hospital-design-decisions
description: Key design decisions and fixes for the hospital system
metadata:
  type: reference
---

# 关键设计决策

1. **方案选型**: 模块化分层架构——单 FastAPI 应用，内部模块化，Docker Compose 编排
2. **RAG + DL 双通道并行**: `/api/diagnose` 中 `asyncio.gather` 并行执行
3. **RAG 用真实数据**: 中文医疗对话（22万→1.8万条，191种病）——文本检索需要多样性
4. **DL 用合成数据**: 医学知识驱动的化验指标模式（62种病，12000条）——中文结构化化验数据不可得
5. **Embedding 选型**: text2vec → bge-base-zh-v1.5（768维），中文检索更强
6. **DL 模型选型**: XGBoost 主力（50.9%, 62类），MLP 做基线对照（39.9%）——表格数据上梯度提升优于 MLP
7. **OCR 双引擎**: 百度 OCR 精确版 + EasyOCR 降级 + DeepSeek 结构化 + 自动提取患者信息
8. **API Key 安全**: `.env` 管理，git 历史已清洗
9. **开发体验**: 源码挂载 + uvicorn --reload，改代码自动生效

## 遇到的坑和修复

1. **pgvector Python list 传参失败**: 修复：转字符串 + SQL 用 `CAST(:param AS vector)`
2. **Docker 内 HF 模型下载超时**: 修复：挂载本机 HF 缓存 + 配置 `HF_ENDPOINT` 镜像
3. **Dockerfile 每次启动都训练模型**: 修复：加 `if [ ! -f ]` 判断，模型存在则跳过
4. **百度 OCR `accurateBasic` 不存在**: 修复：正确方法名是 `basicAccurate`，且传原始 bytes 而非 base64
5. **百度 SDK 缺少 `chardet` 依赖**: 修复：加入 requirements.txt
6. **疾病名提取噪音大**: 放弃通用正则，改用 200+ 常见病名列表精确匹配 + 标题优先策略

**Why:** 记录关键决策和 bug 修复，避免重复踩坑。[[hospital-system-project-overview]]
