---
name: hospital-design-decisions
description: Key design decisions and fixes for the hospital system
metadata:
  type: reference
---

# 关键设计决策

1. **方案选型**: 模块化分层架构（方案A）而非微服务——单 FastAPI 应用，内部模块化，Docker Compose 编排
2. **MLP vs Transformer**: 化验单是结构化数值数据，用 MLP 而非 Transformer。Transformer 仅用于 RAG 的文本 embedding
3. **pgvector vs 独立向量数据库**: 用 pgvector 扩展，一个 PG 同时存结构化数据和向量，简化架构
4. **模型加载策略**: Embedding 模型和 DL 模型都采用全局单例模式，启动时加载一次
5. **RAG + DL 并行**: `/api/diagnose` 中用 `asyncio.gather` 并行执行两个通道
6. **OCR 双引擎 + AI 结构化**: 百度 OCR 精确版（主）+ EasyOCR（降级）+ DeepSeek 结构化，百度不可用时自动降级
7. **RAG 数据策略**: 用真实医疗对话（22万条 → 清洗出 18K/191 种疾病），不用合成文本——多样性远超 10 种病的模拟数据
8. **Embedding 选型**: text2vec → bge-base-zh-v1.5，中文检索 benchmark 更强，零代码切换（同 768 维）
9. **OCR 自动提取患者信息**: 化验单本身有姓名/年龄/性别，深度学习 prompt 中加字段即可同时提取，避免重复录入
10. **API Key 安全**: 所有密钥通过 `.env` 文件管理，git 历史已用 `filter-branch` 清洗

## 遇到的坑和修复

1. **pgvector Python list 传参失败**: 修复：转字符串 + SQL 用 `CAST(:param AS vector)`
2. **Docker 内 HF 模型下载超时**: 修复：挂载本机 HF 缓存 + 配置 `HF_ENDPOINT` 镜像
3. **Dockerfile 每次启动都训练模型**: 修复：加 `if [ ! -f ]` 判断，模型存在则跳过
4. **百度 OCR `accurateBasic` 不存在**: 修复：正确方法名是 `basicAccurate`，且传原始 bytes 而非 base64
5. **百度 SDK 缺少 `chardet` 依赖**: 修复：加入 requirements.txt
6. **疾病名提取噪音大**: 放弃通用正则，改用 200+ 常见病名列表精确匹配 + 标题优先策略

**Why:** 记录关键决策和 bug 修复，避免重复踩坑。[[hospital-system-project-overview]]
