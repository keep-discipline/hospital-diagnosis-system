---
name: hospital-system-project-overview
description: Complete project summary of the hospital diagnosis system
metadata:
  type: project
---

# 智能医疗诊断辅助系统 — 项目概览

**项目路径**: `d:\hospital_system`
**GitHub**: https://github.com/keep-discipline/hospital-diagnosis-system
**Git**: ~22 commits on `main`

## 技术栈

| 层 | 技术 |
|---|---|
| 前端 | React 18 + TypeScript 5 + Vite，端口 3000 |
| 后端 | FastAPI (Python 3.11)，端口 8000 |
| 数据库 | PostgreSQL 16 + pgvector，端口 5432 |
| Embedding | BAAI/bge-base-zh-v1.5 (768维)，中文检索最强之一 |
| 诊断模型 | PyTorch MLP (20→128→64→32→10)，合成数据训练 |
| RAG 数据 | 18K 条真实医疗对话（191 种疾病），来源好大夫+寻医问药 |
| OCR | 百度 OCR 精确版 + EasyOCR 降级 + DeepSeek 结构化 |
| 部署 | Docker Compose 三容器 |

## 项目结构

```
hospital-diagnosis-system/
├── backend/app/
│   ├── main.py                   # FastAPI 入口，启动自动导入数据
│   ├── api/                      # 6 个端点 (health, diagnose, patients, diseases, ocr)
│   ├── services/                 # rag_service, diagnosis_service, patient_service, ocr_service
│   ├── ml/                       # embedding.py, diagnosis_model.py, train.py, data_generator.py
│   ├── models/patient.py         # SQLAlchemy ORM (含 pgvector Vector(768))
│   ├── schemas/diagnosis.py      # Pydantic (LabReport 21项)
│   └── scripts/                  # clean_medical_data.py, import_real_data.py
├── frontend/src/
│   ├── pages/                    # DiagnosisPage (双栏+加载动画), ResultPage
│   ├── components/               # PatientForm(校验), LabReportForm(OCR), DiagnosisResult, SimilarCases, LoadingSkeleton
│   ├── services/api.ts           # axios 封装
│   └── types/diagnosis.ts        # TS 类型（对齐后端 Pydantic）
├── data/
│   ├── chinese_medical_dialogue/ # 22万条真实医疗对话原始数据
│   ├── mock_patients.json        # 500条模拟病例（已废弃，替换为真实数据）
│   └── models/                   # 训练好的 DL 模型权重
├── docker-compose.yml            # 三容器编排 + .env 管理密钥
├── .env.example                  # 环境变量模板
└── docs/superpowers/             # 设计文档 + 实现计划
```

## OCR 化验单识别

```
POST /api/ocr (图片上传)
  ├── 百度 OCR 精确版 (主引擎) → 文字提取
  │   └── 失败时降级 → EasyOCR (本地 PyTorch，无需网络)
  └── DeepSeek API (结构化) → { patient_info: {name, age, gender} + lab_data: 21项 }
  → 前端自动填入患者信息 + 化验指标
```

## 数据流

```
POST /api/diagnose
  ├── RAG 通道: 症状文字 → bge-base-zh embedding → pgvector 余弦检索 → Top-5 相似病例 (18K 真实数据)
  └── DL 通道: 化验单 → Tensor[1,20] → MLP → Softmax → Top-3 疾病 + 治疗建议
  → 合并返回 + 异步存库
```

## 关键设计决策

1. **RAG vs DL 数据分离**: RAG 用真实对话（不限病种），DL 用合成化验数据（结构化数值）
2. **OCR 自动提取患者信息**: 化验单上有姓名/年龄/性别，不必手动输入
3. **191 种疾病 vs 初始 10 种**: RAG 不需要限制疾病种类，越多越好
4. **bge-base-zh**: 中文检索 benchmark 最强通用模型之一
5. **API Key 管理**: 通过 `.env` 文件，不硬编码

## 如何运行

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 API Key

# 2. 启动
docker compose up --build

# 3. 浏览器:
# http://localhost:3000     前端
# http://localhost:8000/docs API文档
```

**Why:** 完整项目上下文，下次继续开发时快速恢复。[[hospital-design-decisions]]

**How to apply:** 下次对话时提及此项目即可恢复上下文。
