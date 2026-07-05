---
name: hospital-system-project-overview
description: Complete project summary of the hospital diagnosis system
metadata:
  type: project
---

# 智能医疗诊断辅助系统 — 项目概览

**项目路径**: `d:\hospital_system`
**GitHub**: https://github.com/keep-discipline/hospital-diagnosis-system
**Git**: ~25 commits on `main`

## 技术栈

| 层 | 技术 |
|---|---|
| 前端 | React 18 + TypeScript 5 + Vite，端口 3000 |
| 后端 | FastAPI (Python 3.11)，端口 8000 |
| 数据库 | PostgreSQL 16 + pgvector，端口 5432 |
| Embedding | BAAI/bge-base-zh-v1.5 (768维)，通过 ModelScope 下载 |
| RAG 数据 | 1.8 万条真实医疗对话（191 种疾病），来源好大夫+寻医问药 |
| DL 诊断 | XGBoost + 合成数据 12000 条（62 种疾病，8 大系统） |
| OCR | 百度 OCR 精确版 + EasyOCR 降级 + DeepSeek 结构化 |
| 部署 | Docker Compose 三容器 + 源码挂载热重载 |

## 数据来源策略

```
RAG 通道 ── 真实医疗对话（Chinese-medical-dialogue-data）
             22 万条 CSV → NLP 清洗 → 1.8 万条入库
             包含症状描述 + 诊断 + 治疗方案（191 种病）

DL 通道 ── 医学知识模拟（data_generator_v2.py）
             基于真实指标模式生成 62 种病 × 200 条 = 12000 条
             含合并症 (20%) + 缺失值 (8%) + 随机噪声
```

## 项目结构

```
hospital-diagnosis-system/
├── backend/app/
│   ├── main.py                   # FastAPI 入口，启动自动导入数据
│   ├── api/                      # 6 个端点
│   ├── services/                 # rag / diagnosis / patient / ocr
│   ├── ml/                       # embedding (bge), data_generator_v2, train_v2, diagnosis_model
│   ├── models/patient.py         # SQLAlchemy ORM
│   ├── schemas/diagnosis.py      # Pydantic schemas
│   └── scripts/                  # clean_medical_data, import_real_data
├── frontend/src/
│   ├── pages/                    # DiagnosisPage (双栏+加载动画)
│   ├── components/               # PatientForm(校验), LabReportForm(OCR+患者信息), Result, Similar, Skeleton
├── data/
│   ├── models/                   # xgboost_baseline.pkl + model_meta.json
│   └── init.sql
├── docker-compose.yml            # 三容器 + .env + 源码挂载
├── .env.example
└── docs/superpowers/
```

## 如何运行

```bash
cp .env.example .env    # 填入 API Key
docker compose up --build
# http://localhost:3000   前端
# http://localhost:8000/docs  API 文档
```

**Why:** 完整项目上下文，下次继续开发时快速恢复。[[hospital-design-decisions]]

**How to apply:** 下次对话时提及此项目即可恢复上下文。
