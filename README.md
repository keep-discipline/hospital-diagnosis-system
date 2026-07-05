# 🏥 智能医疗诊断辅助系统

> An AI-powered hospital diagnosis assistant that combines **RAG** (Retrieval-Augmented Generation) with **Deep Learning** to help doctors make faster, more accurate diagnoses.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61dafb)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.5-3178c6)](https://www.typescriptlang.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-ee4c2c)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ed)](https://docs.docker.com/compose/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📖 目录

- [系统架构](#-系统架构)
- [功能特性](#-功能特性)
- [技术栈](#-技术栈)
- [快速开始](#-快速开始)
- [API 接口](#-api-接口)
- [项目结构](#-项目结构)
- [配置说明](#-配置说明)
- [数据流](#-数据流)
- [未来计划](#-未来计划)

---

## 🏗 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                     React 前端 (:3000)                    │
│   PatientForm │ LabReportForm │ DiagnosisResult │ OCR   │
└──────────────────────┬──────────────────────────────────┘
                       │ POST /api/diagnose
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI 后端 (:8000)                     │
│                                                          │
│   ┌──────────────────┐   ┌──────────────────┐           │
│   │   RAG 检索通道    │   │   DL 诊断通道    │           │
│   │                   │   │                  │           │
│   │  症状文本         │   │  化验单数据      │           │
│   │     ↓             │   │     ↓            │           │
│   │  Transformer      │   │  MLP 模型        │           │
│   │  Embedding        │   │  (20→128→64→32  │           │
│   │     ↓             │   │    →10)          │           │
│   │  pgvector         │   │     ↓            │           │
│   │  余弦检索         │   │  Softmax         │           │
│   │     ↓             │   │     ↓            │           │
│   │  Top-5 相似病例   │   │  Top-3 疾病预测  │           │
│   └──────────────────┘   └──────────────────┘           │
│                       │  asyncio.gather 并行              │
│                       ▼                                  │
│                 合并结果 → 诊断建议 + 治疗方案              │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              PostgreSQL 16 + pgvector (:5432)            │
│     patients 表 (含 symptom_embedding vector(768))       │
└─────────────────────────────────────────────────────────┘
```

**OCR 化验单识别流程：**

```
化验单图片 ──→ 百度 OCR 精确版（主引擎）
                  │
                  ├── 成功 ──→ DeepSeek API 结构化 ──→ 21 项化验指标 JSON
                  │
                  └── 失败 ──→ EasyOCR（本地降级）──→ DeepSeek API 结构化
```

---

## ✨ 功能特性

- 🔍 **AI 智能诊断**：输入症状描述 + 化验单数据，深度学习模型给出 Top-3 疾病预测及置信度
- 📚 **RAG 相似病例检索**：基于 Transformer 文本向量化 + pgvector 余弦检索，匹配最相似的历史病例作为参考
- 📸 **化验单 OCR 识别**：上传化验单照片，自动识别 21 项血液指标填入表单
  - 百度 OCR 精确版（中英文混合识别，高准确率）
  - EasyOCR 本地引擎（降级备用，无需网络）
  - DeepSeek API 结构化解析
- 💊 **治疗建议**：根据诊断结果自动生成治疗建议
- 🐳 **一键部署**：Docker Compose 三容器编排，开箱即用
- 📊 **191 种疾病覆盖**：基于 22 万条真实医疗对话数据，覆盖内科常见病
- 🎨 **现代 UI**：React + TypeScript 响应式界面，左右分栏布局

---

## 🛠 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 前端 | React 18 + TypeScript 5 + Vite | SPA，端口 3000 |
| 后端 | FastAPI (Python 3.11) | 异步 API，端口 8000 |
| 数据库 | PostgreSQL 16 + pgvector | 结构化存储 + 向量检索，端口 5432 |
| Embedding | `BAAI/bge-base-zh-v1.5` | 中文检索最强通用模型，768 维 |
| 诊断模型 | PyTorch MLP (20→128→64→32→10) | 训练准确率 ~94% |
| OCR | 百度 OCR + EasyOCR + DeepSeek | 化验单图片识别 + AI 结构化 |
| 部署 | Docker + Docker Compose | 三容器一键编排 |
| 容器化 | `python:3.11-slim` / `node:20-slim` / `nginx:alpine` | 镜像构建 |

---

## 🚀 快速开始

### 前置要求

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)（需运行中）
- [Git](https://git-scm.com/)

### 1. 克隆项目

```bash
git clone https://github.com/keep-discipline/hospital-diagnosis-system.git
cd hospital-diagnosis-system
```

### 2. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 填入你的 API Key（OCR 功能需要）
# DEEPSEEK_API_KEY    — DeepSeek API Key（必需，用于 OCR 结构化）
# BAIDU_OCR_APP_ID     — 百度 OCR App ID（可选，不填则只用 EasyOCR）
# BAIDU_OCR_API_KEY    — 百度 OCR API Key
# BAIDU_OCR_SECRET_KEY — 百度 OCR Secret Key
```

> 💡 不配置百度 OCR 也能使用——系统会自动降级到 EasyOCR（本地引擎，无需 API Key）。

### 3. 启动服务

```bash
# 首次启动需构建镜像（含 PyTorch/EasyOCR 依赖，约 5-15 分钟）
docker compose up --build

# 后续启动直接：
docker compose up -d
```

### 4. 打开浏览器

| 地址 | 说明 |
|------|------|
| http://localhost:3000 | 前端界面 |
| http://localhost:8000/docs | Swagger API 文档 |
| http://localhost:8000/api/health | 健康检查 |

### 5. 使用

1. 填写患者信息（姓名、年龄、性别、症状描述）
2. 上传化验单照片自动识别，或手动输入 21 项化验指标
3. 点击「开始智能诊断」
4. 查看 AI 诊断结果 + 相似历史病例参考

---

## 📡 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/health` | 健康检查 |
| `POST` | `/api/diagnose` | **核心接口**：提交症状+化验单，返回诊断 |
| `POST` | `/api/ocr` | 上传化验单图片，OCR 识别 + 结构化 |
| `GET` | `/api/patients` | 病人列表（分页） |
| `GET` | `/api/patients/{id}` | 病人详情 |
| `GET` | `/api/diseases` | 支持的疾病列表 |

### POST /api/diagnose 示例

```bash
curl -X POST http://localhost:8000/api/diagnose \
  -H "Content-Type: application/json" \
  -d '{
    "name": "张三",
    "age": 45,
    "gender": "male",
    "symptom_description": "发烧咳嗽三天，黄痰，胸闷气短",
    "lab_report": {
      "wbc": 15.2, "neutrophil_pct": 88.5, "lymphocyte_pct": 10.2,
      "crp": 65.8, "temperature": 38.9, "systolic_bp": 128,
      "diastolic_bp": 82, "heart_rate": 102, "respiratory_rate": 24,
      "spo2": 92, "rbc": 4.5, "hemoglobin": 138, "hematocrit": 41.5,
      "platelet": 260, "glucose": 5.6, "creatinine": 78, "bun": 4.8,
      "alt": 24, "ast": 22, "total_cholesterol": 4.6, "triglycerides": 1.3
    }
  }'
```

### POST /api/ocr 示例

```bash
curl -X POST http://localhost:8000/api/ocr \
  -F "file=@lab_report.jpg"
```

---

## 📁 项目结构

```
hospital-diagnosis-system/
├── backend/                          # FastAPI 后端
│   ├── app/
│   │   ├── main.py                   # 应用入口 + 生命周期管理
│   │   ├── config.py                 # 配置（环境变量驱动）
│   │   ├── database.py               # 数据库连接管理
│   │   ├── api/
│   │   │   ├── router.py             # 路由聚合
│   │   │   ├── health.py             # 健康检查
│   │   │   ├── diagnosis.py          # 诊断核心接口
│   │   │   ├── patients.py           # 病人查询接口
│   │   │   └── ocr.py                # 化验单 OCR 接口
│   │   ├── services/
│   │   │   ├── rag_service.py        # RAG 检索服务
│   │   │   ├── diagnosis_service.py  # DL 诊断推理服务
│   │   │   ├── patient_service.py    # 病人 CRUD 服务
│   │   │   └── ocr_service.py        # OCR 引擎服务
│   │   ├── ml/
│   │   │   ├── embedding.py          # Transformer 文本向量化
│   │   │   ├── diagnosis_model.py    # MLP 诊断模型定义
│   │   │   ├── data_generator.py     # 模拟医疗数据生成器
│   │   │   └── train.py              # 模型训练脚本
│   │   ├── models/
│   │   │   └── patient.py            # SQLAlchemy ORM 模型
│   │   └── schemas/
│   │       └── diagnosis.py          # Pydantic 请求/响应 Schema
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                         # React + TypeScript 前端
│   ├── src/
│   │   ├── main.tsx                  # 入口
│   │   ├── App.tsx                   # 路由
│   │   ├── App.css                   # 全局样式（医疗主题）
│   │   ├── pages/
│   │   │   ├── DiagnosisPage.tsx     # 诊断录入页（左-右两栏）
│   │   │   └── ResultPage.tsx        # 独立结果页
│   │   ├── components/
│   │   │   ├── PatientForm.tsx       # 患者基本信息表单
│   │   │   ├── LabReportForm.tsx     # 化验单录入 + OCR 上传
│   │   │   ├── DiagnosisResult.tsx   # AI 诊断结果卡片
│   │   │   └── SimilarCases.tsx      # 相似病例列表
│   │   ├── services/
│   │   │   └── api.ts                # Axios API 封装
│   │   └── types/
│   │       └── diagnosis.ts          # TypeScript 类型定义
│   ├── vite.config.ts
│   ├── nginx.conf
│   └── Dockerfile
├── data/
│   ├── init.sql                      # 数据库 DDL（含 pgvector 扩展）
│   ├── mock_patients.json            # 500 条模拟病例
│   └── models/                       # 训练好的模型权重
├── docs/
│   └── superpowers/                  # 设计文档 + 实现计划
├── docker-compose.yml                # Docker 编排文件
├── .env.example                      # 环境变量模板
└── README.md
```

---

## ⚙️ 配置说明

通过 `.env` 文件或 `docker-compose.yml` 中的 `environment` 配置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DATABASE_URL` | `postgresql+asyncpg://hospital:hospital@db:5432/hospital` | 数据库连接串 |
| `EMBEDDING_MODEL` | `shibing624/text2vec-base-chinese` | 文本 Embedding 模型 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace 镜像（国内加速） |
| `DEEPSEEK_API_KEY` | — | DeepSeek API Key（OCR 结构化必需） |
| `BAIDU_OCR_APP_ID` | — | 百度 OCR App ID（可选） |
| `BAIDU_OCR_API_KEY` | — | 百度 OCR API Key（可选） |
| `BAIDU_OCR_SECRET_KEY` | — | 百度 OCR Secret Key（可选） |

> 🔧 更换 Embedding 模型只需修改 `EMBEDDING_MODEL` 环境变量，例如换成医学专用模型 `CatherineWong/MedicineBERT`。

---

## 🔄 数据流

### 诊断流程

1. 用户在**前端**填写患者信息、症状描述、化验单数据
2. 前端发送 `POST /api/diagnose` 请求
3. 后端**并行执行**两个通道：
   - **RAG 通道**：症状文本 → Transformer Embedding → pgvector 余弦检索 → Top-5 相似病例
   - **DL 通道**：化验单数值 → Tensor[1, 20] → MLP 模型 → Softmax → Top-3 疾病 + 治疗建议
4. 合并两通道结果返回前端，同时异步存入数据库

### OCR 流程

1. 用户上传化验单照片 → `POST /api/ocr`
2. 百度 OCR 精确版提取文字（失败则降级 EasyOCR）
3. DeepSeek API 将 OCR 文本结构化为 21 项化验指标 JSON
4. 前端自动将识别结果填入 LabReportForm 对应字段

---

## 🔮 未来计划

- [ ] 接入真实医疗数据集（如 MIMIC）替换模拟数据
- [ ] 支持更多疾病类型（当前 10 种）
- [ ] 换用医学专用 Embedding 模型
- [ ] 模型量化/蒸馏以减小体积
- [ ] 添加用户认证与权限管理
- [ ] 诊断历史记录与趋势分析
- [ ] 化验单模板匹配优化 OCR 准确率
- [ ] 国际化（i18n）支持

---

## 📄 许可证

MIT License

---

*Built with ❤️ using FastAPI, React, PyTorch, and pgvector.*
