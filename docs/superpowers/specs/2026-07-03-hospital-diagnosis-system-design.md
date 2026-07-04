# 智能医疗诊断辅助系统 — 设计文档

**日期**: 2026-07-03
**目标**: 学习项目，综合运用 Transformer、RAG、机器学习、深度学习、Docker 等技术

---

## 1. 项目概述

构建一个医院诊断辅助系统，病人输入化验单数据和症状描述，系统通过 RAG 检索相似历史病例、通过深度学习模型预测病症，最终给出诊断建议和治疗方案。

### 技术栈

| 层 | 技术 | 用途 |
|---|---|---|
| 前端 | React + TypeScript | 病人信息录入、化验单上传、结果可视化 |
| 后端 | FastAPI | API 服务、模型推理调度 |
| 数据库 | PostgreSQL + pgvector | 结构化数据存储 + 向量相似度检索 |
| Embedding | 预训练中文模型（text2vec-base-chinese） | 症状描述 → 向量 |
| RAG 检索 | pgvector 余弦相似度 | 检索相似病人案例 |
| 诊断模型 | PyTorch 自建 MLP 网络 | 化验单数据 → 病症预测 |
| OCR | 百度 OCR（主）+ EasyOCR（降级）+ DeepSeek | 化验单图片 → 结构化数据 |
| 部署 | Docker + Docker Compose | 容器化一键启动 |

### 架构模式

**模块化分层架构（方案 A）**：单一 FastAPI 应用，内部按职责拆分为 `rag/`、`diagnosis/`、`data/` 模块。所有服务通过 Docker Compose 编排。后续可演进为微服务。

---

## 2. 项目结构

```
hospital_system/
├── frontend/                    # React + TypeScript
│   ├── src/
│   │   ├── components/          # 可复用组件
│   │   ├── pages/               # 页面（诊断录入、结果展示）
│   │   ├── services/            # API 调用封装
│   │   └── types/               # TypeScript 类型定义
│   └── Dockerfile
│
├── backend/                     # FastAPI
│   ├── app/
│   │   ├── main.py              # FastAPI 入口
│   │   ├── api/                 # 路由层（endpoints）
│   │   ├── models/              # SQLAlchemy 数据库模型
│   │   ├── schemas/             # Pydantic 请求/响应 schema
│   │   ├── services/            # 业务逻辑层
│   │   │   ├── rag_service.py   # RAG 检索 + Transformer embedding
│   │   │   ├── diagnosis_service.py  # DL 模型推理
│   │   │   ├── patient_service.py    # 病人数据管理
│   │   │   └── ocr_service.py   # 化验单 OCR 识别（百度 + EasyOCR）
│   │   └── ml/                  # ML/DL 模型
│   │       ├── embedding.py     # Transformer 文本向量化
│   │       ├── diagnosis_model.py    # 诊断预测模型定义
│   │       └── train.py         # 模型训练脚本
│   ├── requirements.txt
│   └── Dockerfile
│
├── data/                        # 模拟数据 + 模型权重
│   ├── init.sql                 # 数据库初始化 + 模拟病例
│   ├── mock_patients.json       # 模拟病人数据
│   └── models/                  # 保存训练好的模型权重
│
├── docker-compose.yml
└── docs/
```

---

## 3. 数据流

```
病人输入 ──▶ React 前端 ──▶ POST /api/diagnose ──▶ FastAPI
                                                        │
                                         ┌──────────────┼──────────────┐
                                         ▼                              ▼
                                   RAG 检索通道                   诊断预测通道
                                         │                              │
                                   ① Transformer                  ④ 预处理化验单
                                   文字 → embedding                  → 数值特征
                                         │                              │
                                   ② pgvector                     ⑤ MLP 模型
                                   余弦相似度检索                    预测病症
                                         │                              │
                                   ③ Top-K 相似病例              ⑥ 预测结果 + 建议
                                         │                              │
                                         └──────────────┬───────────────┘
                                                        ▼
                                               ⑦ 合并结果返回前端
                                          （相似案例参考 + AI 预测诊断）

另外，前端支持拍照/上传化验单图片，通过 POST /api/ocr 自动识别填入表单：

化验单图片 ──▶ POST /api/ocr ──▶ 百度 OCR（主引擎）
                                      │
                                      ├── 成功 → DeepSeek 结构化 → 21 项数值
                                      │
                                      └── 失败 → EasyOCR（降级）→ DeepSeek 结构化
```

两个通道**并行执行**，最终结果合并返回。

---

## 4. RAG 模块

### 流程

1. **文本预处理**：清洗病人症状描述
2. **Transformer Embedding**：使用预训练中文模型（默认 `shibing624/text2vec-base-chinese`）将文本转为 768 维向量
3. **向量检索**：在 pgvector 中执行余弦相似度查询，返回 Top-5 相似病例
4. **结果增强**：将相似病例的诊断和治疗方案作为参考上下文

### Embedding 模型

默认使用 `text2vec-base-chinese`（轻量、下载快），后续可替换为医学专用模型（如 BenTsao）。模型通过接口抽象，改一行配置即可切换。

### 数据库向量表

```sql
CREATE TABLE patients (
    id SERIAL PRIMARY KEY,
    name TEXT,
    age INT,
    gender TEXT,
    symptom_description TEXT,
    symptom_embedding vector(768),
    diagnosis TEXT,
    treatment TEXT,
    lab_data JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX ON patients USING ivfflat (symptom_embedding vector_cosine_ops);
```

---

## 5. 诊断预测模块（DL 模型）

### 输入与输出

- **输入**：结构化化验单数据（20-30 个数值指标，如白细胞、CRP、体温、血压等）
- **输出**：Top-3 疾病预测 + 概率分布 + 治疗建议

### 模型结构（MLP）

```
Input(30) → BatchNorm → Linear(128) → ReLU → Dropout(0.3)
         → Linear(64) → ReLU → Dropout(0.3)
         → Linear(32) → ReLU
         → Linear(N_diseases) → Softmax
```

### 设计决策

- **为什么用 MLP 而不是 Transformer？** 化验单是结构化数值数据，没有序列关系。MLP 对表格型数据效果好且训练快。Transformer 在系统中负责处理非结构化文本（RAG 的 embedding 环节），各司其职。
- **为什么用 Softmax 而不是直接回归？** 诊断本质是多分类问题（给定指标 → 判断属于哪种疾病），Softmax 输出概率分布便于解释。

### 训练数据

- 基于医学知识生成 500-1000 条模拟病例
- 数据增强（加噪声、微调数值）
- 后续可接入真实数据集（接口预留）

---

## 6. API 设计

### 接口列表

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/diagnose` | 核心接口：提交症状+化验单，返回诊断 |
| POST | `/api/ocr` | 化验单图片上传，OCR 识别 + AI 结构化，返回 21 项指标 |
| GET | `/api/patients/{id}` | 查询历史病人 |
| GET | `/api/patients` | 病人列表（分页） |
| GET | `/api/diseases` | 支持的疾病列表 |
| GET | `/api/health` | 健康检查 |

### POST /api/diagnose

**Request**:
```json
{
  "name": "张三",
  "age": 45,
  "gender": "male",
  "symptom_description": "头痛发烧三天，咳嗽有黄痰，胸闷",
  "lab_report": {
    "wbc": 12.5,
    "neutrophil_pct": 85,
    "crp": 45,
    "temperature": 38.6,
    "systolic_bp": 130,
    "diastolic_bp": 85
  }
}
```

**Response**:
```json
{
  "diagnosis": {
    "top_prediction": "细菌性肺炎",
    "confidence": 0.783,
    "top3": [
      {"disease": "细菌性肺炎", "probability": 0.783},
      {"disease": "病毒性感冒", "probability": 0.121},
      {"disease": "急性支气管炎", "probability": 0.062}
    ],
    "treatment_suggestion": "建议使用抗生素治疗..."
  },
  "similar_cases": [
    {
      "similarity": 0.92,
      "symptom_description": "咳嗽咳痰发烧胸闷3天",
      "diagnosis": "细菌性肺炎",
      "treatment": "头孢类抗生素 + 退烧药..."
    }
  ]
}
```

---

### POST /api/ocr

化验单图片上传识别，支持百度 OCR（云端精确版）和 EasyOCR（本地降级引擎）。

**Request**: `multipart/form-data`，字段 `file`（JPG/PNG/WebP，最大 10MB）

**Response**:
```json
{
  "lab_data": {
    "wbc": 12.5, "neutrophil_pct": 85, "crp": 45, "temperature": 38.6,
    "...": "...其余 17 项指标"
  },
  "raw_text": "白细胞计数 12.5 ×10⁹/L 中性粒细胞百分比 85%..."
}
```

**降级策略**：百度 OCR 调用失败时自动切换 EasyOCR，确保服务可用性。

---

## 7. 前端页面

### 页面 1：诊断录入页（首页）

- 基本信息表单（姓名、年龄、性别）
- 症状描述文本区
- 化验单数据录入区（可折叠展开，支持分组展示）
- 拍照/上传化验单图片自动 OCR 识别填入
- "开始诊断"提交按钮，调用 POST /api/diagnose

### 页面 2：诊断结果页

- AI 诊断结果卡片（Top-3 预测 + 置信度，颜色编码风险等级）
- 治疗建议展示
- 相似历史病例列表（RAG 检索结果）

---

## 8. Docker 部署

```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: hospital
      POSTGRES_PASSWORD: hospital
      POSTGRES_DB: hospital
    ports: ["5432:5432"]
    volumes:
      - ./data/init.sql:/docker-entrypoint-initdb.d/init.sql
      - pgdata:/var/lib/postgresql/data

  backend:
    build: ./backend
    ports: ["8000:8000"]
    depends_on: [db]
    environment:
      DATABASE_URL: postgresql://hospital:hospital@db:5432/hospital

  frontend:
    build: ./frontend
    ports: ["3000:3000"]
    depends_on: [backend]

volumes:
  pgdata:
```

一键启动：`docker compose up --build`

---

## 9. 技术要点汇总

| 技术点 | 在系统中的体现 |
|---|---|
| **Transformer** | RAG 模块：预训练中文模型将症状描述转为 embedding |
| **RAG** | 检索相似病例作为诊断参考上下文 |
| **机器学习** | 数据预处理、特征工程、模型评估指标 |
| **深度学习** | PyTorch MLP 模型从化验单数据预测疾病 |
| **Docker** | 前后端 + 数据库容器化，docker-compose 编排 |
| **FastAPI** | 异步 API 服务、Pydantic 数据校验、自动文档 |
| **React + TS** | 前端表单、结果可视化、类型安全 |
| **OCR 识别** | 百度 OCR（云端）+ EasyOCR（本地降级）+ DeepSeek 结构化解析 |
```

