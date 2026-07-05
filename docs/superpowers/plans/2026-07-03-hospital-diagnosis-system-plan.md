# 智能医疗诊断辅助系统 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 构建一个医院诊断辅助系统，病人输入化验单和症状描述，RAG 检索相似病例 + DL 模型预测病症，给出诊断建议。

**Architecture:** 模块化分层 FastAPI 单体应用，React + TypeScript 前端，PostgreSQL + pgvector 存储，Docker Compose 编排。RAG 通道和诊断预测通道在 /api/diagnose 中并行执行后合并结果。化验单支持图片上传 OCR 识别（百度 OCR 主引擎 + EasyOCR 降级 + DeepSeek 结构化）。

**Tech Stack:** React 18 + TypeScript 5, FastAPI (Python 3.11), PostgreSQL 16 + pgvector, PyTorch 2.x, sentence-transformers, SQLAlchemy 2.x, 百度 OCR / EasyOCR, DeepSeek API, Docker + Docker Compose

## Global Constraints

- Python 3.11+, Node.js 20+
- 前端默认运行在 localhost:3000，后端 localhost:8000
- 所有 API 路径以 `/api/` 为前缀
- Embedding 模型默认 `BAAI/bge-base-zh-v1.5`（768 维）
- 诊断模型为 MLP 结构：输入 20 维化验指标 → 输出 10 种疾病概率
- 编码规范：英文变量名，注释用中文
- RAG 数据：22 万条真实医疗对话 → 清洗后 1.8 万条，191 种疾病（已替换合成数据）
- OCR 引擎：百度 OCR 精确版（主）+ EasyOCR（降级）+ DeepSeek（结构化）

---

### Task 1: 项目根目录搭建

**Files:**
- Create: `docker-compose.yml`
- Create: `.gitignore`
- Create: `data/models/.gitkeep`

**Interfaces:**
- Consumes: nothing
- Produces: `docker-compose.yml` 中定义的服务名 `db`, `backend`, `frontend` 供后续 Task 使用

- [x] **Step 1: 创建 .gitignore**

```bash
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
node_modules/
dist/
.env
*.pt
*.pth
data/models/*
!data/models/.gitkeep
.DS_Store
EOF
```

- [x] **Step 2: 创建 docker-compose.yml**

```yaml
version: "3.9"

services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: hospital
      POSTGRES_PASSWORD: hospital
      POSTGRES_DB: hospital
    ports:
      - "5432:5432"
    volumes:
      - ./data/init.sql:/docker-entrypoint-initdb.d/01-init.sql
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U hospital"]
      interval: 5s
      timeout: 5s
      retries: 5

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    depends_on:
      db:
        condition: service_healthy
    environment:
      DATABASE_URL: postgresql+asyncpg://hospital:hospital@db:5432/hospital
      EMBEDDING_MODEL: shibing624/text2vec-base-chinese
      MODEL_PATH: /app/data/models/diagnosis_model.pt

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend

volumes:
  pgdata:
```

- [x] **Step 3: 创建 models 目录占位文件**

```bash
mkdir -p data/models
touch data/models/.gitkeep
```

- [x] **Step 4: 验证 docker-compose 语法**

Run: `docker compose config`
Expected: 输出完整的 compose 配置，无错误

- [x] **Step 5: Commit**

```bash
git add .gitignore docker-compose.yml data/
git commit -m "feat: add project scaffolding with docker-compose"
```

---

### Task 2: 数据库初始化 SQL + SQLAlchemy 模型

**Files:**
- Create: `data/init.sql`
- Create: `backend/requirements.txt`
- Create: `backend/app/__init__.py`
- Create: `backend/app/config.py`
- Create: `backend/app/database.py`
- Create: `backend/app/models/__init__.py`
- Create: `backend/app/models/patient.py`

**Interfaces:**
- Consumes: `DATABASE_URL` 环境变量
- Produces: `Patient` SQLAlchemy model（`id`, `name`, `age`, `gender`, `symptom_description`, `diagnosis`, `treatment`, `lab_data`, `created_at`），`get_db()` 异步 session 生成器，`init_db()` 建表函数

- [x] **Step 1: 创建 data/init.sql**

```sql
-- 启用 pgvector 扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 创建病人表
CREATE TABLE IF NOT EXISTS patients (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    age INT NOT NULL,
    gender VARCHAR(10) NOT NULL,
    symptom_description TEXT NOT NULL,
    symptom_embedding vector(768),
    diagnosis VARCHAR(200),
    treatment TEXT,
    lab_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 向量索引（IVFFlat 需要表中有一定数据量才能创建，先创建后续再用）
-- CREATE INDEX ON patients USING ivfflat (symptom_embedding vector_cosine_ops) WITH (lists = 100);
```

- [x] **Step 2: 创建 backend/requirements.txt**

```
fastapi==0.111.0
uvicorn[standard]==0.30.1
sqlalchemy[asyncio]==2.0.30
asyncpg==0.29.0
pgvector==0.3.1
pydantic==2.7.4
pydantic-settings==2.3.4
torch==2.3.1
sentence-transformers==3.0.1
numpy==1.26.4
python-multipart==0.0.9
```

- [x] **Step 3: 创建 backend/app/config.py**

```python
"""应用配置"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://hospital:hospital@localhost:5432/hospital"
    embedding_model: str = "shibing624/text2vec-base-chinese"
    model_path: str = "data/models/diagnosis_model.pt"
    top_k_similar: int = 5  # RAG 检索返回的相似病例数

    class Config:
        env_file = ".env"


settings = Settings()
```

- [x] **Step 4: 创建 backend/app/database.py**

```python
"""数据库连接管理"""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

engine = create_async_engine(settings.database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    """FastAPI 依赖注入：获取数据库 session"""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """创建所有表"""
    from app.models.patient import Patient  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
```

- [x] **Step 5: 创建 backend/app/models/patient.py**

```python
"""病人数据库模型"""

from datetime import datetime

from sqlalchemy import Integer, String, Text, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from app.database import Base


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    gender: Mapped[str] = mapped_column(String(10), nullable=False)
    symptom_description: Mapped[str] = mapped_column(Text, nullable=False)
    symptom_embedding = mapped_column(Vector(768), nullable=True)
    diagnosis: Mapped[str] = mapped_column(String(200), nullable=True)
    treatment: Mapped[str] = mapped_column(Text, nullable=True)
    lab_data: Mapped[dict] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<Patient(id={self.id}, name='{self.name}', diagnosis='{self.diagnosis}')>"
```

- [x] **Step 6: 验证导入**

Run: `cd backend && python -c "from app.models.patient import Patient; print('Patient model OK:', Patient.__tablename__)"`
Expected: `Patient model OK: patients`

- [x] **Step 7: Commit**

```bash
git add data/init.sql backend/requirements.txt backend/app/
git commit -m "feat: add database config, SQLAlchemy models, and init SQL"
```

---

### Task 3: Pydantic Schemas（请求/响应模型）

**Files:**
- Create: `backend/app/schemas/__init__.py`
- Create: `backend/app/schemas/diagnosis.py`

**Interfaces:**
- Consumes: `Patient` model (字段名: `id`, `name`, `age`, `gender`, `symptom_description`, `diagnosis`, `treatment`, `lab_data`, `symptom_embedding`)
- Produces: `DiagnoseRequest`（`name: str`, `age: int`, `gender: str`, `symptom_description: str`, `lab_report: dict`），`DiagnosisItem`，`DiagnosisResult`，`SimilarCase`，`DiagnoseResponse`

- [x] **Step 1: 创建 backend/app/schemas/diagnosis.py**

```python
"""诊断相关 Pydantic schemas"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Request ──────────────────────────────────────────────

class LabReport(BaseModel):
    """化验单标准指标（20 项）"""
    wbc: float = Field(..., description="白细胞计数 (×10⁹/L)")
    neutrophil_pct: float = Field(..., description="中性粒细胞百分比 (%)")
    lymphocyte_pct: float = Field(..., description="淋巴细胞百分比 (%)")
    crp: float = Field(..., description="C反应蛋白 (mg/L)")
    temperature: float = Field(..., description="体温 (°C)")
    systolic_bp: float = Field(..., description="收缩压 (mmHg)")
    diastolic_bp: float = Field(..., description="舒张压 (mmHg)")
    heart_rate: float = Field(..., description="心率 (次/分)")
    respiratory_rate: float = Field(..., description="呼吸频率 (次/分)")
    spo2: float = Field(..., description="血氧饱和度 (%)")
    rbc: float = Field(..., description="红细胞计数 (×10¹²/L)")
    hemoglobin: float = Field(..., description="血红蛋白 (g/L)")
    hematocrit: float = Field(..., description="血细胞比容 (%)")
    platelet: float = Field(..., description="血小板计数 (×10⁹/L)")
    glucose: float = Field(..., description="空腹血糖 (mmol/L)")
    creatinine: float = Field(..., description="肌酐 (μmol/L)")
    bun: float = Field(..., description="尿素氮 (mmol/L)")
    alt: float = Field(..., description="谷丙转氨酶 (U/L)")
    ast: float = Field(..., description="谷草转氨酶 (U/L)")
    total_cholesterol: float = Field(..., description="总胆固醇 (mmol/L)")
    triglycerides: float = Field(..., description="甘油三酯 (mmol/L)")


class DiagnoseRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., ge=0, le=150)
    gender: str = Field(..., pattern="^(male|female|other)$")
    symptom_description: str = Field(..., min_length=2, max_length=2000)
    lab_report: LabReport


# ── Response ─────────────────────────────────────────────

class DiagnosisItem(BaseModel):
    disease: str
    probability: float


class DiagnosisResult(BaseModel):
    top_prediction: str
    confidence: float
    top3: list[DiagnosisItem]
    treatment_suggestion: str


class SimilarCase(BaseModel):
    id: int
    similarity: float
    symptom_description: str
    diagnosis: str
    treatment: str


class DiagnoseResponse(BaseModel):
    diagnosis: DiagnosisResult
    similar_cases: list[SimilarCase]


class DiseaseInfo(BaseModel):
    name: str
    description: str


class PatientSummary(BaseModel):
    id: int
    name: str
    age: int
    gender: str
    diagnosis: Optional[str] = None
    created_at: Optional[datetime] = None


class PatientDetail(PatientSummary):
    symptom_description: str
    treatment: Optional[str] = None
    lab_data: Optional[dict] = None
```

- [x] **Step 2: 验证 schema 导入**

Run: `cd backend && python -c "from app.schemas.diagnosis import DiagnoseRequest, DiagnoseResponse, LabReport; print('Schemas OK')"`
Expected: `Schemas OK`

- [x] **Step 3: Commit**

```bash
git add backend/app/schemas/
git commit -m "feat: add Pydantic request/response schemas"
```

---

### Task 4: ML — Transformer Embedding 模块

**Files:**
- Create: `backend/app/ml/__init__.py`
- Create: `backend/app/ml/embedding.py`

**Interfaces:**
- Consumes: `settings.embedding_model`（默认 `shibing624/text2vec-base-chinese`）
- Produces: `EmbeddingModel` 类，方法 `encode(text: str) -> list[float]`（768 维），`encode_batch(texts: list[str]) -> list[list[float]]`

- [x] **Step 1: 创建 backend/app/ml/embedding.py**

```python
"""Transformer 文本向量化模块"""

from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingModel:
    """封装 SentenceTransformer，将中文症状描述转为 768 维向量"""

    def __init__(self, model_name: str | None = None):
        model_name = model_name or settings.embedding_model
        self._model = SentenceTransformer(model_name)

    def encode(self, text: str) -> list[float]:
        """单条文本 → 向量"""
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """批量文本 → 向量列表"""
        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        return embeddings.tolist()


# 全局单例（模型只需加载一次）
embedding_model = EmbeddingModel()
```

- [x] **Step 2: 验证 embedding 模型首次加载（会下载模型，约 400MB）**

Run: `cd backend && python -c "from app.ml.embedding import embedding_model; v = embedding_model.encode('头痛发烧咳嗽'); print(f'Vector dim: {len(v)}, first 5: {v[:5]}')"`
Expected: `Vector dim: 768, first 5: [0.xxx, ...]`

- [x] **Step 3: Commit**

```bash
git add backend/app/ml/__init__.py backend/app/ml/embedding.py
git commit -m "feat: add Transformer embedding module with text2vec-base-chinese"
```

---

### Task 5: ML — 诊断预测模型定义

**Files:**
- Create: `backend/app/ml/diagnosis_model.py`

**Interfaces:**
- Consumes: 20 维化验指标向量
- Produces: `DiagnosisPredictor(nn.Module)`，构造函数参数 `input_dim=20, num_diseases=10`；
  方法 `forward(x) -> Tensor` 返回 softmax 概率；`DISEASE_LABELS` 常量列表；`LAB_FEATURE_NAMES` 常量列表

- [x] **Step 1: 创建 backend/app/ml/diagnosis_model.py**

```python
"""诊断预测深度学习模型（MLP）"""

import torch
import torch.nn as nn

# 疾病标签（10 种）
DISEASE_LABELS = [
    "细菌性肺炎",
    "病毒性感冒",
    "急性支气管炎",
    "高血压",
    "2型糖尿病",
    "冠心病",
    "慢性胃炎",
    "尿路感染",
    "缺铁性贫血",
    "甲状腺功能亢进",
]

# 化验指标名称（20 维，对应 LabReport schema）
LAB_FEATURE_NAMES = [
    "wbc", "neutrophil_pct", "lymphocyte_pct", "crp",
    "temperature", "systolic_bp", "diastolic_bp", "heart_rate",
    "respiratory_rate", "spo2", "rbc", "hemoglobin",
    "hematocrit", "platelet", "glucose", "creatinine",
    "bun", "alt", "ast", "total_cholesterol",
]


class DiagnosisPredictor(nn.Module):
    """MLP 诊断预测模型"""

    def __init__(self, input_dim: int = 20, num_diseases: int = 10, dropout: float = 0.3):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, num_diseases)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.batch_norm(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.output(x)
        return torch.softmax(x, dim=-1)


def create_model(model_path: str | None = None) -> DiagnosisPredictor:
    """工厂函数：创建并加载训练好的模型"""
    from app.config import settings

    model = DiagnosisPredictor(input_dim=20, num_diseases=len(DISEASE_LABELS))
    path = model_path or settings.model_path
    try:
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
    except FileNotFoundError:
        pass  # 未训练时返回随机权重模型
    return model
```

- [x] **Step 2: 验证模型结构**

Run: `cd backend && python -c "from app.ml.diagnosis_model import DiagnosisPredictor, DISEASE_LABELS; m = DiagnosisPredictor(); x = m(torch.randn(2, 20)); print(f'Input: (2,20) → Output: {x.shape}, sum~1: {x.sum(dim=-1).tolist()}')"`
Expected: `Input: (2,20) → Output: torch.Size([2, 10]), sum~1: [1.0, 1.0]`

- [x] **Step 3: Commit**

```bash
git add backend/app/ml/diagnosis_model.py
git commit -m "feat: add MLP diagnosis prediction model definition"
```

---

### Task 6: ML — 模拟数据生成 + 模型训练

**Files:**
- Create: `backend/app/ml/data_generator.py`
- Create: `backend/app/ml/train.py`
- Create: `data/mock_patients.json`

**Interfaces:**
- Consumes: `DISEASE_LABELS`, `LAB_FEATURE_NAMES` 来自 `diagnosis_model.py`；`EmbeddingModel.encode_batch()` 来自 `embedding.py`；`Patient` model；`settings.database_url`
- Produces: `generate_mock_data(n: int) -> list[dict]`；`train_model()` 函数；`data/models/diagnosis_model.pt` 训练好的权重文件

- [x] **Step 1: 创建 backend/app/ml/data_generator.py**

```python
"""模拟医疗数据生成器

基于医学知识生成各疾病的典型化验指标模式，加入随机噪声模拟真实数据。
"""

import random
import numpy as np
from app.ml.diagnosis_model import DISEASE_LABELS, LAB_FEATURE_NAMES

# 各疾病的典型化验指标模式 (正常值 + 疾病特征偏移)
DISEASE_PATTERNS = {
    "细菌性肺炎": {  # 感染性指标显著升高
        "wbc": (15.0, 3.0), "neutrophil_pct": (88, 5), "lymphocyte_pct": (10, 3),
        "crp": (60, 20), "temperature": (39.0, 0.5), "heart_rate": (100, 10),
        "respiratory_rate": (24, 3), "spo2": (93, 2),
    },
    "病毒性感冒": {  # 轻度异常
        "wbc": (7.0, 2.0), "neutrophil_pct": (60, 8), "lymphocyte_pct": (35, 8),
        "crp": (10, 5), "temperature": (38.0, 0.5), "heart_rate": (85, 10),
        "respiratory_rate": (18, 2), "spo2": (97, 2),
    },
    "急性支气管炎": {  # 中度呼吸道症状
        "wbc": (10.0, 2.5), "neutrophil_pct": (72, 8), "lymphocyte_pct": (22, 5),
        "crp": (25, 10), "temperature": (37.8, 0.5), "heart_rate": (90, 10),
        "respiratory_rate": (20, 3), "spo2": (95, 2),
    },
    "高血压": {  # 血压显著升高，其他基本正常
        "systolic_bp": (160, 10), "diastolic_bp": (100, 8),
        "heart_rate": (80, 10), "total_cholesterol": (5.8, 0.8), "triglycerides": (2.0, 0.5),
    },
    "2型糖尿病": {  # 血糖、血脂异常
        "glucose": (9.0, 2.0), "total_cholesterol": (5.5, 0.8),
        "triglycerides": (2.2, 0.6), "bun": (7.0, 1.5), "creatinine": (90, 15),
    },
    "冠心病": {  # 心血管指标异常
        "heart_rate": (90, 12), "systolic_bp": (150, 12), "diastolic_bp": (95, 8),
        "total_cholesterol": (6.2, 1.0), "triglycerides": (2.5, 0.7),
        "glucose": (6.0, 1.0),
    },
    "慢性胃炎": {  # 轻度血液指标变化
        "rbc": (4.0, 0.5), "hemoglobin": (110, 15), "hematocrit": (35, 5),
        "wbc": (8.5, 2.0),
    },
    "尿路感染": {  # 感染指标升高
        "wbc": (13.0, 3.0), "neutrophil_pct": (82, 6), "crp": (35, 15),
        "temperature": (38.2, 0.5), "heart_rate": (92, 10),
    },
    "缺铁性贫血": {  # 红细胞相关指标显著降低
        "rbc": (3.0, 0.5), "hemoglobin": (80, 15), "hematocrit": (28, 4),
        "platelet": (350, 50), "heart_rate": (95, 10), "spo2": (96, 2),
    },
    "甲状腺功能亢进": {  # 代谢率升高
        "heart_rate": (105, 12), "temperature": (37.5, 0.3),
        "systolic_bp": (145, 10), "diastolic_bp": (85, 7),
        "glucose": (6.5, 1.0), "total_cholesterol": (3.5, 0.5),
    },
}

# 正常默认值（当某疾病无特定模式时使用）
NORMAL_DEFAULTS = {
    "wbc": (6.5, 1.5), "neutrophil_pct": (58, 8), "lymphocyte_pct": (33, 6),
    "crp": (5, 3), "temperature": (36.8, 0.3), "systolic_bp": (120, 8),
    "diastolic_bp": (78, 6), "heart_rate": (72, 8), "respiratory_rate": (16, 2),
    "spo2": (98, 1), "rbc": (4.8, 0.5), "hemoglobin": (140, 12),
    "hematocrit": (42, 4), "platelet": (250, 50), "glucose": (5.0, 0.8),
    "creatinine": (75, 12), "bun": (4.5, 1.2), "alt": (22, 8),
    "ast": (20, 7), "total_cholesterol": (4.5, 0.6), "triglycerides": (1.2, 0.4),
}

SYMPTOM_TEMPLATES = {
    "细菌性肺炎": [
        "发烧{temp}度，咳嗽有黄痰，胸闷气短，全身乏力",
        "高热不退，咳脓痰，胸痛，呼吸急促",
        "持续发热伴寒战，咳嗽剧烈，痰中带血丝",
    ],
    "病毒性感冒": [
        "鼻塞流涕，打喷嚏，喉咙痛，轻度发热{temp}度",
        "全身酸痛乏力，头痛，流清涕，轻微咳嗽",
        "畏寒发热，咽痛，鼻塞，食欲不振",
    ],
    "急性支气管炎": [
        "咳嗽频繁，有白色黏痰，胸闷不适，低热",
        "刺激性干咳，夜间加重，咽部不适",
        "感冒后持续咳嗽，咳痰，轻度呼吸困难",
    ],
    "高血压": [
        "头晕头痛，颈项僵硬，心悸，失眠多梦",
        "后脑勺胀痛，眩晕耳鸣，注意力不集中",
        "无明显不适，体检发现血压偏高",
    ],
    "2型糖尿病": [
        "口渴多饮，多尿，体重下降，疲乏无力",
        "视物模糊，四肢麻木，伤口愈合缓慢",
        "经常饥饿，食量增大但体重减轻",
    ],
    "冠心病": [
        "活动后胸闷胸痛，气短，心悸，左肩放射痛",
        "心前区压榨感，持续数分钟，休息后缓解",
        "夜间阵发性呼吸困难，下肢轻度水肿",
    ],
    "慢性胃炎": [
        "上腹部隐痛，饭后饱胀，反酸嗳气，食欲减退",
        "胃部灼烧感，恶心，空腹时疼痛加重",
        "消化不良，腹胀腹泻交替，口苦口干",
    ],
    "尿路感染": [
        "尿频尿急尿痛，下腹部不适，腰酸",
        "排尿灼热感，尿液混浊，低热",
        "腰痛发热，尿频加重，全身不适",
    ],
    "缺铁性贫血": [
        "面色苍白，头晕乏力，心慌气短，注意力不集中",
        "指甲脆薄易裂，口角炎，异食癖",
        "活动后心悸加重，耳鸣，月经量多（女性）",
    ],
    "甲状腺功能亢进": [
        "心悸手抖，怕热多汗，食欲亢进但体重下降",
        "情绪易激动，失眠，大便次数增多",
        "颈部增粗，眼球突出，乏力消瘦",
    ],
}

TREATMENTS = {
    "细菌性肺炎": "抗生素治疗（如头孢曲松+阿奇霉素），退热，氧疗，充分休息，多饮水",
    "病毒性感冒": "对症支持治疗：退热药（对乙酰氨基酚），抗组胺药，充分休息，补充维生素C",
    "急性支气管炎": "止咳祛痰（氨溴索），支气管扩张剂，雾化吸入，多饮水，避免刺激",
    "高血压": "降压药（氨氯地平/缬沙坦），低盐饮食，规律运动，控制体重，监测血压",
    "2型糖尿病": "降糖药（二甲双胍），饮食控制，规律运动，血糖监测，糖尿病教育",
    "冠心病": "抗血小板药（阿司匹林），他汀类降脂药，硝酸酯类，控制危险因素",
    "慢性胃炎": "抑酸药（奥美拉唑），胃黏膜保护剂，规律饮食，避免辛辣刺激",
    "尿路感染": "抗生素（左氧氟沙星/呋喃妥因），多饮水，注意个人卫生",
    "缺铁性贫血": "补铁剂（硫酸亚铁），维生素C促进吸收，富含铁的食物，病因治疗",
    "甲状腺功能亢进": "抗甲状腺药（甲巯咪唑），β受体阻滞剂，定期复查甲功",
}


def _generate_value(feature: str, disease: str) -> float:
    """为指定疾病和指标生成带噪声的数值"""
    pattern = DISEASE_PATTERNS.get(disease, {})
    mean, std = pattern.get(feature, NORMAL_DEFAULTS[feature])
    val = np.random.normal(mean, std)
    return round(max(0.0, val), 2)


def generate_mock_data(n_per_disease: int = 50) -> list[dict]:
    """生成模拟病例数据，每种疾病 n_per_disease 条，总计约 500 条"""
    patients = []
    for disease in DISEASE_LABELS:
        for i in range(n_per_disease):
            gender = random.choice(["male", "female"])
            age = random.randint(18, 80)
            templates = SYMPTOM_TEMPLATES[disease]
            template = random.choice(templates)
            temp_val = round(random.uniform(36.5, 39.5), 1)
            symptom_text = template.format(temp=temp_val)

            lab_data = {feat: _generate_value(feat, disease) for feat in LAB_FEATURE_NAMES}
            # 确保体温与症状描述一致
            lab_data["temperature"] = temp_val

            patients.append({
                "name": f"模拟患者_{disease}_{i+1:03d}",
                "age": age,
                "gender": gender,
                "symptom_description": symptom_text,
                "diagnosis": disease,
                "treatment": TREATMENTS[disease],
                "lab_data": lab_data,
            })
    random.shuffle(patients)
    return patients
```

- [x] **Step 2: 创建 backend/app/ml/train.py**

```python
"""诊断模型训练脚本"""

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from app.ml.diagnosis_model import DiagnosisPredictor, DISEASE_LABELS, LAB_FEATURE_NAMES
from app.ml.data_generator import generate_mock_data


def prepare_data(patients: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """将病例数据转为训练用的特征张量和标签张量"""
    X = np.array([[p["lab_data"][feat] for feat in LAB_FEATURE_NAMES] for p in patients], dtype=np.float32)
    y = np.array([DISEASE_LABELS.index(p["diagnosis"]) for p in patients], dtype=np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)


def train(num_epochs: int = 50, batch_size: int = 32, lr: float = 0.001):
    """训练诊断预测模型并保存权重"""
    print("生成模拟训练数据...")
    patients = generate_mock_data(n_per_disease=50)
    X, y = prepare_data(patients)

    # 8:2 划分训练集和验证集
    n = len(patients)
    n_train = int(n * 0.8)
    indices = torch.randperm(n)
    X_train, y_train = X[indices[:n_train]], y[indices[:n_train]]
    X_val, y_val = X[indices[n_train:]], y[indices[n_train:]]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    model = DiagnosisPredictor(input_dim=20, num_diseases=len(DISEASE_LABELS))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    print(f"开始训练（{num_epochs} epochs, batch_size={batch_size}）...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        acc = correct / total

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss/len(train_loader):.4f} | "
                  f"Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "data/models/diagnosis_model.pt")

    print(f"\n训练完成！最佳验证准确率: {best_acc:.4f}")
    print(f"模型已保存至 data/models/diagnosis_model.pt")

    # 同时保存模拟数据为 JSON（供后续导入数据库使用）
    with open("data/mock_patients.json", "w", encoding="utf-8") as f:
        json.dump(patients, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    train()
```

- [x] **Step 3: 运行训练脚本**

Run: `cd backend && python -m app.ml.train`
Expected: 训练过程输出，最终 `Val Acc` > 0.8（模拟数据容易拟合），生成 `data/models/diagnosis_model.pt` 和 `data/mock_patients.json`

- [x] **Step 4: Commit**

```bash
git add backend/app/ml/data_generator.py backend/app/ml/train.py data/mock_patients.json data/models/.gitkeep
git commit -m "feat: add mock data generator and model training script"
```

---

### Task 7: RAG 服务 + Patient 服务

**Files:**
- Create: `backend/app/services/__init__.py`
- Create: `backend/app/services/rag_service.py`
- Create: `backend/app/services/patient_service.py`

**Interfaces:**
- Consumes: `embedding_model.encode()` from Task 4；`Patient` model from Task 2；`async_session` from Task 2
- Produces: `search_similar_patients(db, symptom_text, top_k) -> list[dict]`；`create_patient(db, data) -> Patient`；`get_patient(db, id) -> Patient`；`get_all_patients(db, skip, limit) -> list[Patient]`

- [x] **Step 1: 创建 backend/app/services/rag_service.py**

```python
"""RAG 检索服务：症状描述 → embedding → pgvector 相似度检索"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.embedding import embedding_model


async def search_similar_patients(
    db: AsyncSession,
    symptom_text: str,
    top_k: int = 5,
) -> list[dict]:
    """根据症状描述检索最相似的 K 个历史病例

    Args:
        db: 数据库 session
        symptom_text: 病人症状描述文本
        top_k: 返回的相似病例数量

    Returns:
        [{"id": 1, "similarity": 0.92, "symptom_description": "...",
          "diagnosis": "...", "treatment": "..."}, ...]
    """
    # 1. 将症状文字转为向量
    query_embedding = embedding_model.encode(symptom_text)

    # 2. 在 pgvector 中执行余弦相似度查询
    query = text("""
        SELECT
            id,
            symptom_description,
            diagnosis,
            treatment,
            1 - (symptom_embedding <=> :embedding) AS similarity
        FROM patients
        WHERE symptom_embedding IS NOT NULL
        ORDER BY symptom_embedding <=> :embedding
        LIMIT :top_k
    """)

    result = await db.execute(
        query,
        {"embedding": query_embedding, "top_k": top_k},
    )
    rows = result.fetchall()

    return [
        {
            "id": row.id,
            "similarity": round(float(row.similarity), 4),
            "symptom_description": row.symptom_description,
            "diagnosis": row.diagnosis or "未知",
            "treatment": row.treatment or "暂无方案",
        }
        for row in rows
    ]


async def generate_embedding_for_patient(
    db: AsyncSession,
    patient_id: int,
    symptom_text: str,
) -> None:
    """为指定病人生成并存储 symptom embedding"""
    embedding = embedding_model.encode(symptom_text)

    update_query = text("""
        UPDATE patients
        SET symptom_embedding = :embedding
        WHERE id = :patient_id
    """)
    await db.execute(
        update_query,
        {"embedding": embedding, "patient_id": patient_id},
    )
    await db.commit()
```

- [x] **Step 2: 创建 backend/app/services/patient_service.py**

```python
"""病人数据管理服务"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.patient import Patient


async def create_patient(
    db: AsyncSession,
    name: str,
    age: int,
    gender: str,
    symptom_description: str,
    diagnosis: str | None = None,
    treatment: str | None = None,
    lab_data: dict | None = None,
) -> Patient:
    """创建新病人记录"""
    patient = Patient(
        name=name,
        age=age,
        gender=gender,
        symptom_description=symptom_description,
        diagnosis=diagnosis,
        treatment=treatment,
        lab_data=lab_data,
    )
    db.add(patient)
    await db.commit()
    await db.refresh(patient)
    return patient


async def get_patient(db: AsyncSession, patient_id: int) -> Patient | None:
    """根据 ID 查询病人"""
    result = await db.execute(select(Patient).where(Patient.id == patient_id))
    return result.scalar_one_or_none()


async def get_all_patients(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 20,
) -> list[Patient]:
    """分页查询病人列表（按创建时间倒序）"""
    result = await db.execute(
        select(Patient)
        .order_by(Patient.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    return list(result.scalars().all())


async def import_mock_patients(
    db: AsyncSession,
    patients_data: list[dict],
) -> int:
    """批量导入模拟病人数据（含 embedding）"""
    from app.ml.embedding import embedding_model

    texts = [p["symptom_description"] for p in patients_data]
    embeddings = embedding_model.encode_batch(texts)

    count = 0
    for patient_data, embedding in zip(patients_data, embeddings):
        patient = Patient(
            name=patient_data["name"],
            age=patient_data["age"],
            gender=patient_data["gender"],
            symptom_description=patient_data["symptom_description"],
            symptom_embedding=embedding,
            diagnosis=patient_data["diagnosis"],
            treatment=patient_data["treatment"],
            lab_data=patient_data["lab_data"],
        )
        db.add(patient)
        count += 1

    await db.commit()
    return count
```

- [x] **Step 3: 验证导入**

Run: `cd backend && python -c "from app.services.rag_service import search_similar_patients; from app.services.patient_service import create_patient, get_patient; print('Services OK')"`
Expected: `Services OK`

- [x] **Step 4: Commit**

```bash
git add backend/app/services/
git commit -m "feat: add RAG search service and patient CRUD service"
```

---

### Task 8: 诊断推理服务

**Files:**
- Create: `backend/app/services/diagnosis_service.py`

**Interfaces:**
- Consumes: `DiagnosisPredictor` from Task 5 (`forward(x)` 返回 `Tensor([N, 10])`)；`DISEASE_LABELS` from Task 5；`TREATMENTS` from Task 6
- Produces: `predict_diagnosis(lab_data: dict) -> dict` 返回 `{"top_prediction", "confidence", "top3": [...], "treatment_suggestion"}`

- [x] **Step 1: 创建 backend/app/services/diagnosis_service.py**

```python
"""诊断预测推理服务"""

import torch
import numpy as np

from app.ml.diagnosis_model import (
    DiagnosisPredictor,
    DISEASE_LABELS,
    LAB_FEATURE_NAMES,
    create_model,
)

# 治疗方案映射（与 data_generator.py 保持一致）
TREATMENTS = {
    "细菌性肺炎": "抗生素治疗（如头孢曲松+阿奇霉素），退热，氧疗，充分休息，多饮水",
    "病毒性感冒": "对症支持治疗：退热药（对乙酰氨基酚），抗组胺药，充分休息，补充维生素C",
    "急性支气管炎": "止咳祛痰（氨溴索），支气管扩张剂，雾化吸入，多饮水，避免刺激",
    "高血压": "降压药（氨氯地平/缬沙坦），低盐饮食，规律运动，控制体重，监测血压",
    "2型糖尿病": "降糖药（二甲双胍），饮食控制，规律运动，血糖监测，糖尿病教育",
    "冠心病": "抗血小板药（阿司匹林），他汀类降脂药，硝酸酯类，控制危险因素",
    "慢性胃炎": "抑酸药（奥美拉唑），胃黏膜保护剂，规律饮食，避免辛辣刺激",
    "尿路感染": "抗生素（左氧氟沙星/呋喃妥因），多饮水，注意个人卫生",
    "缺铁性贫血": "补铁剂（硫酸亚铁），维生素C促进吸收，富含铁的食物，病因治疗",
    "甲状腺功能亢进": "抗甲状腺药（甲巯咪唑），β受体阻滞剂，定期复查甲功",
}


class DiagnosisService:
    """诊断预测推理服务（单例模式）"""

    _instance = None
    _model: DiagnosisPredictor | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def model(self) -> DiagnosisPredictor:
        if self._model is None:
            self._model = create_model()
        return self._model

    def preprocess(self, lab_report: dict) -> torch.Tensor:
        """将化验单 dict 转为模型输入张量 [1, 20]"""
        features = [float(lab_report.get(feat, 0.0)) for feat in LAB_FEATURE_NAMES]
        return torch.tensor([features], dtype=torch.float32)

    def predict(self, lab_report: dict) -> dict:
        """根据化验单数据预测疾病

        Args:
            lab_report: 包含 20 项化验指标的 dict

        Returns:
            {"top_prediction": str, "confidence": float, "top3": [...], "treatment_suggestion": str}
        """
        x = self.preprocess(lab_report)
        with torch.no_grad():
            probs = self.model(x)[0]  # [num_diseases]

        # 取 Top-3
        top3_indices = torch.topk(probs, 3).indices.tolist()
        top3_probs = torch.topk(probs, 3).values.tolist()

        top_prediction = DISEASE_LABELS[top3_indices[0]]
        confidence = round(top3_probs[0], 4)

        top3 = [
            {"disease": DISEASE_LABELS[idx], "probability": round(prob, 4)}
            for idx, prob in zip(top3_indices, top3_probs)
        ]

        treatment = TREATMENTS.get(top_prediction, "请咨询专业医生获取治疗方案")

        return {
            "top_prediction": top_prediction,
            "confidence": confidence,
            "top3": top3,
            "treatment_suggestion": treatment,
        }


# 全局单例
diagnosis_service = DiagnosisService()
```

- [x] **Step 2: 验证推理服务（需要先完成 Task 6 训练）**

Run: `cd backend && python -c "from app.services.diagnosis_service import diagnosis_service; from app.ml.data_generator import generate_mock_data; p = generate_mock_data(1)[0]; r = diagnosis_service.predict(p['lab_data']); print(f'Prediction: {r[\"top_prediction\"]} ({r[\"confidence\"]:.2%})')"`
Expected: 输出预测疾病名和置信度

- [x] **Step 3: Commit**

```bash
git add backend/app/services/diagnosis_service.py
git commit -m "feat: add diagnosis prediction inference service"
```

---

### Task 9: API 路由 + FastAPI 入口

**Files:**
- Create: `backend/app/api/__init__.py`
- Create: `backend/app/api/health.py`
- Create: `backend/app/api/patients.py`
- Create: `backend/app/api/diagnosis.py`
- Create: `backend/app/api/router.py`
- Modify: `backend/app/main.py` (create)

**Interfaces:**
- Consumes: `diagnosis_service.predict()` from Task 8；`search_similar_patients()` from Task 7；`patient_service` from Task 7；`DiagnoseRequest`, `DiagnoseResponse`, `PatientSummary`, `PatientDetail`, `DiseaseInfo` from Task 3；`get_db()` from Task 2
- Produces: FastAPI app on port 8000, routes: `GET /api/health`, `GET /api/patients`, `GET /api/patients/{id}`, `GET /api/diseases`, `POST /api/diagnose`

- [x] **Step 1: 创建 backend/app/api/health.py**

```python
"""健康检查接口"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "hospital-diagnosis-system"}
```

- [x] **Step 2: 创建 backend/app/api/patients.py**

```python
"""病人查询接口"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.diagnosis import PatientSummary, PatientDetail
from app.services import patient_service

router = APIRouter(prefix="/patients", tags=["patients"])


@router.get("", response_model=list[PatientSummary])
async def list_patients(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """分页查询病人列表"""
    patients = await patient_service.get_all_patients(db, skip=skip, limit=limit)
    return [
        PatientSummary(
            id=p.id,
            name=p.name,
            age=p.age,
            gender=p.gender,
            diagnosis=p.diagnosis,
            created_at=p.created_at,
        )
        for p in patients
    ]


@router.get("/{patient_id}", response_model=PatientDetail)
async def get_patient(patient_id: int, db: AsyncSession = Depends(get_db)):
    """根据 ID 查询病人详情"""
    patient = await patient_service.get_patient(db, patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="病人不存在")
    return PatientDetail(
        id=patient.id,
        name=patient.name,
        age=patient.age,
        gender=patient.gender,
        symptom_description=patient.symptom_description,
        diagnosis=patient.diagnosis,
        treatment=patient.treatment,
        lab_data=patient.lab_data,
        created_at=patient.created_at,
    )
```

- [x] **Step 3: 创建 backend/app/api/diagnosis.py**

```python
"""诊断核心接口"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.diagnosis import (
    DiagnoseRequest,
    DiagnoseResponse,
    DiagnosisResult,
    SimilarCase,
    DiseaseInfo,
    DiagnosisItem,
)
from app.ml.diagnosis_model import DISEASE_LABELS
from app.services.rag_service import search_similar_patients
from app.services.diagnosis_service import diagnosis_service
from app.services.patient_service import create_patient

router = APIRouter(tags=["diagnosis"])


@router.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(request: DiagnoseRequest, db: AsyncSession = Depends(get_db)):
    """核心诊断接口：接收症状+化验单，并行执行 RAG 检索和 DL 预测，合并返回"""
    # 并行执行 RAG 检索和诊断预测
    rag_task = search_similar_patients(
        db, request.symptom_description, top_k=5
    )
    # diagnosis_service.predict 是同步的，用 asyncio.to_thread 避免阻塞
    diagnosis_task = asyncio.to_thread(
        diagnosis_service.predict,
        request.lab_report.model_dump(),
    )

    similar_cases_raw, diagnosis_raw = await asyncio.gather(rag_task, diagnosis_task)

    # 构造响应
    top3 = [
        DiagnosisItem(disease=item["disease"], probability=item["probability"])
        for item in diagnosis_raw["top3"]
    ]
    diagnosis_result = DiagnosisResult(
        top_prediction=diagnosis_raw["top_prediction"],
        confidence=diagnosis_raw["confidence"],
        top3=top3,
        treatment_suggestion=diagnosis_raw["treatment_suggestion"],
    )

    similar_cases = [
        SimilarCase(
            id=case["id"],
            similarity=case["similarity"],
            symptom_description=case["symptom_description"],
            diagnosis=case["diagnosis"],
            treatment=case["treatment"],
        )
        for case in similar_cases_raw
    ]

    # 将本次诊断结果存入数据库
    try:
        await create_patient(
            db=db,
            name=request.name,
            age=request.age,
            gender=request.gender,
            symptom_description=request.symptom_description,
            diagnosis=diagnosis_raw["top_prediction"],
            treatment=diagnosis_raw["treatment_suggestion"],
            lab_data=request.lab_report.model_dump(),
        )
    except Exception:
        pass  # 存储失败不阻塞响应

    return DiagnoseResponse(diagnosis=diagnosis_result, similar_cases=similar_cases)


@router.get("/diseases", response_model=list[DiseaseInfo])
async def list_diseases():
    """返回系统支持的疾病列表"""
    return [
        DiseaseInfo(name=name, description=f"{name}的诊断与治疗")
        for name in DISEASE_LABELS
    ]
```

- [x] **Step 4: 创建 backend/app/api/router.py**

```python
"""API 路由聚合"""

from fastapi import APIRouter

from app.api.health import router as health_router
from app.api.patients import router as patients_router
from app.api.diagnosis import router as diagnosis_router

api_router = APIRouter(prefix="/api")
api_router.include_router(health_router)
api_router.include_router(patients_router)
api_router.include_router(diagnosis_router)
```

- [x] **Step 5: 创建 backend/app/main.py**

```python
"""FastAPI 应用入口"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭时的生命周期事件"""
    await init_db()
    yield


app = FastAPI(
    title="智能医疗诊断辅助系统",
    description="基于 RAG 和深度学习的医疗诊断辅助 API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 配置（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
```

- [x] **Step 6: 在 backend/app/services/__init__.py 中添加导入**

```python
from app.services import rag_service, patient_service, diagnosis_service
```

(确认 `backend/app/services/__init__.py` 文件已存在并包含以上内容)

- [x] **Step 7: 启动后端验证 API**

Run: `cd backend && uvicorn app.main:app --reload --port 8000`
Open `http://localhost:8000/docs` 查看 Swagger 文档
Expected: 看到 5 个 API 接口（health, patients list, patient detail, diagnose, diseases）

- [x] **Step 8: Commit**

```bash
git add backend/app/api/ backend/app/main.py backend/app/services/__init__.py
git commit -m "feat: add API endpoints (health, patients, diagnose, diseases) and FastAPI entry point"
```

---

### Task 10: 前端项目搭建

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/tsconfig.node.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/App.css`
- Create: `frontend/src/vite-env.d.ts`

**Interfaces:**
- Consumes: nothing
- Produces: React 18 + TypeScript + Vite 项目框架，`App` 组件

- [x] **Step 1: 创建 frontend/package.json**

```json
{
  "name": "hospital-frontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.23.1",
    "axios": "^1.7.2"
  },
  "devDependencies": {
    "@types/react": "^18.3.3",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.1",
    "typescript": "^5.5.2",
    "vite": "^5.3.1"
  }
}
```

- [x] **Step 2: 创建 frontend/vite.config.ts**

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
```

- [x] **Step 3: 创建 frontend/tsconfig.json**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

- [x] **Step 4: 创建 frontend/tsconfig.node.json**

```json
{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true
  },
  "include": ["vite.config.ts"]
}
```

- [x] **Step 5: 创建 frontend/index.html**

```html
<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🏥</text></svg>" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>智能医疗诊断辅助系统</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [x] **Step 6: 创建 frontend/src/main.tsx**

```tsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './App.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
```

- [x] **Step 7: 创建 frontend/src/App.tsx**

```tsx
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import DiagnosisPage from './pages/DiagnosisPage'
import ResultPage from './pages/ResultPage'

function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <header className="app-header">
          <h1>🏥 智能医疗诊断辅助系统</h1>
        </header>
        <main className="app-main">
          <Routes>
            <Route path="/" element={<DiagnosisPage />} />
            <Route path="/result" element={<ResultPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

export default App
```

- [x] **Step 8: 创建 frontend/src/App.css**

```css
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  background: #f0f4f8;
  color: #1a202c;
}

.app {
  min-height: 100vh;
}

.app-header {
  background: linear-gradient(135deg, #1a56db, #1e40af);
  color: white;
  padding: 1.25rem 2rem;
  text-align: center;
  box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}

.app-header h1 {
  font-size: 1.5rem;
  font-weight: 600;
}

.app-main {
  max-width: 960px;
  margin: 2rem auto;
  padding: 0 1rem;
}
```

- [x] **Step 9: 创建 frontend/src/vite-env.d.ts**

```typescript
/// <reference types="vite/client" />
```

- [x] **Step 10: 安装依赖并验证**

Run: `cd frontend && npm install && npm run dev`
Expected: Vite 启动成功，打开 `http://localhost:3000` 看到标题 "🏥 智能医疗诊断辅助系统"

- [x] **Step 11: Commit**

```bash
git add frontend/
git commit -m "feat: scaffold React + TypeScript + Vite frontend project"
```

---

### Task 11: 前端 — Type 定义 + API 服务层

**Files:**
- Create: `frontend/src/types/diagnosis.ts`
- Create: `frontend/src/services/api.ts`

**Interfaces:**
- Consumes: `DiagnoseResponse` schema 结构（参考 Task 3）
- Produces: TypeScript 类型 `LabReport`, `DiagnoseRequest`, `DiagnoseResponse`, `DiagnosisItem`, `DiagnosisResult`, `SimilarCase`；`api.diagnose(req)`, `api.getPatients()`, `api.getPatient(id)`, `api.getDiseases()`

- [x] **Step 1: 创建 frontend/src/types/diagnosis.ts**

```typescript
// ── Request types ──────────────────────────────

export interface LabReport {
  wbc: number;
  neutrophil_pct: number;
  lymphocyte_pct: number;
  crp: number;
  temperature: number;
  systolic_bp: number;
  diastolic_bp: number;
  heart_rate: number;
  respiratory_rate: number;
  spo2: number;
  rbc: number;
  hemoglobin: number;
  hematocrit: number;
  platelet: number;
  glucose: number;
  creatinine: number;
  bun: number;
  alt: number;
  ast: number;
  total_cholesterol: number;
  triglycerides: number;
}

export interface DiagnoseRequest {
  name: string;
  age: number;
  gender: 'male' | 'female' | 'other';
  symptom_description: string;
  lab_report: LabReport;
}

// ── Response types ─────────────────────────────

export interface DiagnosisItem {
  disease: string;
  probability: number;
}

export interface DiagnosisResult {
  top_prediction: string;
  confidence: number;
  top3: DiagnosisItem[];
  treatment_suggestion: string;
}

export interface SimilarCase {
  id: number;
  similarity: number;
  symptom_description: string;
  diagnosis: string;
  treatment: string;
}

export interface DiagnoseResponse {
  diagnosis: DiagnosisResult;
  similar_cases: SimilarCase[];
}

export interface DiseaseInfo {
  name: string;
  description: string;
}

export interface PatientSummary {
  id: number;
  name: string;
  age: number;
  gender: string;
  diagnosis?: string;
  created_at?: string;
}

export interface PatientDetail extends PatientSummary {
  symptom_description: string;
  treatment?: string;
  lab_data?: Record<string, number>;
}
```

- [x] **Step 2: 创建 frontend/src/services/api.ts**

```typescript
import axios from 'axios';
import type {
  DiagnoseRequest,
  DiagnoseResponse,
  DiseaseInfo,
  PatientSummary,
  PatientDetail,
} from '../types/diagnosis';

const client = axios.create({
  baseURL: '/api',
  timeout: 60000,  // 模型推理可能需要一些时间
  headers: { 'Content-Type': 'application/json' },
});

export const api = {
  /** 提交诊断请求 */
  diagnose(data: DiagnoseRequest): Promise<{ data: DiagnoseResponse }> {
    return client.post('/diagnose', data);
  },

  /** 获取疾病列表 */
  getDiseases(): Promise<{ data: DiseaseInfo[] }> {
    return client.get('/diseases');
  },

  /** 获取病人列表 */
  getPatients(skip = 0, limit = 20): Promise<{ data: PatientSummary[] }> {
    return client.get('/patients', { params: { skip, limit } });
  },

  /** 获取病人详情 */
  getPatient(id: number): Promise<{ data: PatientDetail }> {
    return client.get(`/patients/${id}`);
  },

  /** 健康检查 */
  healthCheck(): Promise<{ data: { status: string } }> {
    return client.get('/health');
  },
};
```

- [x] **Step 3: 验证编译**

Run: `cd frontend && npx tsc --noEmit`
Expected: 无错误

- [x] **Step 4: Commit**

```bash
git add frontend/src/types/ frontend/src/services/
git commit -m "feat: add TypeScript type definitions and API service layer"
```

---

### Task 12: 前端 — 表单组件 + 页面

**Files:**
- Create: `frontend/src/components/PatientForm.tsx`
- Create: `frontend/src/components/LabReportForm.tsx`
- Create: `frontend/src/components/DiagnosisResult.tsx`
- Create: `frontend/src/components/SimilarCases.tsx`
- Create: `frontend/src/pages/DiagnosisPage.tsx`
- Create: `frontend/src/pages/ResultPage.tsx`

**Interfaces:**
- Consumes: `api` from Task 11；types from Task 11；路由 `/` 和 `/result`
- Produces: 完整的诊断录入页面和结果展示页面

- [x] **Step 1: 创建 frontend/src/components/PatientForm.tsx**

```tsx
import type { DiagnoseRequest } from '../types/diagnosis';

interface Props {
  formData: DiagnoseRequest;
  onChange: (data: DiagnoseRequest) => void;
}

export default function PatientForm({ formData, onChange }: Props) {
  const update = (field: string, value: string | number) => {
    onChange({ ...formData, [field]: value });
  };

  return (
    <section className="form-section">
      <h2>📋 基本信息</h2>
      <div className="form-row">
        <label>
          姓名
          <input
            type="text"
            value={formData.name}
            onChange={(e) => update('name', e.target.value)}
            placeholder="请输入姓名"
          />
        </label>
        <label>
          年龄
          <input
            type="number"
            value={formData.age || ''}
            onChange={(e) => update('age', Number(e.target.value))}
            placeholder="0"
            min={0}
            max={150}
          />
        </label>
        <label>
          性别
          <select
            value={formData.gender}
            onChange={(e) => update('gender', e.target.value)}
          >
            <option value="">请选择</option>
            <option value="male">男</option>
            <option value="female">女</option>
            <option value="other">其他</option>
          </select>
        </label>
      </div>

      <label className="form-field-full">
        症状描述
        <textarea
          value={formData.symptom_description}
          onChange={(e) => update('symptom_description', e.target.value)}
          placeholder="请详细描述您的症状，例如：头痛发烧三天，咳嗽有黄痰，胸闷气短..."
          rows={4}
        />
      </label>
    </section>
  );
}
```

- [x] **Step 2: 创建 frontend/src/components/LabReportForm.tsx**（含 OCR 图片上传）

Component includes: 4 collapsible lab groups (炎症指标, 生命体征, 血常规, 生化指标), image upload area with drag/drop, OCR button calling `POST /api/ocr`, auto-fill form fields from OCR result.

- [x] **Step 3: 创建 frontend/src/components/DiagnosisResult.tsx**

```tsx
import type { DiagnosisResult as DiagnosisResultType } from '../types/diagnosis';

interface Props {
  result: DiagnosisResultType;
}

const RISK_COLORS: Record<number, string> = {
  0: '#22c55e',  // 低风险 → 绿
  1: '#f59e0b',  // 中风险 → 黄
  2: '#ef4444',  // 高风险 → 红
};

export default function DiagnosisResultCard({ result }: Props) {
  return (
    <section className="form-section result-card">
      <h2>📊 AI 诊断结果</h2>

      <div className="primary-diagnosis">
        <span className="disease-label">{result.top_prediction}</span>
        <span className="confidence">
          置信度 {(result.confidence * 100).toFixed(1)}%
        </span>
      </div>

      <div className="top3-list">
        {result.top3.map((item, idx) => (
          <div key={item.disease} className="top3-item">
            <span
              className="risk-dot"
              style={{ background: RISK_COLORS[idx] || '#6b7280' }}
            />
            <span className="disease-name">{item.disease}</span>
            <span className="probability">
              {(item.probability * 100).toFixed(1)}%
            </span>
            <div className="prob-bar">
              <div
                className="prob-bar-fill"
                style={{
                  width: `${(item.probability * 100).toFixed(1)}%`,
                  background: RISK_COLORS[idx] || '#6b7280',
                }}
              />
            </div>
          </div>
        ))}
      </div>

      <div className="treatment-box">
        <h3>💊 建议治疗方案</h3>
        <p>{result.treatment_suggestion}</p>
      </div>
    </section>
  );
}
```

- [x] **Step 4: 创建 frontend/src/components/SimilarCases.tsx**

```tsx
import type { SimilarCase } from '../types/diagnosis';

interface Props {
  cases: SimilarCase[];
}

export default function SimilarCasesList({ cases }: Props) {
  if (cases.length === 0) {
    return (
      <section className="form-section">
        <h2>📚 相似历史病例</h2>
        <p className="empty-hint">暂无相似病例参考</p>
      </section>
    );
  }

  return (
    <section className="form-section">
      <h2>📚 相似历史病例 (RAG 检索)</h2>
      {cases.map((c) => (
        <div key={c.id} className="similar-case-item">
          <div className="case-header">
            <span className="similarity-badge">
              相似度 {(c.similarity * 100).toFixed(0)}%
            </span>
            <span className="case-diagnosis">确诊: {c.diagnosis}</span>
          </div>
          <p className="case-symptoms">症状: {c.symptom_description}</p>
          <p className="case-treatment">治疗: {c.treatment}</p>
        </div>
      ))}
    </section>
  );
}
```

- [x] **Step 5: 创建 frontend/src/pages/DiagnosisPage.tsx**（左-右两栏布局，含 OCR）

Left panel: PatientForm + LabReportForm (with OCR upload) + Submit button
Right panel: Results (DiagnosisResultCard + SimilarCasesList) or empty state
Results shown inline (no page navigation).

- [x] **Step 6: 创建 frontend/src/pages/ResultPage.tsx**（独立结果页，兼容旧链接）

- [x] **Step 7: 添加组件样式到 App.css**（完整医疗主题 CSS）

- [x] **Step 8: 验证前端页面**

Run: `cd frontend && npx tsc --noEmit`
Expected: 无 TypeScript 错误

Start: `cd frontend && npm run dev`
Open `http://localhost:3000` — 应看到完整的诊断录入表单含 OCR 上传区

- [x] **Step 9: Commit**

```bash
git add frontend/src/components/ frontend/src/pages/ frontend/src/App.css
git commit -m "feat: add patient form, lab report form with OCR, diagnosis result, and similar cases components"
```

---

### Task 13: Dockerfile + 数据导入

**Files:**
- Create: `backend/Dockerfile`
- Create: `frontend/Dockerfile`
- Create: `frontend/nginx.conf`
- Modify: `data/init.sql` (add mock data seed)

**Interfaces:**
- Consumes: `docker-compose.yml` 中定义的服务配置
- Produces: 可一键 `docker compose up --build` 启动的完整系统

- [x] **Step 1: 创建 backend/Dockerfile**（含 EasyOCR 系统依赖和 apt/pip 镜像源）

- [x] **Step 2: 创建 frontend/Dockerfile**（Node.js 构建 + Nginx 部署）

- [x] **Step 3: 创建 frontend/nginx.conf**（API 代理到 backend:8000）

- [x] **Step 4: 添加应用启动时自动导入模拟数据的 lifespan 逻辑**

- [x] **Step 5: 更新 docker-compose.yml 添加 OCR 环境变量**

```yaml
environment:
  DEEPSEEK_API_KEY: sk-xxx
  BAIDU_OCR_APP_ID: "xxx"
  BAIDU_OCR_API_KEY: xxx
  BAIDU_OCR_SECRET_KEY: xxx
  HF_ENDPOINT: https://hf-mirror.com
```

- [x] **Step 6: 启动完整服务进行验证**

Run: `docker compose up --build`
Expected: 三个服务启动成功，OCR 端点可用

- [x] **Step 7: Commit**

```bash
git add backend/Dockerfile frontend/Dockerfile frontend/nginx.conf backend/app/main.py docker-compose.yml
git commit -m "feat: add Dockerfiles, nginx config, OCR env vars, and auto data seeding"
```

---

### Task 14: 端到端验证

**Files:**
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/test_diagnosis.py`

**Interfaces:**
- Consumes: 完整运行的后端服务
- Produces: 验证报告

- [x] **Step 1: 创建 backend/tests/test_diagnosis.py**（3 个测试：health / diseases / diagnose）

- [x] **Step 2: 运行测试**

```bash
cd backend && pip install pytest httpx
python -m pytest tests/ -v
```
Expected: 3 tests pass

- [x] **Step 3: Commit**

```bash
git add backend/tests/
git commit -m "test: add end-to-end diagnosis API tests"
```

---

### Task 15: OCR 化验单识别（百度 OCR + EasyOCR 降级 + DeepSeek 结构化）

**Files:**
- Create: `backend/app/api/ocr.py`
- Create: `backend/app/services/ocr_service.py`
- Modify: `backend/app/config.py` — 新增 `baidu_ocr_app_id`, `baidu_ocr_api_key`, `baidu_ocr_secret_key`, `deepseek_api_key`
- Modify: `backend/app/api/router.py` — 注册 OCR 路由
- Modify: `backend/requirements.txt` — 新增 `easyocr`, `openai`, `Pillow`, `baidu-aip`
- Modify: `backend/Dockerfile` — 安装 EasyOCR 系统依赖（libgl1, libgomp1 等）
- Modify: `docker-compose.yml` — 添加 `BAIDU_OCR_*` 和 `DEEPSEEK_API_KEY` 环境变量

**Architecture:**
```
化验单图片 → 百度 OCR 精确版 (主引擎)
                │
                ├── 成功 ──→ DeepSeek 结构化 ──→ {"lab_data": {...}}
                │
                └── 失败 ──→ EasyOCR (降级) ──→ DeepSeek 结构化 ──→ 同上
```

**Interfaces:**
- Consumes: `settings.baidu_ocr_*`, `settings.deepseek_api_key`
- Produces: `POST /api/ocr` — 接受图片上传，返回 `{"lab_data": {...}, "raw_text": "..."}`

**Key implementation details:**
- `OCRService` 单例模式，百度 OCR client 和 EasyOCR Reader 均延迟加载
- `baidu_available` 属性检查凭证是否配置，未配置直接走 EasyOCR
- 百度 OCR 调用 `accurateBasic` API（精确版，中英文混合），图片以 base64 传入
- 百度 OCR 失败时 `logger.warning` 记录原因，自动降级到 EasyOCR
- 前端 `LabReportForm` 含图片上传区和 OCR 按钮，识别结果自动填入表单字段

- [x] **Step 1: 更新 requirements.txt**

```
easyocr>=1.7.0
openai>=1.0.0
Pillow>=10.0.0
baidu-aip>=4.16.0
```

- [x] **Step 2: 更新 config.py 添加 OCR 和 DeepSeek 配置项**

- [x] **Step 3: 创建 ocr_service.py（百度主引擎 + EasyOCR 降级 + DeepSeek 结构化）**

- [x] **Step 4: 创建 api/ocr.py（POST /api/ocr 端点）**

- [x] **Step 5: 注册路由、更新 Dockerfile 系统依赖、更新 docker-compose.yml 环境变量**

- [x] **Step 6: 启动验证**

```bash
docker compose up --build
curl -X POST http://localhost:8000/api/ocr -F "file=@lab_report.jpg"
```
Expected: 返回 21 项化验指标 JSON

- [x] **Step 7: Commit**

```bash
git add backend/app/services/ocr_service.py backend/app/api/ocr.py \
        backend/app/api/router.py backend/app/config.py \
        backend/requirements.txt backend/Dockerfile docker-compose.yml
git commit -m "feat: add OCR lab report recognition (Baidu OCR + EasyOCR fallback + DeepSeek)"
```

---

### Task 16: 前端 UX 优化 — 表单校验 + 加载动画

- [x] **PatientForm**: 实时字段校验（姓名/年龄/性别/症状），红色错误提示 + 必填标记 + 字数统计
- [x] **DiagnosisPage**: 四步加载动画（分析症状→检索病例→AI诊断→生成方案），骨架屏 shimmer
- [x] **LoadingSkeleton**: 新组件，步骤进度 + 骨架卡片
- [x] **CSS**: 校验状态、spinner、shimmer 动画、alert 样式

---

### Task 17: OCR 自动提取患者信息

- [x] **ocr_service.py**: DeepSeek prompt 加入患者信息提取（name/age/gender）
- [x] **LabReportForm**: 新增 `onPatientInfo` 回调
- [x] **DiagnosisPage**: OCR 后自动填入 PatientForm 姓名/年龄/性别

---

### Task 18: RAG 数据源升级 — 真实医疗对话

- [x] **数据下载**: Chinese-medical-dialogue-data（22 万条内科对话 CSV）
- [x] **数据清洗**: `backend/scripts/clean_medical_data.py` — NLP 清洗，200+ 常见病名精确匹配，标题优先
- [x] **结果**: 清洗出 14.9 万条有效记录，191 种疾病，分层采样 1.8 万条
- [x] **导入脚本**: `backend/scripts/import_real_data.py` — 批量 embedding + pgvector 入库

---

### Task 19: Embedding 模型升级

- [x] **切换**: `text2vec-base-chinese` → `BAAI/bge-base-zh-v1.5`（768 维不变，零代码改动）
- [x] **docker-compose.yml**: 更新 `EMBEDDING_MODEL` 环境变量

---

### Task 20: 安全 — API Key 管理

- [x] **docker-compose.yml**: 硬编码密钥 → `${VAR}` 引用
- [x] **.env.example**: 环境变量模板
- [x] **.gitignore**: `.env` 已配置
- [x] **git history**: `filter-branch` 清洗全部 22 个 commits 中的密钥