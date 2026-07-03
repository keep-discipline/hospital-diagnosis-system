"""诊断相关 Pydantic schemas"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Request ──────────────────────────────────────────────

class LabReport(BaseModel):
    """化验单标准指标（21 项）"""
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
