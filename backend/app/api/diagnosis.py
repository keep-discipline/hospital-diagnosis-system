"""诊断核心接口 — RAG + DL 并行执行"""

import asyncio

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.diagnosis import (
    DiagnoseRequest,
    DiagnoseResponse,
    DiagnosisResult,
    DiagnosisItem,
    SimilarCase,
    DiseaseInfo,
)
from app.ml.diagnosis_model import DISEASE_LABELS
from app.services.rag_service import search_similar_patients
from app.services.diagnosis_service import diagnosis_service
from app.services.patient_service import create_patient

router = APIRouter(tags=["diagnosis"])


@router.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(request: DiagnoseRequest, db: AsyncSession = Depends(get_db)):
    """核心诊断接口

    并行执行:
      - RAG 通道: 症状描述 → embedding → pgvector 检索相似病例
      - DL 通道:  化验单数据 → MLP 模型 → 疾病预测
    最终合并返回。
    """
    # 两个通道并行执行
    rag_task = search_similar_patients(db, request.symptom_description, top_k=5)
    diagnosis_task = asyncio.to_thread(
        diagnosis_service.predict, request.lab_report.model_dump()
    )

    similar_cases_raw, diagnosis_raw = await asyncio.gather(
        rag_task, diagnosis_task
    )

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

    # 存储本次诊断结果（失败不阻塞响应）
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
        pass

    return DiagnoseResponse(
        diagnosis=diagnosis_result, similar_cases=similar_cases
    )


@router.get("/diseases", response_model=list[DiseaseInfo])
async def list_diseases():
    """返回系统支持的疾病列表"""
    return [
        DiseaseInfo(name=name, description=f"{name}的诊断与治疗")
        for name in DISEASE_LABELS
    ]
