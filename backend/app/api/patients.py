"""病人查询接口"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.diagnosis import PatientSummary, PatientDetail
from app.services import patient_service

router = APIRouter(prefix="/patients", tags=["patients"])


@router.get("")
async def list_patients(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    q: str = Query("", description="搜索关键词：姓名/症状/诊断"),
    db: AsyncSession = Depends(get_db),
):
    """分页查询病人列表，支持搜索"""
    patients, total = await patient_service.search_patients(db, q=q, skip=skip, limit=limit)
    return {
        "total": total,
        "data": [
            PatientSummary(
                id=p.id, name=p.name, age=p.age, gender=p.gender,
                diagnosis=p.diagnosis, created_at=p.created_at,
            )
            for p in patients
        ],
    }


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
