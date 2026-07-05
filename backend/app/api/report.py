"""诊断报告导出接口"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from app.services.pdf_service import generate_report

router = APIRouter(tags=["report"])


class ReportRequest(BaseModel):
    name: str
    age: int
    gender: str
    symptom_description: str
    lab_data: dict
    diagnosis: dict
    similar_cases: list = []


@router.post("/report")
async def export_report(req: ReportRequest):
    """生成诊断报告 PDF"""
    try:
        pdf_bytes = generate_report(
            patient_name=req.name,
            patient_age=req.age,
            patient_gender=req.gender,
            symptom=req.symptom_description,
            lab_data=req.lab_data,
            diagnosis=req.diagnosis,
            similar_cases=req.similar_cases,
        )
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=diagnosis_report.pdf"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 生成失败: {e}")
