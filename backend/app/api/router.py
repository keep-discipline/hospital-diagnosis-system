"""API 路由聚合"""

from fastapi import APIRouter

from app.api.health import router as health_router
from app.api.patients import router as patients_router
from app.api.diagnosis import router as diagnosis_router
from app.api.ocr import router as ocr_router

api_router = APIRouter(prefix="/api")
api_router.include_router(health_router)
api_router.include_router(patients_router)
api_router.include_router(diagnosis_router)
api_router.include_router(ocr_router)
