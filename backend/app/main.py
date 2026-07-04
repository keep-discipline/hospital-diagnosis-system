"""FastAPI 应用入口"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

# 配置应用日志，确保 INFO 级别日志输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
# 降低第三方库日志噪音
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时: 建表 + 导入模拟数据（如果数据库为空）"""
    await init_db()

    # 如果数据库为空，自动导入模拟病例
    mock_file = Path("data/mock_patients.json")
    if mock_file.exists():
        from app.services.patient_service import import_mock_patients
        from app.database import async_session
        from sqlalchemy import text

        async with async_session() as db:
            result = await db.execute(text("SELECT COUNT(*) FROM patients"))
            count = result.scalar()
            if count == 0:
                with open(mock_file, "r", encoding="utf-8") as f:
                    patients_data = json.load(f)
                n = await import_mock_patients(db, patients_data)
                print(f"已导入 {n} 条模拟病例数据（含 embedding）")
    else:
        print("未找到模拟数据文件，跳过数据导入")

    yield


app = FastAPI(
    title="智能医疗诊断辅助系统",
    description="基于 RAG 和深度学习的医疗诊断辅助 API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
