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
