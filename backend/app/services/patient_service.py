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
    result = await db.execute(
        select(Patient).where(Patient.id == patient_id)
    )
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
    """批量导入模拟病人数据（含 embedding）

    Args:
        db: 数据库 session
        patients_data: generate_mock_data() 产出的病例列表

    Returns:
        导入的记录数
    """
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
