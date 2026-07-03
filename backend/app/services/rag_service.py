"""RAG 检索服务

核心流程: 症状描述 → Transformer embedding → pgvector 余弦相似度 → Top-K 相似病例
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.embedding import embedding_model


def _to_vector_str(embedding: list[float]) -> str:
    """将 Python 列表转为 pgvector 可识别的字符串格式 '[x1,x2,...]'"""
    return "[" + ",".join(str(v) for v in embedding) + "]"


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
        相似病例列表，每项包含 id/similarity/symptom_description/diagnosis/treatment
    """
    # 1. 文字 → 向量
    query_embedding = embedding_model.encode(symptom_text)
    vector_str = _to_vector_str(query_embedding)

    # 2. pgvector 余弦相似度查询（<=> 是 pgvector 的余弦距离操作符）
    query = text(f"""
        SELECT
            id,
            symptom_description,
            diagnosis,
            treatment,
            1 - (symptom_embedding <=> CAST(:embedding AS vector)) AS similarity
        FROM patients
        WHERE symptom_embedding IS NOT NULL
        ORDER BY symptom_embedding <=> CAST(:embedding AS vector)
        LIMIT :top_k
    """)

    result = await db.execute(
        query,
        {"embedding": vector_str, "top_k": top_k},
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


async def update_patient_embedding(
    db: AsyncSession,
    patient_id: int,
    symptom_text: str,
) -> None:
    """为指定病人生成并更新 symptom embedding"""
    embedding = embedding_model.encode(symptom_text)
    vector_str = _to_vector_str(embedding)
    update_query = text("""
        UPDATE patients
        SET symptom_embedding = CAST(:embedding AS vector)
        WHERE id = :patient_id
    """)
    await db.execute(
        update_query,
        {"embedding": vector_str, "patient_id": patient_id},
    )
    await db.commit()
