"""导入真实医疗数据到数据库

1. 清空旧 patients 表
2. 读取 cleaned_final.json
3. 用 bge-base-zh-v1.5 生成 embedding
4. 批量导入 pgvector
"""

import sys, json, asyncio, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from app.database import async_session
from app.ml.embedding import embedding_model


async def import_data(json_path: str, batch_size: int = 500):
    print(f"📖 读取: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        patients = json.load(f)

    print(f"📊 {len(patients)} 条记录待导入")
    print(f"🔤 使用模型: {embedding_model._model.get_sentence_embedding_dimension()} 维")

    async with async_session() as db:
        # 1. 清空旧数据
        print("🗑️ 清空旧数据...")
        await db.execute(text("DELETE FROM patients"))
        await db.commit()

        # 2. 分批处理
        total = len(patients)
        imported = 0
        for i in range(0, total, batch_size):
            batch = patients[i:i+batch_size]
            texts = [p['symptom_description'] for p in batch]

            # 生成 embedding (异步，避免阻塞)
            embeddings = embedding_model.encode_batch(texts)

            # 批量 INSERT
            for p, emb in zip(batch, embeddings):
                vec_str = "[" + ",".join(f"{v:.8f}" for v in emb) + "]"
                await db.execute(
                    text("""
                        INSERT INTO patients (name, age, gender, symptom_description,
                                             symptom_embedding, diagnosis, treatment)
                        VALUES (:name, :age, :gender, :desc,
                                CAST(:emb AS vector), :diag, :treat)
                    """),
                    {
                        "name": p["name"], "age": p["age"], "gender": p["gender"],
                        "desc": p["symptom_description"], "emb": vec_str,
                        "diag": p["diagnosis"], "treat": p["treatment"],
                    },
                )
            await db.commit()

            imported += len(batch)
            pct = imported / total * 100
            print(f"  ⏳ {imported}/{total} ({pct:.1f}%)")

    # 3. 验证
    async with async_session() as db:
        result = await db.execute(text("SELECT COUNT(*) FROM patients"))
        count = result.scalar()
        result2 = await db.execute(text(
            "SELECT COUNT(*) FROM patients WHERE symptom_embedding IS NOT NULL"
        ))
        with_emb = result2.scalar()
        print(f"\n✅ 导入完成: {count} 条记录, {with_emb} 条含 embedding")


if __name__ == '__main__':
    json_path = sys.argv[1] if len(sys.argv) > 1 else '/app/data/cleaned_final.json'
    asyncio.run(import_data(json_path))
