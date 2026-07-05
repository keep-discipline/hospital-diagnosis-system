"""Transformer 文本向量化模块

使用预训练中文模型将症状描述转为固定维度向量，供 pgvector 检索使用。
模型可替换：改 settings.embedding_model 即可切换。
"""

from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingModel:
    """封装 SentenceTransformer，将中文症状描述转为 768 维向量"""

    def __init__(self, model_name: str | None = None):
        model_name = model_name or settings.embedding_model
        self._model = SentenceTransformer(model_name, local_files_only=True)

    def encode(self, text: str) -> list[float]:
        """单条文本 → 向量（已 L2 归一化，方便余弦相似度计算）"""
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """批量文本 → 向量列表"""
        embeddings = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=True
        )
        return embeddings.tolist()


# 全局单例 —— 模型 ~400MB，只加载一次
embedding_model = EmbeddingModel()
