"""应用配置"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://hospital:hospital@localhost:5432/hospital"
    embedding_model: str = "shibing624/text2vec-base-chinese"
    model_path: str = "data/models/diagnosis_model.pt"
    top_k_similar: int = 5  # RAG 检索返回的相似病例数

    class Config:
        env_file = ".env"


settings = Settings()
