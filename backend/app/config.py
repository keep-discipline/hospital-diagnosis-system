"""应用配置"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://hospital:hospital@localhost:5432/hospital"
    embedding_model: str = "shibing624/text2vec-base-chinese"
    model_path: str = "data/models/diagnosis_model.pt"
    top_k_similar: int = 5  # RAG 检索返回的相似病例数
    deepseek_api_key: str = ""  # DeepSeek API Key
    baidu_ocr_app_id: str = ""  # 百度 OCR App ID
    baidu_ocr_api_key: str = ""  # 百度 OCR API Key
    baidu_ocr_secret_key: str = ""  # 百度 OCR Secret Key

    class Config:
        env_file = ".env"


settings = Settings()
