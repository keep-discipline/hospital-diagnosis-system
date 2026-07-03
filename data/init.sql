-- 启用 pgvector 扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 创建病人表
CREATE TABLE IF NOT EXISTS patients (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    age INT NOT NULL,
    gender VARCHAR(10) NOT NULL,
    symptom_description TEXT NOT NULL,
    symptom_embedding vector(768),
    diagnosis VARCHAR(200),
    treatment TEXT,
    lab_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 向量索引（IVFFlat 需要表中有一定数据量再创建）
-- CREATE INDEX ON patients USING ivfflat (symptom_embedding vector_cosine_ops) WITH (lists = 100);
