FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖（编译工具 + PaddleOCR/OpenCV 依赖）
# 使用国内 Debian 镜像加速，加重试防止网络波动
RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖（使用清华镜像加速）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制应用代码
COPY . .

# 创建模型和数据目录
RUN mkdir -p /app/data/models

EXPOSE 8000

# 启动: 模型不存在时训练，否则直接启动 API
CMD ["sh", "-c", "\
  if [ ! -f /app/data/models/diagnosis_model.pt ]; then \
    echo '模型未训练，开始训练...' && python -m app.ml.train; \
  else \
    echo '模型已训练，跳过'; \
  fi && \
  uvicorn app.main:app --host 0.0.0.0 --port 8000"]
