# Imagem oficial Ultralytics (YOLO + PyTorch + OpenCV).
# GPU no Windows: Docker Desktop com backend WSL2 + driver NVIDIA;
# em Linux/WSL2, instalar NVIDIA Container Toolkit para usar --gpus all.
FROM ultralytics/ultralytics:latest

# Dependências comuns do OpenCV (leitura de vídeo / GUI opcional).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir pandas

COPY src/ /app/src/

ENV PYTHONUNBUFFERED=1

# Substitui em runtime, ex.:
# docker run ... racing-line-ai python /app/src/extract_line.py --video /data/v.mp4 --out /data/out --auto-select --start-x 100 --start-y 200
CMD ["python", "/app/src/extract_line.py", "--help"]
