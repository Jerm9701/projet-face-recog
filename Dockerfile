FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY eye-tracking/F03-face-tracking/retinaface-remake/*.py /app/

ENTRYPOINT ["python", "/app/infer_retinaface_r50.py"]