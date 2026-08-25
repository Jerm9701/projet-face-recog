FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "F03-face-tracking/retinaface-remake/infer_retinaface_r50.py"]