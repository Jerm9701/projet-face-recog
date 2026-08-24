import boto3
from pathlib import Path

s3 = boto3.client("s3")

bucket = "projet-face-recog-models"
key = "retinaface/retinaface_r50.pth"

local_file = Path("models/retinaface_r50.pth")
local_file.parent.mkdir(parents=True, exist_ok=True)

if not local_file.exists():
    print("Téléchargement du modèle...")
    s3.download_file(bucket, key, str(local_file))

print("Modèle disponible :", local_file)