from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import iresnet50


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    embedding_dim = checkpoint.get("embedding_dim", 512)

    model = iresnet50(embedding_dim=embedding_dim)
    state = checkpoint.get("backbone", checkpoint)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def preprocess():
    paper_std = 128.0 / 255.0
    return transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(paper_std, paper_std, paper_std),
            ),
        ]
    )


@torch.no_grad()
def embed(model, image_path: str, device: torch.device) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess()(image).unsqueeze(0).to(device)
    vector = F.normalize(model(tensor), dim=1)
    return vector.squeeze(0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    vectors = [embed(model, path, device) for path in args.images]

    for path, vector in zip(args.images, vectors):
        print(f"{path}: embedding shape={vector.shape}, norm={np.linalg.norm(vector):.6f}")

    if len(vectors) == 2:
        similarity = float(np.dot(vectors[0], vectors[1]))
        print(f"cosine_similarity={similarity:.6f}")


if __name__ == "__main__":
    main()
