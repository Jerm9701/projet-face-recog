# ArcFace-style ResNet training on CASIA-WebFace

This project recreates the main face-recognition training ingredients described
in the ArcFace paper using Python 3 and PyTorch.

## What is reproduced

- 112 x 112 RGB face crops
- pixel normalization equivalent to `(pixel - 127.5) / 128`
- 3 x 3, stride-1 first convolution
- improved residual (`IR`) blocks with PReLU
- 512-dimensional embeddings
- output head similar to `BN -> Dropout(0.4) -> FC -> BN`
- ArcFace additive angular margin:
  - scale `s = 64`
  - angular margin `m = 0.5`
- SGD:
  - initial learning rate `0.1`
  - momentum `0.9`
  - weight decay `5e-4`
- iteration-based LR drops at:
  - 100,000
  - 140,000
  - 160,000
- maximum 200,000 optimizer steps
- cosine similarity for inference

The original paper used MXNet and multiple P40 GPUs with a global batch size of
512. The default PyTorch batch size here is 128 so it can run on more modest
hardware. Use gradient accumulation or distributed data parallel when you need
a global batch size of 512.

## Dataset layout

Download and extract CASIA-WebFace. The training directory must follow the
`ImageFolder` convention:

```text
CASIA-WebFace/
├── 0000045/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
├── 0000099/
│   ├── 001.jpg
│   └── ...
└── ...
```

Each folder name represents one identity.

## Important preprocessing note

The paper aligns faces using five landmarks: both eye centers, nose tip, and
both mouth corners. This repository resizes the available dataset images to
112 x 112, but it does not estimate landmarks.

For a closer reproduction, align every source image before training with a
face detector/landmark model such as RetinaFace or MTCNN, then store the aligned
112 x 112 crops in the directory structure above.

Do not combine CASIA-WebFace identities with evaluation identities without
checking identity overlap.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows PowerShell

pip install -r requirements.txt
```

## Training

A practical single-GPU run:

```bash
python train.py \
  --data-dir /path/to/CASIA-WebFace \
  --output-dir runs/casia_arcface \
  --batch-size 128 \
  --workers 8
```

Closer to the paper's global batch size on a sufficiently large GPU:

```bash
python train.py \
  --data-dir /path/to/CASIA-WebFace \
  --output-dir runs/casia_arcface_bs512 \
  --batch-size 512 \
  --workers 16
```

Resume:

```bash
python train.py \
  --data-dir /path/to/CASIA-WebFace \
  --output-dir runs/casia_arcface \
  --resume runs/casia_arcface/last.pt
```

## Outputs

- `last.pt`: complete latest training state
- `best.pt`: complete state with lowest validation loss
- `backbone_final.pt`: lightweight embedding backbone
- `config.json`: resolved command-line configuration

## Extract embeddings and compare two faces

```bash
python extract_embeddings.py \
  --checkpoint runs/casia_arcface/backbone_final.pt \
  face_a.jpg face_b.jpg
```

The script prints the cosine similarity of the two L2-normalized embeddings.

## Validation interpretation

The small internal split reports classification loss and top-1 identity
accuracy. It is useful for detecting broken training, but it is not a standard
open-set face-verification benchmark.

For research-quality evaluation, use an identity-disjoint benchmark such as
LFW and compute verification accuracy with cross-validated cosine thresholds.
