#!/usr/bin/env python3
"""Low-budget RetinaFace-like training on WIDER FACE.

Expected InsightFace/RetinaFace label.txt format:
    # 0--Parade/0_Parade_marchingband_1_849.jpg
    x y w h lx ly v rx ry v nx ny v lmx lmy v rmx rmy v ...

Example:
python train_retinaface_r50.py \
  --train-images /data/WIDER_FACE/WIDER_train/images \
  --train-labels /data/WIDER_FACE/WIDER_train/label.txt \
  --val-images /data/WIDER_FACE/WIDER_val/images \
  --val-labels /data/WIDER_FACE/WIDER_val/label.txt \
  --output-dir runs/retinaface_r50 --image-size 512 --batch-size 4
"""
from __future__ import annotations
import os
import argparse, json, math, random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import box_iou, nms, sigmoid_focal_loss
from tqdm import tqdm


@dataclass
class Config:
    train_images: str = os.path.join("data", "WIDER_FACE", "WIDER_train", "images")
    train_labels: str = os.path.join("data", "WIDER_FACE", "WIDER_train", "label.txt")
    val_images: str = os.path.join("data", "WIDER_FACE", "WIDER_val", "images")
    val_labels: str = os.path.join("data", "WIDER_FACE", "WIDER_val", "label.txt")
    output_dir: str = "runs/retinaface_r50"
    image_size: int = 512
    batch_size: int = 4
    accumulation_steps: int = 4
    workers: int = 4
    epochs: int = 20
    fpn_channels: int = 128
    backbone_lr: float = 1e-5
    head_lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    positive_iou: float = 0.5
    negative_iou: float = 0.3
    score_threshold: float = 0.35
    nms_threshold: float = 0.4
    freeze_backbone_epochs: int = 1
    amp: bool = True
    grad_clip: float = 5.0
    seed: int = 42
    resume: str = ""

STRIDES = (8, 16, 32)
ANCHOR_SIZES = ((16, 32), (64, 128), (256, 512))


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@dataclass
class Record:
    image: Path
    boxes: np.ndarray
    landmarks: np.ndarray
    landmark_valid: np.ndarray


def parse_label_file(image_root: Path, label_file: Path) -> List[Record]:
    records: List[Record] = []
    current: Optional[Path] = None
    boxes, landmarks, valid = [], [], []

    def flush() -> None:
        nonlocal current, boxes, landmarks, valid
        if current is None:
            return
        records.append(Record(
            image=image_root / current,
            boxes=np.asarray(boxes, np.float32).reshape(-1, 4),
            landmarks=np.asarray(landmarks, np.float32).reshape(-1, 5, 2),
            landmark_valid=np.asarray(valid, np.bool_).reshape(-1, 5),
        ))
        boxes, landmarks, valid = [], [], []

    with label_file.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                flush(); current = Path(line[1:].strip()); continue
            vals = [float(x) for x in line.split()]
            if len(vals) < 4:
                continue
            x, y, w, h = vals[:4]
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w, y + h])
            lm, vm = [], []
            for i in range(5):
                b = 4 + 3 * i
                if b + 1 < len(vals):
                    lx, ly = vals[b], vals[b + 1]
                    vis = vals[b + 2] if b + 2 < len(vals) else 1.0
                    ok = lx >= 0 and ly >= 0 and vis >= 0
                else:
                    lx, ly, ok = -1.0, -1.0, False
                lm.append([lx, ly]); vm.append(ok)
            landmarks.append(lm); valid.append(vm)
    flush()
    records = [r for r in records if r.image.is_file()]
    if not records:
        raise RuntimeError("No images found. Check --*-images and --*-labels paths.")
    return records


class WiderFaceDataset(Dataset):
    def __init__(self, image_root: str, label_file: str, size: int, training: bool):
        self.records = parse_label_file(Path(image_root), Path(label_file))
        self.size, self.training = size, training
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self): return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        image = cv2.imread(str(r.image), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read {r.image}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        boxes, lms, lm_valid = r.boxes.copy(), r.landmarks.copy(), r.landmark_valid.copy()
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
        image = cv2.resize(image, (self.size, self.size))
        sx, sy = self.size / w, self.size / h
        boxes[:, [0, 2]] *= sx; boxes[:, [1, 3]] *= sy
        lms[:, :, 0] *= sx; lms[:, :, 1] *= sy

        if self.training and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            x1, x2 = boxes[:, 0].copy(), boxes[:, 2].copy()
            boxes[:, 0], boxes[:, 2] = self.size - 1 - x2, self.size - 1 - x1
            lms[:, :, 0] = self.size - 1 - lms[:, :, 0]
            lms = lms[:, [1, 0, 2, 4, 3]]
            lm_valid = lm_valid[:, [1, 0, 2, 4, 3]]

        if self.training and random.random() < 0.5:
            a = random.uniform(0.85, 1.15); b = random.uniform(-15, 15)
            image = np.clip(image.astype(np.float32) * a + b, 0, 255).astype(np.uint8)

        tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0
        tensor = (tensor - self.mean) / self.std
        target = {
            "boxes": torch.from_numpy(boxes).float(),
            "landmarks": torch.from_numpy(lms).float(),
            "landmark_valid": torch.from_numpy(lm_valid),
        }
        return tensor, target


def collate(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


class Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = m.layer1, m.layer2, m.layer3, m.layer4

    def forward(self, x):
        x = self.stem(x); x = self.layer1(x)
        c3 = self.layer2(x); c4 = self.layer3(c3); c5 = self.layer4(c4)
        return c3, c4, c5


class FPN(nn.Module):
    def __init__(self, ch=128):
        super().__init__()
        self.l3, self.l4, self.l5 = nn.Conv2d(512, ch, 1), nn.Conv2d(1024, ch, 1), nn.Conv2d(2048, ch, 1)
        self.o3, self.o4, self.o5 = nn.Conv2d(ch, ch, 3, padding=1), nn.Conv2d(ch, ch, 3, padding=1), nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, c3, c4, c5):
        p5 = self.l5(c5)
        p4 = self.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        return [self.o3(p3), self.o4(p4), self.o5(p5)]


class Head(nn.Module):
    def __init__(self, ch=128, anchors_per_cell=2):
        super().__init__()
        self.tower = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(), nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU())
        self.cls = nn.Conv2d(ch, anchors_per_cell, 1)
        self.box = nn.Conv2d(ch, anchors_per_cell * 4, 1)
        self.lmk = nn.Conv2d(ch, anchors_per_cell * 10, 1)
        nn.init.constant_(self.cls.bias, -math.log(99.0))

    def forward(self, x):
        x = self.tower(x)
        return self.cls(x), self.box(x), self.lmk(x)


class RetinaFaceR50(nn.Module):
    def __init__(self, fpn_channels=128, pretrained=True):
        super().__init__()
        self.backbone, self.fpn, self.head = Backbone(pretrained), FPN(fpn_channels), Head(fpn_channels)

    def forward(self, images):
        features = self.fpn(*self.backbone(images))
        cls_all, box_all, lmk_all = [], [], []
        for f in features:
            cls, box, lmk = self.head(f); b = f.shape[0]
            cls_all.append(cls.permute(0, 2, 3, 1).reshape(b, -1))
            box_all.append(box.permute(0, 2, 3, 1).reshape(b, -1, 4))
            lmk_all.append(lmk.permute(0, 2, 3, 1).reshape(b, -1, 10))
        return {"classification": torch.cat(cls_all, 1), "boxes": torch.cat(box_all, 1), "landmarks": torch.cat(lmk_all, 1)}


def generate_anchors(size: int, device: torch.device) -> Tensor:
    out = []
    for stride, sizes in zip(STRIDES, ANCHOR_SIZES):
        n = math.ceil(size / stride)
        ys, xs = torch.meshgrid((torch.arange(n, device=device) + .5) * stride, (torch.arange(n, device=device) + .5) * stride, indexing="ij")
        centers = torch.stack([xs.flatten(), ys.flatten()], 1)
        for c in centers:
            for s in sizes:
                h = s / 2
                out.append(torch.stack([c[0]-h, c[1]-h, c[0]+h, c[1]+h]))
    return torch.stack(out).float()


def encode_boxes(a, b):
    aw, ah = a[:, 2]-a[:, 0], a[:, 3]-a[:, 1]
    acx, acy = (a[:, 0]+a[:, 2])/2, (a[:, 1]+a[:, 3])/2
    bw, bh = b[:, 2]-b[:, 0], b[:, 3]-b[:, 1]
    bcx, bcy = (b[:, 0]+b[:, 2])/2, (b[:, 1]+b[:, 3])/2
    return torch.stack([(bcx-acx)/aw, (bcy-acy)/ah, torch.log(bw.clamp_min(1)/aw), torch.log(bh.clamp_min(1)/ah)], 1)


def decode_boxes(a, d):
    aw, ah = a[:, 2]-a[:, 0], a[:, 3]-a[:, 1]
    acx, acy = (a[:, 0]+a[:, 2])/2, (a[:, 1]+a[:, 3])/2
    cx, cy = d[:, 0]*aw+acx, d[:, 1]*ah+acy
    w, h = torch.exp(d[:, 2].clamp(max=10))*aw, torch.exp(d[:, 3].clamp(max=10))*ah
    return torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], 1)


def encode_landmarks(a, lm):
    aw, ah = a[:, 2]-a[:, 0], a[:, 3]-a[:, 1]
    acx, acy = (a[:, 0]+a[:, 2])/2, (a[:, 1]+a[:, 3])/2
    out = lm.clone(); out[:, :, 0] = (lm[:, :, 0]-acx[:, None])/aw[:, None]; out[:, :, 1] = (lm[:, :, 1]-acy[:, None])/ah[:, None]
    return out.reshape(-1, 10)


def decode_landmarks(a, d):
    aw, ah = a[:, 2]-a[:, 0], a[:, 3]-a[:, 1]
    acx, acy = (a[:, 0]+a[:, 2])/2, (a[:, 1]+a[:, 3])/2
    d = d.reshape(-1, 5, 2); out = d.clone()
    out[:, :, 0] = d[:, :, 0]*aw[:, None]+acx[:, None]; out[:, :, 1] = d[:, :, 1]*ah[:, None]+acy[:, None]
    return out


def assign(anchors: Tensor, target: Dict[str, Tensor], cfg: Config):
    boxes = target["boxes"].to(anchors.device); lms = target["landmarks"].to(anchors.device); valid = target["landmark_valid"].to(anchors.device)
    n = len(anchors); labels = torch.full((n,), -1, dtype=torch.long, device=anchors.device)
    bt = torch.zeros((n, 4), device=anchors.device); lt = torch.zeros((n, 10), device=anchors.device); lm = torch.zeros((n, 10), dtype=torch.bool, device=anchors.device)
    if boxes.numel() == 0:
        labels.zero_(); return labels, bt, lt, lm
    iou = box_iou(anchors, boxes); best_iou, best_gt = iou.max(1)
    labels[best_iou < cfg.negative_iou] = 0; labels[best_iou >= cfg.positive_iou] = 1
    best_anchor = iou.argmax(0); labels[best_anchor] = 1; best_gt[best_anchor] = torch.arange(len(boxes), device=anchors.device)
    pos = labels == 1; matched = best_gt[pos]
    bt[pos] = encode_boxes(anchors[pos], boxes[matched]); lt[pos] = encode_landmarks(anchors[pos], lms[matched])
    lm[pos] = valid[matched, :, None].expand(-1, -1, 2).reshape(-1, 10)
    return labels, bt, lt, lm


def loss_fn(pred, targets, anchors, cfg):
    cls_losses, box_losses, lmk_losses, pos_counts = [], [], [], []
    for i, target in enumerate(targets):
        labels, bt, lt, lm = assign(anchors, target, cfg)
        valid_cls, pos = labels >= 0, labels == 1; normalizer = pos.sum().clamp_min(1)
        cls_losses.append(sigmoid_focal_loss(pred["classification"][i][valid_cls], labels[valid_cls].float(), alpha=.25, gamma=2, reduction="sum") / normalizer)
        if pos.any():
            box_losses.append(F.smooth_l1_loss(pred["boxes"][i][pos], bt[pos], beta=1/9, reduction="sum") / normalizer)
            pl, tl, mask = pred["landmarks"][i][pos], lt[pos], lm[pos]
            lmk_losses.append(F.smooth_l1_loss(pl[mask], tl[mask], beta=1/9, reduction="sum") / normalizer if mask.any() else pl.sum()*0)
        else:
            box_losses.append(pred["boxes"][i].sum()*0); lmk_losses.append(pred["landmarks"][i].sum()*0)
        pos_counts.append(normalizer.float())
    cls, box, lmk = map(lambda xs: torch.stack(xs).mean(), [cls_losses, box_losses, lmk_losses])
    return {"total": cls+box+lmk, "classification": cls, "box": box, "landmark": lmk, "positive": torch.stack(pos_counts).mean()}


@torch.inference_mode()
def postprocess(pred, anchors, size, score_thr, nms_thr):
    scores = torch.sigmoid(pred["classification"]); keep = scores >= score_thr
    if not keep.any():
        return {"boxes": torch.empty((0,4), device=scores.device), "scores": torch.empty(0, device=scores.device), "landmarks": torch.empty((0,5,2), device=scores.device)}
    scores, a = scores[keep], anchors[keep]
    boxes, lms = decode_boxes(a, pred["boxes"][keep]), decode_landmarks(a, pred["landmarks"][keep])
    boxes[:, [0,2]] = boxes[:, [0,2]].clamp(0, size-1); boxes[:, [1,3]] = boxes[:, [1,3]].clamp(0, size-1)
    if len(scores) > 5000:
        scores, idx = scores.topk(5000); boxes, lms = boxes[idx], lms[idx]
    idx = nms(boxes, scores, nms_thr)[:750]
    return {"boxes": boxes[idx], "scores": scores[idx], "landmarks": lms[idx]}


@torch.inference_mode()
def validate(model, loader, anchors, cfg, device, amp):
    model.eval(); sums = {k:0. for k in ["total","classification","box","landmark","positive"]}; batches=0; matched=gt_count=pred_count=0
    for images, targets in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            pred = model(images); losses = loss_fn(pred, targets, anchors, cfg)
        for k in sums: sums[k] += float(losses[k])
        batches += 1
        for i, target in enumerate(targets):
            out = postprocess({k:v[i] for k,v in pred.items()}, anchors, cfg.image_size, cfg.score_threshold, cfg.nms_threshold)
            gt = target["boxes"].to(device); pd = out["boxes"]
            gt_count += len(gt); pred_count += len(pd)
            if len(gt) and len(pd): matched += int((box_iou(pd, gt).max(0).values >= .5).sum())
    for k in sums: sums[k] /= max(batches,1)
    sums["recall50"] = matched/max(gt_count,1); sums["pred_per_image"] = pred_count/max(len(loader.dataset),1)
    return sums


def build_optimizer(model, cfg):
    return AdamW([
        {"params": model.backbone.parameters(), "lr": cfg.backbone_lr, "initial_lr": cfg.backbone_lr},
        {"params": list(model.fpn.parameters())+list(model.head.parameters()), "lr": cfg.head_lr, "initial_lr": cfg.head_lr},
    ], weight_decay=cfg.weight_decay)


def set_lr(optimizer, step, total, warmup):
    factor = (step+1)/max(warmup,1) if step < warmup else 0.05 + .95*.5*(1+math.cos(math.pi*(step-warmup)/max(total-warmup,1)))
    for g in optimizer.param_groups: g["lr"] = g["initial_lr"] * factor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-images", default="data/WIDER_FACE/WIDER_train/images")
    p.add_argument("--train-labels", default="data/WIDER_FACE/WIDER_train/label.txt")
    p.add_argument("--val-images", default="data/WIDER_FACE/WIDER_val/images")
    p.add_argument("--val-labels", default="data/WIDER_FACE/WIDER_val/label.txt")
    p.add_argument("--output-dir", default="runs/retinaface_r50"); p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4); p.add_argument("--accumulation-steps", type=int, default=4); p.add_argument("--workers", type=int, default=4); p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--fpn-channels", type=int, default=128); p.add_argument("--backbone-lr", type=float, default=1e-5); p.add_argument("--head-lr", type=float, default=1e-4); p.add_argument("--weight-decay", type=float, default=1e-4); p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--positive-iou", type=float, default=.5); p.add_argument("--negative-iou", type=float, default=.3); p.add_argument("--score-threshold", type=float, default=.35); p.add_argument("--nms-threshold", type=float, default=.4); p.add_argument("--freeze-backbone-epochs", type=int, default=1)
    p.add_argument("--no-amp", action="store_true"); p.add_argument("--grad-clip", type=float, default=5.); p.add_argument("--seed", type=int, default=42); p.add_argument("--resume", default="")
    a = p.parse_args()
    return Config(a.train_images,a.train_labels,a.val_images,a.val_labels,a.output_dir,a.image_size,a.batch_size,a.accumulation_steps,a.workers,a.epochs,a.fpn_channels,a.backbone_lr,a.head_lr,a.weight_decay,a.warmup_steps,a.positive_iou,a.negative_iou,a.score_threshold,a.nms_threshold,a.freeze_backbone_epochs,not a.no_amp,a.grad_clip,a.seed,a.resume)


def main():
    cfg = parse_args(); seed_all(cfg.seed); out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True); (out/"config.json").write_text(json.dumps(asdict(cfg), indent=2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = cfg.amp and device.type == "cuda"
    train_ds = WiderFaceDataset(cfg.train_images,cfg.train_labels,cfg.image_size,True); val_ds = WiderFaceDataset(cfg.val_images,cfg.val_labels,cfg.image_size,False)
    opts = dict(batch_size=cfg.batch_size,num_workers=cfg.workers,pin_memory=device.type=="cuda",persistent_workers=cfg.workers>0,collate_fn=collate)
    train_loader = DataLoader(train_ds,shuffle=True,drop_last=True,**opts); val_loader = DataLoader(val_ds,shuffle=False,**opts)
    model = RetinaFaceR50(cfg.fpn_channels, pretrained=not bool(cfg.resume)).to(device); optimizer = build_optimizer(model,cfg); scaler = torch.amp.GradScaler(device=device.type,enabled=amp)
    anchors = generate_anchors(cfg.image_size,device); start_epoch=global_step=0; best=float("inf")
    if cfg.resume:
        ckpt=torch.load(cfg.resume,map_location="cpu"); model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["optimizer"]); scaler.load_state_dict(ckpt.get("scaler",{})); start_epoch=ckpt["epoch"]+1; global_step=ckpt.get("global_step",0); best=ckpt.get("best",best)
    total_steps=math.ceil(len(train_loader)/cfg.accumulation_steps)*cfg.epochs
    print(f"device={device} amp={amp} train={len(train_ds)} val={len(val_ds)} anchors={len(anchors)} effective_batch={cfg.batch_size*cfg.accumulation_steps}")
    for epoch in range(start_epoch,cfg.epochs):
        freeze=epoch<cfg.freeze_backbone_epochs
        for p in model.backbone.parameters(): p.requires_grad=not freeze
        model.train(); model.backbone.eval() if freeze else None; optimizer.zero_grad(set_to_none=True)
        sums={k:0. for k in ["total","classification","box","landmark","positive"]}
        bar=tqdm(enumerate(train_loader),total=len(train_loader),desc=f"epoch {epoch+1}/{cfg.epochs}")
        for bi,(images,targets) in bar:
            images=images.to(device,non_blocking=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp): pred=model(images); losses=loss_fn(pred,targets,anchors,cfg); loss=losses["total"]/cfg.accumulation_steps
            scaler.scale(loss).backward()
            if (bi+1)%cfg.accumulation_steps==0 or bi+1==len(train_loader):
                set_lr(optimizer,global_step,total_steps,cfg.warmup_steps); scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); global_step+=1
            for k in sums: sums[k]+=float(losses[k].detach())
            n=bi+1; bar.set_postfix(loss=f"{sums['total']/n:.3f}",cls=f"{sums['classification']/n:.3f}",box=f"{sums['box']/n:.3f}",lmk=f"{sums['landmark']/n:.3f}",pos=f"{sums['positive']/n:.1f}")
        metrics=validate(model,val_loader,anchors,cfg,device,amp); print(f"val_loss={metrics['total']:.4f} cls={metrics['classification']:.4f} box={metrics['box']:.4f} lmk={metrics['landmark']:.4f} recall@0.5={metrics['recall50']:.3%} pred/image={metrics['pred_per_image']:.2f}")
        state={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),"epoch":epoch,"global_step":global_step,"best":best,"config":asdict(cfg)}; torch.save(state,out/"last.pt")
        if metrics["total"]<best: best=metrics["total"]; state["best"]=best; torch.save(state,out/"best.pt")

if __name__ == "__main__": main()
