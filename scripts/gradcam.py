#!/usr/bin/env python3
"""Grad-CAM heatmaps over the trained classifier's last conv layer -- the
design doc's explainability floor, showing *where* the model is looking.
Run after train_classifier_head.py has produced models/classifier_head.pt.

    python scripts/gradcam.py --per-class 2
"""
import argparse
import pathlib

import matplotlib.cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets

from data_utils import build_transform
from train_contrastive import EncoderWithProjection


def gradcam_for_image(encoder, head, image_tensor, target_class):
    x = image_tensor.unsqueeze(0).requires_grad_(True)
    feature_map = encoder.features(x)
    feature_map.retain_grad()
    pooled = encoder.avgpool(feature_map).flatten(1)
    embedding = F.normalize(encoder.projection(pooled), dim=1)
    logits = head(embedding)
    encoder.zero_grad()
    head.zero_grad()
    logits[0, target_class].backward()

    grads = feature_map.grad
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feature_map).sum(dim=1)).squeeze(0)
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().numpy()


def overlay_heatmap(original: Image.Image, cam: np.ndarray, alpha=0.45) -> Image.Image:
    cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize(original.size, Image.BILINEAR)
    colored = (matplotlib.cm.jet(np.array(cam_resized) / 255.0)[:, :, :3] * 255).astype(np.uint8)
    base = np.array(original.convert("RGB")).astype(np.float32)
    blended = (1 - alpha) * base + alpha * colored.astype(np.float32)
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--model", default="models/classifier_head.pt")
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--out-dir", default="media/gradcam")
    args = ap.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    encoder = EncoderWithProjection()
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()
    head = torch.nn.Linear(128, len(checkpoint["classes"]))
    head.load_state_dict(checkpoint["head_state_dict"])
    head.eval()

    dataset = datasets.ImageFolder(args.data_dir, transform=build_transform())
    assert dataset.classes == checkpoint["classes"]

    seen_per_class = {c: 0 for c in range(len(dataset.classes))}
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, (path, label) in enumerate(dataset.samples):
        if seen_per_class[label] >= args.per_class:
            continue
        image_tensor, _ = dataset[idx]
        cam = gradcam_for_image(encoder, head, image_tensor, label)
        original = Image.open(path).resize((224, 224))
        overlay = overlay_heatmap(original, cam)
        species = dataset.classes[label]
        dest = out_dir / f"{species}_{seen_per_class[label]}.png"
        overlay.save(dest)
        print(f"wrote {dest}")
        seen_per_class[label] += 1
        if all(n >= args.per_class for n in seen_per_class.values()):
            break


if __name__ == "__main__":
    main()
