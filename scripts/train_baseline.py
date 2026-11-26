#!/usr/bin/env python3
"""Baseline poison-oak-vs-lookalikes classifier: frozen pretrained backbone +
linear head, trained on data/raw/<species>/*.jpg. CPU-friendly (head-only
training), per the Week 1 baseline-classifier step in the design doc.

    python scripts/train_baseline.py                  # default run
    python scripts/train_baseline.py --epochs 10
"""
import argparse
import pathlib
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def stratified_split(dataset, val_frac, test_frac, seed=0):
    by_class = {}
    for idx, (_, label) in enumerate(dataset.samples):
        by_class.setdefault(label, []).append(idx)
    rng = random.Random(seed)
    train_idx, val_idx, test_idx = [], [], []
    for label, idxs in by_class.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_val = max(1, int(n * val_frac))
        n_test = max(1, int(n * test_frac))
        val_idx += idxs[:n_val]
        test_idx += idxs[n_val:n_val + n_test]
        train_idx += idxs[n_val + n_test:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--out", default="models/baseline.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # rescale every image to the backbone's expected 224x224 input
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    print(f"classes: {dataset.classes}")
    if len(dataset.classes) < 2:
        raise SystemExit(f"need >=2 species folders with images in {args.data_dir}, found {dataset.classes}")

    train_set, val_set, test_set = stratified_split(dataset, args.val_frac, args.test_frac)
    print(f"train/val/test sizes: {len(train_set)}/{len(val_set)}/{len(test_set)}")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    for p in model.parameters():
        p.requires_grad = False
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(dataset.classes))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.classifier[-1].parameters(), lr=1e-3)

    best_val_acc = 0.0
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device)
        print(f"epoch {epoch}: train loss {train_loss:.3f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.3f} acc {val_acc:.3f}")
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({"state_dict": model.state_dict(), "classes": dataset.classes}, args.out)

    test_loss, test_acc = run_epoch(model, test_loader, device)
    print(f"test loss {test_loss:.3f} acc {test_acc:.3f}")
    print(f"best checkpoint saved to {args.out} (val acc {best_val_acc:.3f})")


if __name__ == "__main__":
    main()
