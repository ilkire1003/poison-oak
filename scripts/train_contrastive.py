#!/usr/bin/env python3
"""Supervised contrastive (InfoNCE/SupCon) pretraining of the embedding
space, per the design doc's chosen model architecture: same-species images
are positives, every other species in the batch is a negative, so the
embedding space itself encodes "contrasting cases." Full backbone
fine-tuning is heavy -- meant for HPC (gpua100), not the desktop.

    python scripts/train_contrastive.py                 # full fine-tune (HPC)
    python scripts/train_contrastive.py --freeze-backbone --epochs 2  # cheap dry run
"""
import argparse

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from data_utils import IMAGENET_MEAN, IMAGENET_STD, stratified_split


class EncoderWithProjection(nn.Module):
    def __init__(self, embed_dim=128, freeze_backbone=False):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        feat_dim = backbone.classifier[0].in_features
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(inplace=True), nn.Linear(256, embed_dim),
        )

    def forward(self, x):
        x = self.avgpool(self.features(x)).flatten(1)
        z = self.projection(x)
        return F.normalize(z, dim=1)


def supcon_loss(embeddings, labels, temperature=0.1):
    sim = embeddings @ embeddings.T / temperature
    sim.fill_diagonal_(float("-inf"))
    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
    same_label.fill_diagonal_(False)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    has_positive = same_label.any(dim=1)
    # the diagonal of log_prob is -inf (self-similarity was masked out above); torch.where
    # avoids multiplying that -inf by a 0 mask, which would otherwise produce NaN
    masked_log_prob = torch.where(same_label, log_prob, torch.zeros_like(log_prob))
    per_anchor_loss = -masked_log_prob.sum(dim=1) / same_label.sum(dim=1).clamp(min=1)
    return per_anchor_loss[has_positive].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--out", default="models/encoder.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, freeze_backbone: {args.freeze_backbone}")

    # light augmentation so same-image views aren't trivially identical
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    print(f"classes: {dataset.classes}")
    train_set, val_set, _ = stratified_split(dataset, val_frac=0.15, test_frac=0.15)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    model = EncoderWithProjection(freeze_backbone=args.freeze_backbone).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, n_batches = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)
            loss = supcon_loss(embeddings, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(n_batches, 1)

        model.eval()
        val_loss, n_val_batches = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                embeddings = model(images)
                val_loss += supcon_loss(embeddings, labels).item()
                n_val_batches += 1
        val_loss /= max(n_val_batches, 1)

        print(f"epoch {epoch}: train supcon loss {train_loss:.3f} | val supcon loss {val_loss:.3f}")
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            torch.save({"state_dict": model.state_dict(), "classes": dataset.classes}, args.out)

    print(f"best encoder checkpoint saved to {args.out} (val supcon loss {best_val_loss:.3f})")


if __name__ == "__main__":
    main()
