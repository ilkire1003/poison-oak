#!/usr/bin/env python3
"""Light classifier head on top of the frozen contrastive embedding space
(design doc: "embedding space -> light classifier head"). Run after
train_contrastive.py has produced models/encoder.pt.

    python scripts/train_classifier_head.py
"""
import argparse
import pathlib

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from data_utils import build_transform, stratified_split
from train_contrastive import EncoderWithProjection


@torch.no_grad()
def extract_embeddings(encoder, loader, device):
    embeddings, labels = [], []
    for images, batch_labels in loader:
        embeddings.append(encoder(images.to(device)).cpu())
        labels.append(batch_labels)
    return torch.cat(embeddings), torch.cat(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--encoder", default="models/encoder.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default="models/classifier_head.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    checkpoint = torch.load(args.encoder, map_location=device, weights_only=True)
    encoder = EncoderWithProjection().to(device)
    encoder.load_state_dict(checkpoint["state_dict"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    dataset = datasets.ImageFolder(args.data_dir, transform=build_transform())
    assert dataset.classes == checkpoint["classes"], "class order mismatch vs encoder checkpoint"
    train_set, val_set, test_set = stratified_split(dataset, val_frac=0.15, test_frac=0.15)

    train_emb, train_labels = extract_embeddings(encoder, DataLoader(train_set, batch_size=args.batch_size), device)
    val_emb, val_labels = extract_embeddings(encoder, DataLoader(val_set, batch_size=args.batch_size), device)
    test_emb, test_labels = extract_embeddings(encoder, DataLoader(test_set, batch_size=args.batch_size), device)
    print(f"train/val/test sizes: {len(train_labels)}/{len(val_labels)}/{len(test_labels)}")

    embed_dim = train_emb.shape[1]
    head = nn.Linear(embed_dim, len(dataset.classes)).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    def accuracy(emb, labels):
        with torch.no_grad():
            return (head(emb.to(device)).argmax(1).cpu() == labels).float().mean().item()

    best_val_acc = 0.0
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        head.train()
        logits = head(train_emb.to(device))
        loss = loss_fn(logits, train_labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        head.eval()
        val_acc = accuracy(val_emb, val_labels)
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch {epoch}: train loss {loss.item():.3f} | val acc {val_acc:.3f}")
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "encoder_state_dict": encoder.state_dict(),
                "head_state_dict": head.state_dict(),
                "classes": dataset.classes,
            }, args.out)

    test_acc = accuracy(test_emb, test_labels)
    print(f"test acc {test_acc:.3f} | best val acc {best_val_acc:.3f}")
    print(f"final model saved to {args.out}")


if __name__ == "__main__":
    main()
