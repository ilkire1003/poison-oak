"""Shared dataset loading/splitting used by the training scripts."""
import random

from torchvision import transforms
from torch.utils.data import Subset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform():
    # rescale every image to the backbone's expected 224x224 input
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


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
