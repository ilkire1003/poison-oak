#!/usr/bin/env python3
"""Classifier-guided diffusion counterfactuals: starting from a real lookalike
photo, run Stable Diffusion's img2img denoising (SDEdit -- keeps the edit close
to the original) while a step callback nudges the latents toward flipping
*our* trained classifier's prediction to poison-oak. Shows the smallest
realistic edit that would make the model change its mind, per the design
doc's diffusion explainability stretch goal (Grad-CAM shows *where* it looks;
this shows *what change* would matter). Depends on models/classifier_head.pt
-- run train_contrastive.py + train_classifier_head.py first.

    python scripts/diffusion_counterfactual.py --per-class 2
"""
import argparse
import pathlib

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from torchvision import datasets

from data_utils import IMAGENET_MEAN, IMAGENET_STD, build_transform
from train_contrastive import EncoderWithProjection

SD_MODEL = "runwayml/stable-diffusion-v1-5"


def classifier_logit(encoder, head, pixel_batch, target_class):
    # pixel_batch is SD latents decoded to [-1, 1] at 512x512; resize/normalize
    # to what the classifier was trained on (224x224, ImageNet stats)
    x = ((pixel_batch + 1) / 2).clamp(0, 1)
    x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return head(encoder(x))[0, target_class]


def make_guidance_callback(pipe, encoder, head, target_class, guidance_scale):
    def callback(_pipe, _step_index, _timestep, callback_kwargs):
        latents = callback_kwargs["latents"].detach().requires_grad_(True)
        pixels = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
        logit = classifier_logit(encoder, head, pixels, target_class)
        grad = torch.autograd.grad(logit, latents)[0]
        callback_kwargs["latents"] = (latents + guidance_scale * grad).detach()
        return callback_kwargs
    return callback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--model", default="models/classifier_head.pt")
    ap.add_argument("--target-class", default="poison-oak")
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--strength", type=float, default=0.4)
    ap.add_argument("--guidance-scale", type=float, default=40.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out-dir", default="media/diffusion")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    classes = checkpoint["classes"]
    target_idx = classes.index(args.target_class)

    encoder = EncoderWithProjection().to(device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()
    head = torch.nn.Linear(128, len(classes)).to(device)
    head.load_state_dict(checkpoint["head_state_dict"])
    head.eval()
    for p in list(encoder.parameters()) + list(head.parameters()):
        p.requires_grad = False

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        SD_MODEL, safety_checker=None, requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    dataset = datasets.ImageFolder(args.data_dir, transform=build_transform())
    assert dataset.classes == classes, "class order mismatch vs classifier checkpoint"

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_per_class = {c: 0 for c in range(len(classes)) if c != target_idx}
    for path, label in dataset.samples:
        if label == target_idx or seen_per_class[label] >= args.per_class:
            continue
        species = classes[label]
        original = Image.open(path).convert("RGB").resize((512, 512))
        callback = make_guidance_callback(pipe, encoder, head, target_idx, args.guidance_scale)
        result = pipe(
            prompt="", image=original, strength=args.strength,
            num_inference_steps=args.steps, guidance_scale=1.0,
            callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]
        n = seen_per_class[label]
        original.save(out_dir / f"{species}_{n}_original.png")
        result.save(out_dir / f"{species}_{n}_to-{args.target_class}.png")
        print(f"wrote {species}_{n} counterfactual pair")
        seen_per_class[label] += 1
        if all(n >= args.per_class for n in seen_per_class.values()):
            break


if __name__ == "__main__":
    main()
