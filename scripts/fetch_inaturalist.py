#!/usr/bin/env python3
"""Pull research-grade, usably-licensed photos from iNaturalist for the species
in species.json. Counts-only by default (Week-1-Day-1 coverage check);
pass --fetch to actually download images.

    python scripts/fetch_inaturalist.py                 # counts only
    python scripts/fetch_inaturalist.py --fetch --smoke  # download 5/species to verify
    python scripts/fetch_inaturalist.py --fetch          # real pull
"""
import argparse
import csv
import json
import pathlib
import sys
import time

import requests

API = "https://api.inaturalist.org/v1/observations"
LICENSES = "cc0,cc-by"  # per design doc: prefer CC0, accept CC-BY
HEADERS = {"User-Agent": "poison-oak-id-trainer/0.1 (educational demo dataset collection)"}
PHOTO_SIZE = "medium"  # ~500px side, keeps the dataset well under the storage cap


def get_count(species_name: str) -> int:
    resp = requests.get(
        API,
        params={"taxon_name": species_name, "quality_grade": "research",
                "photos": "true", "photo_license": LICENSES, "per_page": 1},
        headers=HEADERS, timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["total_results"]


def iter_observations(species_name: str, per_page: int = 200):
    page = 1
    while True:
        resp = requests.get(
            API,
            params={"taxon_name": species_name, "quality_grade": "research",
                    "photos": "true", "photo_license": LICENSES,
                    "per_page": per_page, "page": page, "order_by": "id"},
            headers=HEADERS, timeout=30,
        )
        resp.raise_for_status()
        results = resp.json()["results"]
        if not results:
            return
        yield from results
        page += 1
        time.sleep(1)  # stay well under iNaturalist's rate limit


def download_species(species: dict, out_dir: pathlib.Path, cap: int, manifest_writer, manifest_writer_file) -> tuple[int, int]:
    slug, name = species["slug"], species["name"]
    species_dir = out_dir / slug
    species_dir.mkdir(parents=True, exist_ok=True)
    n_images = 0
    n_bytes = 0
    for obs in iter_observations(name):
        if n_images >= cap:
            break
        for photo in obs.get("photos", []):
            if n_images >= cap:
                break
            url = photo["url"].replace("square", PHOTO_SIZE)
            dest = species_dir / f"{obs['id']}_{photo['id']}.jpg"
            if dest.exists():
                n_images += 1
                continue
            try:
                r = requests.get(url, headers=HEADERS, timeout=30)
                r.raise_for_status()
                dest.write_bytes(r.content)
            except (requests.RequestException, OSError) as e:
                print(f"  skip {slug} obs {obs['id']} photo {photo['id']}: {e}", file=sys.stderr)
                continue
            n_images += 1
            n_bytes += len(r.content)
            manifest_writer.writerow([
                slug, name, obs["id"], photo["id"], dest.relative_to(out_dir.parent),
                photo.get("license_code"), obs.get("uri"),
            ])
            manifest_writer_file.flush()
    return n_images, n_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species-file", default="scripts/species.json")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--fetch", action="store_true", help="download images, not just counts")
    ap.add_argument("--smoke", action="store_true", help="cap at 5 images/species to verify the pipeline")
    ap.add_argument("--cap", type=int, default=750, help="max images per species when --fetch is set")
    args = ap.parse_args()

    species_list = json.loads(pathlib.Path(args.species_file).read_text())
    cap = 5 if args.smoke else args.cap

    print(f"{'species':<28} {'role':<10} {'usable research-grade photos (cc0/cc-by)'}")
    counts = {}
    for sp in species_list:
        try:
            c = get_count(sp["name"])
        except requests.RequestException as e:
            print(f"  ERROR fetching count for {sp['name']}: {e}", file=sys.stderr)
            continue
        counts[sp["slug"]] = c
        print(f"{sp['name']:<28} {sp['role']:<10} {c}")
        time.sleep(1)

    if not args.fetch:
        return

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir.parent / "manifest.csv"
    write_header = not manifest_path.exists()
    with manifest_path.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["slug", "scientific_name", "observation_id", "photo_id", "path", "license_code", "observation_url"])
        total_images = 0
        total_bytes = 0
        for sp in species_list:
            n_images, n_bytes = download_species(sp, out_dir, cap, w, f)
            total_images += n_images
            total_bytes += n_bytes
            print(f"  {sp['slug']}: downloaded {n_images} images ({n_bytes/1e6:.1f} MB)")
    print(f"Total: {total_images} images, {total_bytes/1e9:.2f} GB, manifest at {manifest_path}")


if __name__ == "__main__":
    main()
