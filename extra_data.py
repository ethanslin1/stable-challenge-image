"""Build a broader, UNLABELLED training set for the challenge by mixing in public image datasets.

Why: the challenge train.parquet is only 7,500 images from three narrow domains (Imagenette, Galaxy10 DECaLS,
EuroSAT). The leaderboard scores held-out images and NEW classes, so an encoder that has only seen those three
domains can overfit to them. This file downloads images from many other domains (objects, scenes, textures,
fine-grained animals/plants/cars, traffic signs, aerial imagery, ...), resizes them to 96x96 RGB, and merges
them with the original train.parquet into one parquet in the same format (columns: image, label).

All extra images get label = -1 (unlabelled), so the labelled 10% of the original data is still the only thing
used for monitoring, and no labels enter the SSL loss.

Put this file next to training_data.py, then:

    uv run extra_data.py --list                         # show the available sources
    uv run extra_data.py                                # default mix, 3000 images/source -> data/train_extended.parquet
    uv run extra_data.py --per-source 5000 --sources tiny_imagenet,sun397,dtd,pets
    uv run extra_data.py --hf user/some_dataset:train:image   # add any HuggingFace image dataset of your own

and in code (drop-in replacement for load_training_data):

    from extra_data import load_extended_data
    ds = load_extended_data()          # builds data/train_extended.parquet on first use, then just loads it

Notes
  * Each source is cached in data/extra/<name>.parquet, so re-running only downloads what is missing.
  * Needs network access and `datasets` + `pillow` (already in the challenge environment).
  * Dataset ids below are public HuggingFace datasets. If one has been renamed/removed or needs a login, that
    source is skipped with a warning and the rest still run; fix it by editing SOURCES or using --hf.
  * CHECK THE CHALLENGE RULES: CHALLENGE.md says the data is meant for self-supervised learning and does not
    (in the text we have seen) say whether extra data is allowed. Confirm with the organisers/GitHub before
    submitting a model trained on external data.
"""

import argparse
import os
import sys
import time

import numpy as np
from datasets import Dataset, Features, Image as HFImage, Value, concatenate_datasets, load_dataset
from PIL import Image

from training_data import UNLABELLED, load_training_data

SIZE = 96
HERE = os.path.dirname(os.path.abspath(__file__))
EXTRA_DIR = os.path.join(HERE, "data", "extra")
EXTENDED_PATH = os.path.join(HERE, "data", "train_extended.parquet")
FEATURES = Features({"image": HFImage(), "label": Value("int64")})

# name -> (HF repo, config or None, split, native resolution / what it adds)
SOURCES = {
    "tiny_imagenet": ("zh-plus/tiny-imagenet", None, "train", "64px, 200 everyday-object classes"),
    "caltech101": ("flwrlabs/caltech101", None, "train", "~300px, 101 object categories"),
    "food101": ("ethz/food101", None, "train", "~512px, food photos"),
    "flowers102": ("nelorth/oxford-flowers", None, "train", "~500px, fine-grained flowers"),
    "pets": ("timm/oxford-iiit-pet", None, "train", "~400px, cats/dogs, fine-grained"),
    "stl10": ("tanganke/stl10", None, "train", "96px native, 10 object classes"),
    "sun397": ("tanganke/sun397", None, "train", "scenes (indoor/outdoor), very diverse"),
    "dtd": ("tanganke/dtd", None, "train", "textures"),
    "resisc45": ("tanganke/resisc45", None, "train", "256px aerial/remote-sensing scenes"),
    "gtsrb": ("tanganke/gtsrb", None, "train", "traffic signs, small"),
    "cars": ("tanganke/stanford_cars", None, "train", "fine-grained cars"),
    "aircraft": ("tanganke/fgvc_aircraft", None, "train", "fine-grained aircraft"),
    # opt-in only: native 32px, so upsampled images are much blurrier than the 96px challenge images
    "cifar100": ("uoft-cs/cifar100", None, "train", "32px native (blurry when upsampled), opt-in"),
}
DEFAULT_SOURCES = [k for k in SOURCES if k != "cifar100"]


# ----------------------------------------------------------------------------------------- image handling
def to_96(im):
    """Any PIL image -> 96x96 RGB: shorter side resized to 96 (aspect kept), then centre crop."""
    try:
        im.draft("RGB", (2 * SIZE, 2 * SIZE))  # makes big JPEGs decode much faster; no-op otherwise
    except Exception:
        pass
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    w, h = im.size
    s = SIZE / min(w, h)
    nw, nh = max(SIZE, round(w * s)), max(SIZE, round(h * s))
    im = im.resize((nw, nh), Image.LANCZOS)
    l, t = (nw - SIZE) // 2, (nh - SIZE) // 2
    return im.crop((l, t, l + SIZE, t + SIZE))


def _image_column(ds):
    for k, f in ds.features.items():
        if isinstance(f, HFImage):
            return k
    for k in ("image", "img", "jpg", "png"):
        if k in ds.column_names:
            return k
    raise KeyError(f"no image column found in {ds.column_names}")


# ----------------------------------------------------------------------------------------- one source
def fetch_source(repo, config, split, n, seed, stream=True, image_col=None, shuffle_buffer=10_000):
    """Return up to n 96x96 RGB PIL images sampled from one HuggingFace dataset.

    stream=True : reads only what it needs (fast start, small download). The stream is shuffled with a buffer
                  and shard order, which is usually enough but is not a perfectly uniform sample if the
                  dataset is stored sorted by class.
    stream=False: downloads the whole split, then samples n indices uniformly at random (best sample, big download).
    """
    imgs = []
    if stream:
        ds = load_dataset(repo, config, split=split, streaming=True)
        col = image_col or _image_column(ds)
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
        it = iter(ds)
    else:
        ds = load_dataset(repo, config, split=split)
        col = image_col or _image_column(ds)
        idx = np.random.default_rng(seed).permutation(len(ds))[: int(n * 1.1) + 10]  # spare for bad images
        it = iter(ds.select(idx.tolist()))
    for ex in it:
        try:
            imgs.append(to_96(ex[col]))
        except Exception:
            continue  # skip an undecodable image
        if len(imgs) >= n:
            break
    return imgs


def build_source(name, spec, n, seed, stream, cache_dir):
    """Download (or load from cache) one source; returns its cached parquet path, or None on failure."""
    path = os.path.join(cache_dir, f"{name}_{n}_s{seed}.parquet")
    if os.path.isfile(path):
        print(f"[{name}] cached: {path}")
        return path
    repo, config, split, _ = spec
    t0 = time.time()
    try:
        imgs = fetch_source(repo, config, split, n, seed, stream)
    except Exception as e:
        print(f"[{name}] SKIPPED ({repo}): {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        return None
    if not imgs:
        print(f"[{name}] SKIPPED ({repo}): no images decoded", file=sys.stderr)
        return None
    os.makedirs(cache_dir, exist_ok=True)
    Dataset.from_dict({"image": imgs, "label": [UNLABELLED] * len(imgs)}, features=FEATURES).to_parquet(path)
    print(f"[{name}] {len(imgs)} images in {time.time() - t0:.0f}s -> {path}")
    return path


# ----------------------------------------------------------------------------------------- merge
def build_extended(sources=None, per_source=3000, out=EXTENDED_PATH, repeat_original=1, seed=0, stream=True,
                   extra_hf=(), train_path=None, cache_dir=EXTRA_DIR):
    """Download the extra sources and write original + extra images to one parquet (image, label)."""
    specs = {}
    for name in (sources or DEFAULT_SOURCES):
        if name not in SOURCES:
            raise KeyError(f"unknown source {name!r}; run with --list")
        specs[name] = SOURCES[name]
    for s in extra_hf:  # "repo[:split[:image_col]]"
        repo, *rest = s.split(":")
        split = rest[0] if rest else "train"
        specs[repo.replace("/", "__")] = (repo, None, split, "custom")

    parts = []
    for name, spec in specs.items():
        p = build_source(name, spec, per_source, seed, stream, cache_dir)
        if p:
            parts.append((name, Dataset.from_parquet(p).cast(FEATURES)))
    if not parts:
        raise RuntimeError("No extra source could be downloaded. Check your network connection / dataset ids.")

    orig = load_training_data(train_path).cast(FEATURES)  # original 7,500 (labels kept for monitoring)
    pieces = [orig] * max(1, repeat_original) + [d for _, d in parts]
    ds = concatenate_datasets(pieces)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    ds.to_parquet(out)

    n_extra = sum(len(d) for _, d in parts)
    print(f"\nwrote {out}: {len(ds)} images = {len(orig)} original x{max(1, repeat_original)} "
          f"+ {n_extra} extra from {len(parts)}/{len(specs)} sources "
          f"({sum(l != UNLABELLED for l in orig['label'])} labelled, all from the original set)")
    return out


def load_extended_data(path=None, labelled_only=False, **build_kwargs):
    """Like training_data.load_training_data(), but for the extended set. Builds it on first use."""
    path = path or EXTENDED_PATH
    if not os.path.isfile(path):
        build_extended(out=path, **build_kwargs)
    return load_training_data(path, labelled_only=labelled_only)


# ----------------------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", action="store_true", help="list the available sources and exit")
    ap.add_argument("--sources", default="default",
                    help="comma-separated source names, 'default' (all but cifar100) or 'all'")
    ap.add_argument("--hf", action="append", default=[], metavar="REPO[:SPLIT]",
                    help="extra HuggingFace image dataset, e.g. user/name:train (repeatable)")
    ap.add_argument("--per-source", type=int, default=3000, help="images taken from each source (default 3000)")
    ap.add_argument("--repeat-original", type=int, default=1,
                    help="repeat the original 7,500 images this many times so they are not drowned out")
    ap.add_argument("--no-stream", action="store_true",
                    help="download each full split and sample uniformly (better sample, much bigger download)")
    ap.add_argument("--train-path", default=None, help="path to an existing train.parquet (default: data/train.parquet)")
    ap.add_argument("--out", default=EXTENDED_PATH)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.list:
        for k, (repo, cfg, split, note) in SOURCES.items():
            print(f"{k:14s} {repo:28s} {note}")
        return
    names = list(SOURCES) if a.sources == "all" else DEFAULT_SOURCES if a.sources == "default" \
        else [s.strip() for s in a.sources.split(",") if s.strip()]
    build_extended(names, a.per_source, a.out, a.repeat_original, a.seed, not a.no_stream, a.hf, a.train_path)


if __name__ == "__main__":
    main()
