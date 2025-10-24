#!/usr/bin/env python3
"""
Usage:
  python histogram.py --inputs data/outputs_task3_TSMC.jsonl --outdir data/histograms
"""
import argparse
import json
import os
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


def safe_filename(s: str) -> str:
    keep = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(c if c in keep else "_" for c in s).replace(" ", "_")


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def extract_weights(block: dict) -> Tuple[List[float], List[str]]:
    """Return (weights, texts) for a single continuations block."""
    samples = block.get("samples", [])
    norm = block.get("normalized_weights")
    texts = []
    weights = []

    if norm and isinstance(norm, list) and samples and len(norm) == len(samples):
        for s, w in zip(samples, norm):
            texts.append(s.get("text") if isinstance(s, dict) else (s if isinstance(s, str) else ""))
            weights.append(float(w))
        # normalize numerically
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        return weights, texts

    if samples:
        # samples as dicts with 'weight' and 'text'
        if isinstance(samples[0], dict):
            for s in samples:
                texts.append(s.get("text", ""))
                w = s.get("weight", None)
                weights.append(float(w) if w is not None else 1.0)
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
            else:
                weights = [1.0 / len(weights)] * len(weights)
            return weights, texts
        # samples as strings
        else:
            for s in samples:
                texts.append(s if isinstance(s, str) else str(s))
            if texts:
                weights = [1.0 / len(texts)] * len(texts)
            return weights, texts

    return [], []


def collect_by_method(path: str) -> Dict[str, List[List[float]]]:
    methods = {}
    for row in read_jsonl(path):
        for block in row.get("continuations", []):
            method = block.get("method", "Unknown")
            w, _ = extract_weights(block)
            if w:
                methods.setdefault(method, []).append(w)
    return methods


def save_hist(values: List[float], out_png: str, out_json: str, bins: int = 50):
    if not values:
        return
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=bins, range=(0.0, 1.0))
    plt.xlabel("Normalized weight")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    data = {
        "bins": int(bins),
        "bin_edges": edges.tolist(),
        "counts": counts.tolist(),
        "total": len(values),
        "min": float(min(values)),
        "max": float(max(values)),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    p = argparse.ArgumentParser(description="Make simple histograms of normalized weights.")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--outdir", default="data/histograms")
    p.add_argument("--bins", type=int, default=50)
    p.add_argument("--per-prompt", action="store_true", help="Also save histogram per prompt.")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    for path in args.inputs:
        if not os.path.exists(path):
            print(f"skip missing: {path}")
            continue
        base = os.path.splitext(os.path.basename(path))[0]
        method_map = collect_by_method(path)
        if not method_map:
            print(f"no weights found in {path}")
            continue

        for method, per_prompt_lists in method_map.items():
            flat = [w for lst in per_prompt_lists for w in lst]
            short = safe_filename(method)[:120]
            png = os.path.join(args.outdir, f"{base}__{short}__hist.png")
            jn = os.path.join(args.outdir, f"{base}__{short}__hist.json")
            save_hist(flat, png, jn, bins=args.bins)
            print(f"wrote {png} (n={len(flat)})")

            if args.per_prompt:
                sub = os.path.join(args.outdir, f"{base}__{short}__per_prompt")
                os.makedirs(sub, exist_ok=True)
                for i, wl in enumerate(per_prompt_lists):
                    if not wl:
                        continue
                    pngp = os.path.join(sub, f"prompt_{i:04d}.png")
                    jnp = os.path.join(sub, f"prompt_{i:04d}.json")
                    save_hist(wl, pngp, jnp, bins=args.bins)

    print("done.")


if __name__ == "__main__":
    main()
