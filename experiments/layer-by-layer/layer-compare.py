#!/usr/bin/env python3
"""Compare per-layer tensor dumps from experiments/layer-by-layer/layer-dump.cpp.

    python3 experiments/layer-by-layer/layer-compare.py --ref f32=DIR \
        --run cpu16-base=DIR --run cuda16-base=DIR ... \
        --pair cpu16-base:cuda16-base ... > report.md

For every run, each tensor is compared with the reference (the F32 CPU build):
    rel   relative RMS error  ||x - ref|| / ||ref||   (in %)
For every --pair (CPU vs GPU of the same arithmetic), each tensor is compared
bit for bit: how many tensors are identical, and the largest difference.

Output (markdown):
  1. final logits per run: rel error, cosine, top-1 agreement with the reference
  2. residual stream (l_out) per layer, per run
  3. error per tensor type (mean / max over layers), per run: where it enters
  4. CPU vs GPU per pair: identical tensors, first differing tensor, max diff
Plain Python, no numpy needed.
"""

import argparse
import array
import math
import os
import re
import sys


def load_index(d):
    rows = []
    with open(os.path.join(d, "index.tsv")) as f:
        for line in f:
            file, name, op, *ne = line.rstrip("\n").split("\t")
            rows.append((file, name, op))
    return rows


def load(d, file):
    a = array.array("f")
    with open(os.path.join(d, file), "rb") as f:
        a.frombytes(f.read())
    return a


def rel_err(x, r):
    num = sum((a - b) ** 2 for a, b in zip(x, r))
    den = sum(b * b for b in r)
    if den == 0:
        return 0.0 if num == 0 else float("inf")
    return math.sqrt(num / den)


def cosine(x, r):
    xy = sum(a * b for a, b in zip(x, r))
    xx = sum(a * a for a in x)
    rr = sum(b * b for b in r)
    return xy / math.sqrt(xx * rr) if xx > 0 and rr > 0 else float("nan")


def kind_and_layer(name):
    m = re.match(r"^(.*?)-(\d+)(#\d+)?$", name)
    if not m:
        return name, None
    return m.group(1) + (m.group(3) or ""), int(m.group(2))


def fmt(e):
    return "inf" if e == float("inf") else f"{100 * e:.3g}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--pair", action="append", default=[])
    ap.add_argument("--title", default="Layer-by-layer comparison")
    args = ap.parse_args()

    ref_label, ref_dir = args.ref.split("=", 1)
    runs = [r.split("=", 1) for r in args.run]
    index = load_index(ref_dir)
    for label, d in runs:
        names = [n for _, n, _ in load_index(d)]
        if names != [n for _, n, _ in index]:
            sys.exit(f"error: {label} has a different tensor list than {ref_label}")

    ref = {f: load(ref_dir, f) for f, _, _ in index}
    err = {}  # label -> name -> rel err
    data = {}
    for label, d in runs:
        data[label] = {f: load(d, f) for f, _, _ in index}
        err[label] = {n: rel_err(data[label][f], ref[f]) for f, n, _ in index}

    labels = [l for l, _ in runs]
    print(f"# {args.title}\n")
    print(f"Reference: `{ref_label}`. Errors are relative RMS errors against it "
          f"(||x - ref|| / ||ref||). {len(index)} tensors per run.\n")

    # 1. logits
    logits_file = next(f for f, n, _ in index if n == "result_output")
    r = ref[logits_file]  # last token's logits (llama outputs one row)
    top_r = max(range(len(r)), key=lambda i: r[i])
    print("## 1. Final logits\n")
    print("| Run | logits error | cosine | same top-1 token as reference |")
    print("|---|---|---|---|")
    for label in labels:
        x = data[label][logits_file]
        top_x = max(range(len(x)), key=lambda i: x[i])
        print(f"| {label} | {fmt(err[label]['result_output'])} | {cosine(x, r):.5f} | "
              f"{'yes' if top_r == top_x else 'no'} |")
    print()

    # 2. residual stream per layer
    layers = sorted({kind_and_layer(n)[1] for _, n, _ in index if kind_and_layer(n)[0] == "l_out"})
    print("## 2. Residual stream (`l_out`) error per layer\n")
    print("| Layer | " + " | ".join(labels) + " |")
    print("|---|" + "---|" * len(labels))
    for li in layers:
        print(f"| {li} | " + " | ".join(fmt(err[l][f'l_out-{li}']) for l in labels) + " |")
    for extra in ("result_norm", "result_output"):
        print(f"| {extra} | " + " | ".join(fmt(err[l][extra]) for l in labels) + " |")
    print()

    # 3. per tensor type
    kinds = []
    for _, n, _ in index:
        k, li = kind_and_layer(n)
        if li is not None and k not in kinds:
            kinds.append(k)
    print("## 3. Error per tensor type (mean / max over layers)\n")
    print("| Tensor | op | " + " | ".join(labels) + " |")
    print("|---|---|" + "---|" * len(labels))
    ops = {kind_and_layer(n)[0]: op for _, n, op in index}
    for k in kinds:
        names = [n for _, n, _ in index if kind_and_layer(n)[0] == k]
        cells = []
        for l in labels:
            es = [err[l][n] for n in names]
            cells.append(f"{fmt(sum(es) / len(es))} / {fmt(max(es))}")
        print(f"| {k} | {ops[k]} | " + " | ".join(cells) + " |")
    print()

    # 4. CPU vs GPU
    if args.pair:
        print("## 4. CPU vs GPU, same arithmetic (bit for bit, every tensor)\n")
        print("| Pair | identical tensors | first differing tensor | max abs difference |")
        print("|---|---|---|---|")
        for p in args.pair:
            a, b = p.split(":")
            same = 0
            first = "-"
            maxd = 0.0
            for f, n, _ in index:
                xa, xb = data[a][f], data[b][f]
                if xa.tobytes() == xb.tobytes():
                    same += 1
                else:
                    if first == "-":
                        first = n
                    maxd = max(maxd, max(abs(u - v) for u, v in zip(xa, xb)))
            print(f"| {a} vs {b} | {same} / {len(index)} | {first} | {maxd:.3g} |")
        print()


if __name__ == "__main__":
    main()
