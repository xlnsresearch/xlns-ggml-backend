#!/usr/bin/env python3

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.dirname(HERE))
import runlib  # noqa: E402

LLAMA = os.environ.get("LLAMA_ROOT", os.path.join(REPO, "build", "llama.cpp"))
WORK = os.environ.get("WORK_DIR", os.path.join(REPO, "build", "exp-dot"))
RESULTS = os.path.join(HERE, "results")

CONFIGS = [
    ("base", "lns-base", "LNSCUDA16"),
    ("monte", "lns-monte", "LNSCUDA16"),
    ("lpvip", "lns-lpvip", "LNSCUDA16"),
    ("sum32", "lns-sum32", "LNSCUDA16"),
    ("xlns32", "lns-base", "LNSCUDA32"),
]
DISTS = ["signed", "positive", "attn"]
REPLAY_OPS = ["q", "ffn_up", "ffn_down", "attn_out", "kqv", "rms_norm", "softmax"]


def build(src, tree, name):
    exe = os.path.join(WORK, f"{name}-{tree}")
    lib = os.path.join(LLAMA, f"build-{tree}", "bin")
    if not os.path.exists(os.path.join(lib, "libllama.so")):
        raise SystemExit(f"{lib} is missing: build the trees first (./setup.sh --cuda, or this experiment's run.sh)")
    if os.path.exists(exe) and os.path.getmtime(exe) > os.path.getmtime(src):
        return exe
    runlib.run(["g++", "-O2", "-std=c++17", "-I", os.path.join(LLAMA, "include"), "-I", os.path.join(LLAMA, "ggml", "include"),
                src, "-o", exe + ".part", "-L", lib, "-lllama", "-lggml", "-lggml-base", f"-Wl,-rpath,{lib}"], check=True)
    runlib.publish(exe + ".part", exe)
    return exe


def bench(tree):
    return build(os.path.join(HERE, "dot-bench.cpp"), tree, "dot-bench")


def run_lines(cmd):
    env = dict(os.environ, GGML_LNS_BACKEND="cuda16,cuda32")
    p = runlib.run(cmd, env=env)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-3000:])
        raise SystemExit(f"failed: {' '.join(cmd)}")
    return [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]


def done_keys(path, fields):
    return {tuple(r.get(k) for k in fields) for r in runlib.load_jsonl(path)}


def append_group(path, rows):
    runlib.append_jsonl_many(path, rows)


def synth(kmax):
    path = os.path.join(RESULTS, "synth.jsonl")
    done = done_keys(path, ("config", "dist", "K"))
    for name, tree, dev in CONFIGS:
        for dist in DISTS:
            if all((name, dist, k) in done for k in ks(kmax)):
                continue
            rows = run_lines([bench(tree), "synth", "--device", dev, "--dist", dist, "--kmax", str(kmax)])
            for r in rows:
                r["config"] = name
            append_group(path, [r for r in rows if (name, dist, r["K"]) not in done])
            print(f"  synth {name:7s} {dist:9s} K={rows[0]['K']}..{rows[-1]['K']} "
                  f"rms {rows[0]['acc']['rms']:.2e}..{rows[-1]['acc']['rms']:.2e}", flush=True)


def ks(kmax):
    k, out = 16, []
    while k <= kmax:
        out.append(k)
        k *= 2
    return out


def convert():
    path = os.path.join(RESULTS, "convert.jsonl")
    done = done_keys(path, ("format",))
    for fmt, tree, dev in (("xlns16", "lns-base", "LNSCUDA16"), ("xlns32", "lns-base", "LNSCUDA32")):
        if (fmt,) in done:
            continue
        rows = run_lines([bench(tree), "convert", "--device", dev, "--outputs", "4194304"])
        for r in rows:
            r["format"] = fmt
        append_group(path, rows)
        print(f"  convert {fmt}: mean rel {rows[0]['mean_rel']:+.2e}, rounded down {rows[0]['frac_down']*100:.0f}%", flush=True)


def monte(kmax):
    path = os.path.join(RESULTS, "monte.jsonl")
    done = done_keys(path, ("dist", "K"))
    for dist in DISTS:
        if all((dist, k) in done for k in ks(kmax)):
            continue
        rows = run_lines([bench("lns-monte"), "synth", "--device", "LNSCUDA16", "--dist", dist, "--kmax", str(kmax),
                          "--reps", "32", "--outputs", "2048"])
        append_group(path, [r for r in rows if (dist, r["K"]) not in done])
        print(f"  monte-avg {dist}", flush=True)


def f32_dump(model, name, text):
    dump = os.path.join(WORK, f"{name}-f32-dump")
    if os.path.exists(os.path.join(dump, "index.tsv")):
        return dump
    part = dump + ".part"
    runlib.remove(part)
    os.makedirs(part)
    dumper = build(os.path.join(REPO, "experiments", "layer-by-layer", "layer-dump.cpp"), "ref", "layer-dump")
    runlib.run([dumper, model, text, part], stdout=False, check=True)
    os.rename(part, dump)
    return dump


def replay(model, text):
    name = os.path.basename(model)[:-5]
    out_dir = os.path.join(RESULTS, name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "replay.jsonl")
    dump = f32_dump(model, name, text)
    with open(os.path.join(dump, "index.tsv")) as f:
        layers = sorted({int(l.split("\t")[1].split("-")[1].split("#")[0]) for l in f if l.split("\t")[1].startswith("l_out-")})
    picks = sorted({layers[0], layers[len(layers) // 2], layers[-1]})
    done = done_keys(path, ("config", "layer", "op"))
    for cname, tree, dev in CONFIGS:
        for layer in picks:
            todo = [op for op in REPLAY_OPS if (cname, layer, op) not in done]
            if not todo:
                continue
            rows = run_lines([bench(tree), "replay", "--device", dev, "--model", model, "--dump", dump, "--layer", str(layer),
                              "--ops", ",".join(todo)])
            for r in rows:
                r["config"] = cname
            append_group(path, rows)
            print(f"  replay {name} {cname} layer {layer}: {len(rows)} rows", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Dot-product error scaling of the LNS summation modes, synthetic and replayed")
    ap.add_argument("-m", dest="models", action="append", default=[])
    ap.add_argument("--parts", default="convert,synth,monte,replay")
    ap.add_argument("--kmax", type=int, default=16384)
    ap.add_argument("--text", default=os.path.join(HERE, "long-prompt.txt"))
    args = ap.parse_args()
    parts = set(args.parts.split(","))
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    runlib.on_cleanup(lambda: runlib.remove(os.path.join(WORK, "*.part")))
    if "convert" in parts:
        convert()
    if "synth" in parts:
        synth(args.kmax)
    if "monte" in parts:
        monte(args.kmax)
    if "replay" in parts:
        with open(args.text) as f:
            text = " ".join(f.read().split())
        for m in args.models:
            replay(os.path.abspath(m), text)


if __name__ == "__main__":
    runlib.main(main)
