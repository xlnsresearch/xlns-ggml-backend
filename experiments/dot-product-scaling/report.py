#!/usr/bin/env python3

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import runlib  # noqa: E402
import plot  # noqa: E402

RESULTS = os.path.join(HERE, "results")
CONFIGS = ["base", "monte", "lpvip", "sum32", "xlns32"]
DISTS = ["signed", "positive", "attn"]
REPLAY_DIST = {"q": "signed", "ffn_up": "signed", "ffn_down": "signed", "attn_out": "signed", "kqv": "attn"}


def load(path):
    return runlib.load_jsonl(path)


def fit(points):
    pts = [(math.log(k), math.log(v)) for k, v in points if v > 0]
    if len(pts) < 3:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    a = sxy / sxx
    b = my - a * mx
    ss_tot = sum((p[1] - my) ** 2 for p in pts)
    ss_res = sum((p[1] - (a * p[0] + b)) ** 2 for p in pts)
    return a, math.exp(b), 1 - ss_res / ss_tot if ss_tot > 0 else 1.0


def interp(curve, k):
    ks = sorted(curve)
    if k <= ks[0]:
        return curve[ks[0]]
    if k >= ks[-1]:
        return curve[ks[-1]]
    for lo, hi in zip(ks, ks[1:]):
        if lo <= k <= hi:
            t = (math.log(k) - math.log(lo)) / (math.log(hi) - math.log(lo))
            return math.exp(math.log(curve[lo]) * (1 - t) + math.log(curve[hi]) * t)


FIGS = os.path.join(RESULTS, "figures")


def figure(w, name, made, alt):
    if made:
        w(f"![{alt}](figures/{name}.svg)\n")


def mode_series(by, dist, key):
    out = []
    for i, c in enumerate(CONFIGS):
        m = by.get((c, dist))
        if m:
            out.append((c, plot.slot(c, i), [(k, key(m[k])) for k in sorted(m)]))
    return out


def interp_lin(curve, k):
    ks = sorted(curve)
    if k <= ks[0]:
        return curve[ks[0]]
    if k >= ks[-1]:
        return curve[ks[-1]]
    for lo, hi in zip(ks, ks[1:]):
        if lo <= k <= hi:
            t = (math.log(k) - math.log(lo)) / (math.log(hi) - math.log(lo))
            return curve[lo] * (1 - t) + curve[hi] * t


def e(x):
    return "-" if x is None else f"{x:.2e}"


def main():
    synth = load(os.path.join(RESULTS, "synth.jsonl"))
    monte = load(os.path.join(RESULTS, "monte.jsonl"))
    out = []
    w = out.append
    w("# Dot-product error scaling of the LNS summation modes\n")
    w("Every number comes from the backend's own `mul_mat` kernels (LNSCUDA16 / LNSCUDA32, bit-identical to the "
      "CPU backends), driven through the ggml API by `dot-bench`. The kernel gets the original F32 inputs; the "
      "reference is the exact long-double sum of the products of the values the kernel actually uses after "
      "conversion (`to_f32(from_f32(x))`, read back from the same device), so `acc` measures only what the "
      "accumulation adds (LNS multiplication is exact). `total` is against the unconverted inputs and also "
      "includes the conversion. For xlns32 the read-back through F32 is itself rounded (~6e-8), which sets a "
      "floor on `acc` at small K.\n")
    w("Error of one dot product: e = (computed - exact) / sum|a_i b_i|. Over 4096 dot products per point: "
      "`bias` = mean(e), `sd` = std(e), `rms` = sqrt(mean(e^2)). For the positive distribution sum|a_i b_i| is "
      "the sum itself, so e is the plain relative error.\n")
    w("Distributions: `signed` a, b ~ N(0,1) (projection matmuls); `positive` |N(0,1)| (sums of squares, softmax "
      "denominators); `attn` a ~ N(0,1), b = softmax(2 N(0,1)) over K (attention-weighted sum of V, K = context "
      "length).\n")

    conv = {r["format"]: r for r in load(os.path.join(RESULTS, "convert.jsonl"))}
    if conv:
        w("## Converting F32 to LNS\n")
        w("Every LNS op converts its F32 inputs. 4M values with log-uniform magnitudes over 2^-20 .. 2^20 and random "
          "sign, converted once (`to_f32(from_f32(x))`), then twice and three times. Round-to-nearest would give a mean "
          "of about 0, equal shares rounded up and down, and no change on a second round trip.\n")
        w("| format | mean rel. error | rms rel. error | mean log2 | rounded down / up | changed by round trip 2 / 3 | drift log2 per round trip |")
        w("|---|---|---|---|---|---|---|")
        for f in ("xlns16", "xlns32"):
            c = conv.get(f)
            if c:
                w(f"| {f} | {c['mean_rel']:+.2e} | {c['rms_rel']:.2e} | {c['mean_log2']:+.2e} | "
                  f"{c['frac_down']*100:.0f}% / {c['frac_up']*100:.0f}% | {c['frac_changed_round2']*100:.0f}% / "
                  f"{c['frac_changed_round3']*100:.0f}% | {c['drift_log2_round2']:+.2e}, {c['drift_log2_round3']:+.2e} |")
        w("")

    by = {}
    for r in synth:
        by.setdefault((r["config"], r["dist"]), {})[r["K"]] = r

    for dist in DISTS:
        ks = sorted({k for (c, d), m in by.items() if d == dist for k in m})
        if not ks:
            continue
        w(f"## {dist}: rms accumulation error vs K\n")
        figure(w, f"rms-{dist}", plot.line_chart(
            os.path.join(FIGS, f"rms-{dist}.svg"), f"Accumulation error vs K, {dist} inputs",
            "rms of (computed - exact) / sum|a_i b_i| over 4096 dot products", mode_series(by, dist, lambda r: r["acc"]["rms"]),
            "dot-product length K", "rms error", xlog=True, ylog=True, xbase=2), f"rms error vs K, {dist}")
        if dist == "positive":
            figure(w, "bias-positive", plot.line_chart(
                os.path.join(FIGS, "bias-positive.svg"), "Bias of a sum of positive terms",
                "mean relative error in %; base mode loses the small terms as the sum grows",
                mode_series(by, dist, lambda r: r["acc"]["bias"] * 100), "dot-product length K", "mean error (%)",
                xlog=True, xbase=2), "bias vs K, positive")
        w("| K | " + " | ".join(CONFIGS) + " |")
        w("|---|" + "---|" * len(CONFIGS))
        for k in ks:
            w(f"| {k} | " + " | ".join(e(by.get((c, dist), {}).get(k, {}).get("acc", {}).get("rms")) for c in CONFIGS) + " |")
        w("")
        w(f"Bias and spread (`bias` / `sd`), {dist}:\n")
        w("| K | " + " | ".join(CONFIGS) + " |")
        w("|---|" + "---|" * len(CONFIGS))
        for k in ks:
            cells = []
            for c in CONFIGS:
                a = by.get((c, dist), {}).get(k, {}).get("acc")
                cells.append(f"{a['bias']:+.1e} / {a['sd']:.1e}" if a else "-")
            w(f"| {k} | " + " | ".join(cells) + " |")
        w("")

    w("## Power-law fits, K >= 64\n")
    w("Least squares on log-log: error = c * K^alpha. alpha = 0.5 is the random-walk growth of unbiased rounding, "
      "alpha = 1 is a systematic per-addition bias, alpha = 0 is a constant floor (rounding of the final result "
      "only). A low R^2 means the curve is not a power law (flat, or changing sign, as for Monte on positive "
      "sums), so its alpha is not a growth rate.\n")
    w("| config | dist | alpha (rms) | c | R^2 | alpha (abs bias) | alpha (sd) |")
    w("|---|---|---|---|---|---|---|")
    fits = {}
    for c in CONFIGS:
        for dist in DISTS:
            m = by.get((c, dist))
            if not m:
                continue
            pts = [(k, m[k]["acc"]["rms"]) for k in sorted(m) if k >= 64]
            f = fit(pts)
            fb = fit([(k, abs(m[k]["acc"]["bias"])) for k in sorted(m) if k >= 64])
            fs = fit([(k, m[k]["acc"]["sd"]) for k in sorted(m) if k >= 64])
            fits[(c, dist)] = f
            if f:
                w(f"| {c} | {dist} | {f[0]:.2f} | {f[1]:.2e} | {f[2]:.3f} | {fb[0]:.2f} | {fs[0]:.2f} |" if fb and fs else
                  f"| {c} | {dist} | {f[0]:.2f} | {f[1]:.2e} | {f[2]:.3f} | - | - |")
    w("")

    if monte:
        w("## Monte: averaging repeated stochastic sums\n")
        w("The same inputs summed with independently seeded LFSR dither; rms error of the mean of R repeats. "
          "Unbiased dither falls as 1/sqrt(R) until the bias remains.\n")
        kmax = max(r["K"] for r in monte)
        mser = []
        for i, dist in enumerate(DISTS):
            r = next((r for r in monte if r["dist"] == dist and r["K"] == kmax), None)
            if r:
                mser.append((dist, 5 + i, [(a["reps"], a["rms"] / r["avg"][0]["rms"]) for a in r["avg"]]))
        reps = [a["reps"] for a in monte[0]["avg"]]
        figure(w, "monte-averaging", plot.line_chart(
            os.path.join(FIGS, "monte-averaging.svg"), f"Monte: averaging R repeated sums, K = {kmax}",
            "rms error of the mean of R seeds, relative to one sum; unbiased dither would follow 1/sqrt(R)",
            mser, "repeats averaged (R)", "rms error / rms at R = 1", xlog=True, ylog=True, xbase=2, ybase=2,
            ref=[("1/sqrt(R)", [(r, 1 / math.sqrt(r)) for r in reps])], end_labels=True), "Monte averaging")
        for dist in DISTS:
            rows = sorted((r for r in monte if r["dist"] == dist), key=lambda r: r["K"])
            if not rows:
                continue
            reps = [a["reps"] for a in rows[0]["avg"]]
            w(f"**{dist}** (rms; last column: ratio R=1 / R={reps[-1]}, 1/sqrt(R) gives {math.sqrt(reps[-1]):.1f})\n")
            w("| K | " + " | ".join(f"R={x}" for x in reps) + " | ratio | bias R=1 |")
            w("|---|" + "---|" * (len(reps) + 2))
            for r in rows:
                v = [a["rms"] for a in r["avg"]]
                w(f"| {r['K']} | " + " | ".join(e(x) for x in v) + f" | {v[0]/v[-1]:.1f} | {r['avg'][0]['bias']:+.1e} |")
            w("")

    models = sorted(d for d in os.listdir(RESULTS) if os.path.isdir(os.path.join(RESULTS, d)))
    norm_pts = {}
    for model in models:
        rows = load(os.path.join(RESULTS, model, "replay.jsonl"))
        if not rows:
            continue
        w(f"## Replay with real operands: {model}\n")
        w("Real weights (dequantized from the GGUF) and real activations from an F32 forward pass, through the same "
          "kernels. `pred` is the synthetic rms at the same K, interpolated on the curve of the matching "
          "distribution (signed for projections, attn for kqv); `meas/pred` near 1 means the synthetic model "
          "predicts the real op. `total rel-L2` is ||computed - exact|| / ||exact|| over the op's outputs with "
          "unrounded inputs, the measure used by the layer-by-layer experiment.\n")
        cats, vals = [], {c: [] for c in CONFIGS}
        for op in ("q", "ffn_up", "ffn_down", "attn_out", "kqv"):
            for layer in sorted({r["layer"] for r in rows}):
                sel = [r for r in rows if r["op"] == op and r["layer"] == layer]
                if not sel:
                    continue
                kk = max(r["K"] for r in sel)
                cats.append(f"{op} L{layer}")
                for c in CONFIGS:
                    r = next((r for r in sel if r["config"] == c and r["K"] == kk), None)
                    curve = {k: v["acc"]["rms"] for k, v in by.get((c, REPLAY_DIST[op]), {}).items()}
                    vals[c].append(r["acc"]["rms"] / interp(curve, kk) if r and curve else None)
        short = model.split("-Instruct")[0]
        figure(w, f"replay-ratio-{short}", plot.bar_chart(
            os.path.join(FIGS, f"replay-ratio-{short}.svg"), f"{short}: real op error / synthetic prediction",
            "1 = the synthetic model predicts the real op exactly (kqv at the longest context)", cats,
            [(c, plot.slot(c, i), vals[c]) for i, c in enumerate(CONFIGS)], "measured rms / predicted rms",
            ref=[("1", 1.0)]), f"replay ratio {short}")
        kser = []
        for i, c in enumerate(CONFIGS):
            pts = {}
            for r in rows:
                if r["op"] == "kqv" and r["config"] == c and r["K"] >= 2:
                    pts.setdefault(r["K"], []).append(r["acc"]["rms"])
            if pts:
                kser.append((c, plot.slot(c, i), [(k, sum(v) / len(v)) for k, v in sorted(pts.items())]))
        figure(w, f"kqv-context-{short}", plot.line_chart(
            os.path.join(FIGS, f"kqv-context-{short}.svg"), f"{short}: attention (kqv) error vs context length",
            "real softmax weights x V, rms accumulation error, mean over the replayed layers", kser,
            "context length (tokens summed)", "rms error", xlog=True, ylog=True, xbase=2), f"kqv vs context {short}")
        w("| op | layer | K | config | meas rms | pred rms | meas/pred | bias | total rel-L2 |")
        w("|---|---|---|---|---|---|---|---|---|")
        for r in sorted(rows, key=lambda r: (r["op"], r["layer"], r["K"], CONFIGS.index(r["config"]))):
            if r["op"] not in REPLAY_DIST or (r["op"] == "kqv" and r["K"] < 8):
                continue
            curve = {k: v["acc"]["rms"] for k, v in by.get((r["config"], REPLAY_DIST[r["op"]]), {}).items()}
            pred = interp(curve, r["K"]) if curve else None
            ratio = r["acc"]["rms"] / pred if pred else None
            w(f"| {r['op']} | {r['layer']} | {r['K']} | {r['config']} | {e(r['acc']['rms'])} | {e(pred)} | "
              f"{'-' if ratio is None else f'{ratio:.2f}'} | {r['acc']['bias']:+.1e} | {r['total']['rel_l2']*100:.2f}% |")
        w("")

        norm = [r for r in rows if r["op"] in ("rms_norm", "softmax")]
        if norm:
            w(f"### Normalizers on real activations: {short}\n")
            w("Both are long sums of positive terms. In LNS the final multiply (rms_norm) and divide (softmax) are exact, "
              "so the outputs give the accumulated sum back exactly:\n")
            w("- `rms_norm` (input `ffn_inp-L`, K = width): relative error of the 1/rms the kernel used. Positive = the "
              "sum of squares came out too small, so every normalized output is that much too large. `pred` is "
              "(1+b)^(-1/2) - 1, with b the synthetic positive-sum bias at the same K.")
            w("- `softmax` (input `kq-L`, causal mask, scale 1/8): 1/sum(outputs) - 1, the relative error of the "
              "denominator, bucketed by K = position + 1. Negative = the denominator came out too small. `pred` is b "
              "itself.\n")
            w("| op | layer | K | config | rows | bias | sd | pred bias (positive-sum model) |")
            w("|---|---|---|---|---|---|---|---|")
            for r in sorted(norm, key=lambda r: (r["op"], r["layer"], r["K"], CONFIGS.index(r["config"]))):
                curve = {k: v["acc"]["bias"] for k, v in by.get((r["config"], "positive"), {}).items()}
                b = interp_lin(curve, r["K"]) if curve else None
                pred = None if b is None else ((1 + b) ** -0.5 - 1 if r["op"] == "rms_norm" else b)
                if r["op"] == "softmax" and r["K"] < 8:
                    continue
                if pred is not None and r["outputs"] >= 8:
                    norm_pts.setdefault(r["config"], []).append(
                        (pred * 100, r["acc"]["bias"] * 100, f"{short} {r['op']} L{r['layer']}"))
                w(f"| {r['op']} | {r['layer']} | {r['K']} | {r['config']} | {r['outputs']:.0f} | "
                  f"{r['acc']['bias']*100:+.2f}% | {r['acc']['sd']*100:.2f}% | {'-' if pred is None else f'{pred*100:+.2f}%'} |")
            w("")
            layers_n = sorted({r["layer"] for r in norm if r["op"] == "rms_norm"})
            figure(w, f"rms-norm-{short}", plot.bar_chart(
                os.path.join(FIGS, f"rms-norm-{short}.svg"), f"{short}: rms_norm on real activations",
                "error of 1/rms in %; positive = the sum of squares came out too small, so outputs are too large",
                [f"layer {L}" for L in layers_n],
                [(c, plot.slot(c, i), [next((r["acc"]["bias"] * 100 for r in norm if r["op"] == "rms_norm" and r["config"] == c
                                             and r["layer"] == L), None) for L in layers_n]) for i, c in enumerate(CONFIGS)],
                "error of 1/rms (%)"), f"rms_norm bias {short}")
            sser = []
            for i, c in enumerate(CONFIGS):
                pts = {}
                for r in norm:
                    if r["op"] == "softmax" and r["config"] == c and r["K"] >= 2:
                        pts.setdefault(r["K"], []).append(r["acc"]["bias"] * 100)
                if pts:
                    sser.append((c, plot.slot(c, i), [(k, sum(v) / len(v)) for k, v in sorted(pts.items())]))
            figure(w, f"softmax-{short}", plot.line_chart(
                os.path.join(FIGS, f"softmax-{short}.svg"), f"{short}: softmax denominator on real attention scores",
                "mean relative error of the denominator in %, over heads and replayed layers; negative = too small",
                sser, "context length (terms summed)", "denominator error (%)", xlog=True, xbase=2),
                f"softmax denominator {short}")

    if norm_pts:
        w("## Normalizers: measured against the positive-sum model\n")
        w("Every rms_norm and softmax row above (both models, all layers, softmax K >= 8, rows with at least 8 "
          "outputs) as one point: the bias predicted from the synthetic positive-sum curve at the same K against the "
          "bias measured on real activations. Points on the diagonal are predicted exactly.\n")
        figure(w, "normalizers-predicted", plot.scatter(
            os.path.join(FIGS, "normalizers-predicted.svg"), "Normalizers: measured vs predicted bias",
            "rms_norm (error of 1/rms) and softmax (error of the denominator), both models, in %; diagonal = exact",
            [(c, plot.slot(c, i), norm_pts[c]) for i, c in enumerate(CONFIGS) if c in norm_pts],
            "predicted from the positive-sum model (%)", "measured on real activations (%)", label_top=0, diag=True),
            "normalizers measured vs predicted")

    print("\n".join(out))


if __name__ == "__main__":
    main()
