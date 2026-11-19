import math
import os
from html import escape

LIGHT = {
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7",
    "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
}
DARK = {
    "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
    "grid": "#2c2c2a", "axis": "#383835",
    "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"],
}
MODE_SLOT = {"base": 0, "monte": 1, "lpvip": 2, "sum32": 3, "xlns32": 4}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

W, H = 760, 440
ML, MR, MT, MB = 76, 24, 92, 56


def slot(name, fallback):
    return MODE_SLOT.get(name, fallback)


def _style():
    def block(t):
        rules = [f"--surface:{t['surface']}", f"--ink:{t['ink']}", f"--ink2:{t['ink2']}", f"--muted:{t['muted']}",
                 f"--grid:{t['grid']}", f"--axis:{t['axis']}"]
        rules += [f"--s{i}:{c}" for i, c in enumerate(t["series"])]
        return ";".join(rules)
    css = [f"svg{{{block(LIGHT)}}}",
           f"@media (prefers-color-scheme: dark){{svg{{{block(DARK)}}}}}",
           f"text{{font-family:{FONT};fill:var(--ink2);font-size:12px}}",
           ".title{fill:var(--ink);font-size:15px;font-weight:600}",
           ".sub{fill:var(--ink2);font-size:12px}",
           ".tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}",
           ".lab{fill:var(--ink2);font-size:12px}",
           ".grid{stroke:var(--grid);stroke-width:1}",
           ".axis{stroke:var(--axis);stroke-width:1}",
           ".ref{stroke:var(--muted);stroke-width:1}"]
    css += [f".l{i}{{stroke:var(--s{i});stroke-width:2;fill:none;stroke-linejoin:round;stroke-linecap:round}}"
            f".f{i}{{fill:var(--s{i})}}" for i in range(8)]
    return "<style>" + "".join(css) + "</style>"


class Axis:
    def __init__(self, lo, hi, log, a, b, base=10):
        self.log = log
        self.base = base
        if log:
            lo, hi = math.log(lo, base), math.log(hi, base)
        if hi == lo:
            lo, hi = lo - 1, hi + 1
        self.lo, self.hi, self.a, self.b = lo, hi, a, b

    def __call__(self, v):
        if self.log:
            v = math.log(v, self.base)
        return self.a + (v - self.lo) / (self.hi - self.lo) * (self.b - self.a)

    def ticks(self):
        if self.log:
            k0, k1 = math.ceil(self.lo - 1e-9), math.floor(self.hi + 1e-9)
            ks = list(range(k0, k1 + 1))
            step = max(1, math.ceil(len(ks) / 7))
            return [self.base ** k for k in ks[::step]]
        span = self.hi - self.lo
        raw = span / 6
        mag = 10 ** math.floor(math.log10(raw))
        step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
        t = math.ceil(self.lo / step) * step
        out = []
        while t <= self.hi + 1e-12:
            out.append(round(t, 12))
            t += step
        return out


def fmt(v, log=False, base=10):
    if v == 0:
        return "0"
    if log and base == 2:
        return f"{int(v) // 1024}K" if v >= 1024 else f"{int(v)}" if v >= 1 else f"1/{int(round(1 / v))}"
    a = abs(v)
    if log:
        e = int(round(math.log10(a)))
        if -2 <= e <= 3:
            return f"{v:g}"
        return f"1e{e}"
    if a >= 10000:
        return f"{v / 1000:g}K"
    return f"{v:g}"


def _frame(title, subtitle, legend, h=H):
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="{W}" height="{h}" role="img">',
           f"<title>{escape(title)}</title><desc>{escape(subtitle)}</desc>", _style(),
           f'<rect width="{W}" height="{h}" rx="8" style="fill:var(--surface)"/>',
           f'<text class="title" x="{ML}" y="28">{escape(title)}</text>',
           f'<text class="sub" x="{ML}" y="48">{escape(subtitle)}</text>']
    x = ML
    for name, cls, kind in (legend if len(legend) > 1 else []):
        if kind == "line":
            out.append(f'<line class="{cls}" x1="{x}" y1="68" x2="{x + 18}" y2="68"/>')
            out.append(f'<circle class="{cls.replace("l", "f")}" cx="{x + 9}" cy="68" r="4" style="stroke:var(--surface);stroke-width:2"/>')
            x += 24
        else:
            out.append(f'<rect class="{cls}" x="{x}" y="62" width="12" height="12" rx="3"/>')
            x += 18
        out.append(f'<text class="lab" x="{x}" y="72">{escape(name)}</text>')
        x += 9 + 7 * len(name) + 16
    return out


def _axes(out, xa, ya, xlabel, ylabel, xbase=10, ybase=10, xticks=None, mr=MR):
    for t in ya.ticks():
        y = ya(t)
        out.append(f'<line class="grid" x1="{ML}" y1="{y:.1f}" x2="{W - mr}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{ML - 8}" y="{y + 4:.1f}" text-anchor="end">{fmt(t, ya.log, ybase)}</text>')
    for t in (xticks or xa.ticks()):
        x = xa(t)
        out.append(f'<line class="axis" x1="{x:.1f}" y1="{H - MB}" x2="{x:.1f}" y2="{H - MB + 4}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{H - MB + 18}" text-anchor="middle">{fmt(t, xa.log, xbase)}</text>')
    out.append(f'<line class="axis" x1="{ML}" y1="{H - MB}" x2="{W - mr}" y2="{H - MB}"/>')
    out.append(f'<text class="lab" x="{(ML + W - mr) / 2}" y="{H - 12}" text-anchor="middle">{escape(xlabel)}</text>')
    out.append(f'<text class="lab" transform="translate(16,{(MT + H - MB) / 2}) rotate(-90)" text-anchor="middle">{escape(ylabel)}</text>')


def _write(path, out):
    out.append("</svg>")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(out) + "\n")
    os.replace(tmp, path)


def line_chart(path, title, subtitle, series, xlabel, ylabel, xlog=False, ylog=False, xbase=10, ybase=10,
               ref=None, ylim=None, xticks=None, end_labels=False):
    pts = [(x, y) for _, _, p in series for x, y in p if (not xlog or x > 0) and (not ylog or y > 0)]
    if not pts:
        return False
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if ref:
        ys += [y for _, p in ref for _, y in p if not ylog or y > 0]
    ylo, yhi = (min(ys), max(ys)) if ylim is None else ylim
    if ylog:
        ylo, yhi = ybase ** math.floor(math.log(ylo, ybase)), ybase ** math.ceil(math.log(yhi, ybase))
    else:
        pad = (yhi - ylo) * 0.06 or 1
        ylo, yhi = (0 if ylo >= 0 and ylim is None else ylo - pad), yhi + pad
    mr = MR + (72 if end_labels else 0)
    xa = Axis(min(xs), max(xs), xlog, ML + 8, W - mr - 8, xbase)
    ya = Axis(ylo, yhi, ylog, H - MB, MT, ybase)
    out = _frame(title, subtitle, [(n, f"l{s}", "line") for n, s, _ in series])
    _axes(out, xa, ya, xlabel, ylabel, xbase, ybase, xticks, mr)
    if not ylog and ylo < 0 < yhi:
        out.append(f'<line class="axis" x1="{ML}" y1="{ya(0):.1f}" x2="{W - mr}" y2="{ya(0):.1f}"/>')
    for name, p in ref or []:
        p = [(x, y) for x, y in p if (not xlog or x > 0) and (not ylog or y > 0)]
        if len(p) >= 2:
            d = " ".join(f"{xa(x):.1f},{ya(y):.1f}" for x, y in p)
            out.append(f'<polyline class="ref" points="{d}" fill="none"/>')
            x, y = p[-1]
            out.append(f'<text class="tick" x="{xa(x) - 4:.1f}" y="{ya(y) - 6:.1f}" text-anchor="end">{escape(name)}</text>')
    for _, s, p in series:
        p = sorted((x, y) for x, y in p if (not xlog or x > 0) and (not ylog or y > 0))
        if not p:
            continue
        d = " ".join(f"{xa(x):.1f},{ya(y):.1f}" for x, y in p)
        out.append(f'<polyline class="l{s}" points="{d}"/>')
        for x, y in p:
            out.append(f'<circle class="f{s}" cx="{xa(x):.1f}" cy="{ya(y):.1f}" r="4" style="stroke:var(--surface);stroke-width:2"/>')
    if end_labels:
        for name, _, p in series:
            p = sorted((x, y) for x, y in p if (not xlog or x > 0) and (not ylog or y > 0))
            if p:
                out.append(f'<text class="lab" x="{xa(p[-1][0]) + 10:.1f}" y="{ya(p[-1][1]) + 4:.1f}">{escape(name)}</text>')
    _write(path, out)
    return True


def bar_chart(path, title, subtitle, categories, series, xlabel, xlog=False, ref=None, labels=False):
    vals = [v for _, _, vs in series for v in vs if v is not None and (not xlog or v > 0)]
    if not vals:
        return False
    lo, hi = min(vals + [0 if not xlog else min(vals)]), max(vals + ([0] if not xlog else []))
    if xlog:
        lo, hi = 10 ** math.floor(math.log10(lo)), 10 ** math.ceil(math.log10(hi))
    else:
        pad = (hi - lo) * 0.05 or 1
        lo, hi = (lo - pad if lo < 0 else 0), hi + pad
    left = max(ML + 70, int(20 + 6.6 * max(len(c) for c in categories)))
    h = max(H, MT + MB + len(categories) * (len(series) * 9 + 14))
    xa = Axis(lo, hi, xlog, left, W - MR - 8)
    out = _frame(title, subtitle, [(n, f"f{s}", "rect") for n, s, _ in series], h)
    for t in xa.ticks():
        x = xa(t)
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{MT}" x2="{x:.1f}" y2="{h - MB}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{h - MB + 18}" text-anchor="middle">{fmt(t, xlog)}</text>')
    out.append(f'<text class="lab" x="{(left + W - MR) / 2}" y="{h - 12}" text-anchor="middle">{escape(xlabel)}</text>')
    base = xa(lo) if xlog else xa(0)
    band = (h - MT - MB) / len(categories)
    bar = max(3.0, min(24.0, (band - 8) / len(series) - 2))
    for ci, cat in enumerate(categories):
        y0 = MT + ci * band + (band - len(series) * (bar + 2)) / 2
        out.append(f'<text class="lab" x="{left - 8}" y="{MT + ci * band + band / 2 + 4:.1f}" text-anchor="end">{escape(cat)}</text>')
        for si, (_, s, vs) in enumerate(series):
            v = vs[ci]
            if v is None or (xlog and v <= 0):
                continue
            y = y0 + si * (bar + 2)
            x1 = xa(v)
            a, b = (base, x1) if x1 >= base else (x1, base)
            wdt = max(b - a, 0.5)
            r = min(4, wdt, bar / 2)
            if x1 >= base:
                d = f"M{a:.1f},{y:.1f} H{b - r:.1f} Q{b:.1f},{y:.1f} {b:.1f},{y + r:.1f} V{y + bar - r:.1f} Q{b:.1f},{y + bar:.1f} {b - r:.1f},{y + bar:.1f} H{a:.1f} Z"
            else:
                d = f"M{b:.1f},{y:.1f} H{a + r:.1f} Q{a:.1f},{y:.1f} {a:.1f},{y + r:.1f} V{y + bar - r:.1f} Q{a:.1f},{y + bar:.1f} {a + r:.1f},{y + bar:.1f} H{b:.1f} Z"
            out.append(f'<path class="f{s}" d="{d}"/>')
            if labels:
                tx, anchor = (x1 + 6, "start") if x1 >= base else (x1 - 6, "end")
                out.append(f'<text class="lab" x="{tx:.1f}" y="{y + bar / 2 + 4:.1f}" text-anchor="{anchor}">{v:.3g}</text>')
    out.append(f'<line class="axis" x1="{base:.1f}" y1="{MT}" x2="{base:.1f}" y2="{h - MB}"/>')
    for name, v in ref or []:
        x = xa(v)
        out.append(f'<line class="ref" x1="{x:.1f}" y1="{MT}" x2="{x:.1f}" y2="{h - MB}"/>')
        out.append(f'<text class="tick" x="{x + 4:.1f}" y="{MT - 6}">{escape(name)}</text>')
    _write(path, out)
    return True


def scatter(path, title, subtitle, series, xlabel, ylabel, xlog=False, ylog=False, label_top=6, diag=False):
    pts = [(x, y) for _, _, p in series for x, y, _ in p if (not xlog or x > 0) and (not ylog or y > 0)]
    if not pts:
        return False
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    xlo, xhi, ylo, yhi = min(xs), max(xs), min(ys), max(ys)
    if diag:
        xlo = ylo = min(xlo, ylo)
        xhi = yhi = max(xhi, yhi)
    if xlog:
        xlo, xhi = 10 ** math.floor(math.log10(xlo)), 10 ** math.ceil(math.log10(xhi))
    if ylog:
        ylo, yhi = 10 ** math.floor(math.log10(ylo)), 10 ** math.ceil(math.log10(yhi))
    mr = MR + 60
    xa = Axis(xlo, xhi, xlog, ML + 8, W - mr - 8)
    ya = Axis(ylo, yhi, ylog, H - MB, MT)
    out = _frame(title, subtitle, [(n, f"l{s}", "line") for n, s, _ in series])
    _axes(out, xa, ya, xlabel, ylabel, mr=mr)
    if diag:
        out.append(f'<line class="ref" x1="{xa(xlo):.1f}" y1="{ya(xlo):.1f}" x2="{xa(xhi):.1f}" y2="{ya(xhi):.1f}"/>')
    for _, s, p in series:
        p = [(x, y, t) for x, y, t in p if (not xlog or x > 0) and (not ylog or y > 0)]
        top = sorted(p, key=lambda q: -q[1])[:label_top]
        for x, y, t in p:
            out.append(f'<circle class="f{s}" cx="{xa(x):.1f}" cy="{ya(y):.1f}" r="5" style="stroke:var(--surface);stroke-width:2"/>')
        boxes = [(xa(x) - 6, ya(y) - 6, xa(x) + 6, ya(y) + 6) for x, y, _ in p]
        for x, y, t in top:
            cx, cy, tw = xa(x), ya(y), 6.6 * len(t)
            for lx, ly, anchor in ((cx + 9, cy + 4, "start"), (cx - 9, cy + 4, "end"), (cx, cy - 10, "middle"), (cx, cy + 18, "middle")):
                x0 = lx if anchor == "start" else lx - tw if anchor == "end" else lx - tw / 2
                box = (x0, ly - 11, x0 + tw, ly + 3)
                if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in boxes):
                    break
            boxes.append(box)
            out.append(f'<text class="lab" x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}">{escape(t)}</text>')
    _write(path, out)
    return True
