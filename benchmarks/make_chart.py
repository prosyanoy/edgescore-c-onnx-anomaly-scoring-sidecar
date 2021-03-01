"""Render docs/benchmark.svg from the numbers printed by `make bench-compare`.

    python benchmarks/make_chart.py            # uses the numbers recorded below
Edit ROWS after a new run so the README chart stays honest.
"""
from pathlib import Path

# (label, events/s, request p99 ms, is_edgescore)
ROWS = [
    ("Python baseline", 1067, 51.9, False),
    ("EdgeScore /v1/score", 33362, 5.0, True),
    ("EdgeScore HEC ingest*", 86834, 2.9, True),
]
W, ROW_H, LEFT, RIGHT, TOP = 820, 34, 190, 130, 78
INK, MUTED, GRID, BASE, ACCENT, SURFACE = "#1f2328", "#59636e", "#d8dee4", "#8c959f", "#0b6bcb", "#ffffff"


def panel(y0, title, sub, values, fmt):
    vmax = max(values)
    plot_w = W - LEFT - RIGHT
    out = [f'<text x="24" y="{y0}" font-size="15" font-weight="600" fill="{INK}">{title}</text>',
           f'<text x="24" y="{y0 + 18}" font-size="12" fill="{MUTED}">{sub}</text>']
    for i, (row, v) in enumerate(zip(ROWS, values)):
        y = y0 + 34 + i * ROW_H
        w = max(3.0, plot_w * v / vmax)
        col = ACCENT if row[3] else BASE
        out.append(f'<text x="{LEFT - 12}" y="{y + 17}" font-size="13" text-anchor="end" fill="{INK}">{row[0]}</text>')
        out.append(f'<path d="M{LEFT} {y} H{LEFT + w - 4:.1f} a4 4 0 0 1 4 4 V{y + 18} a4 4 0 0 1 -4 4 H{LEFT} Z" fill="{col}"/>')
        out.append(f'<text x="{LEFT + w + 8:.1f}" y="{y + 16}" font-size="13" font-weight="600" fill="{INK}">{fmt(v)}</text>')
    out.append(f'<line x1="{LEFT}" y1="{y0 + 30}" x2="{LEFT}" y2="{y0 + 34 + len(ROWS) * ROW_H}" stroke="{GRID}"/>')
    return out


eps = [r[1] for r in ROWS]
p99 = [r[2] for r in ROWS]
H = 78 + 2 * (34 + len(ROWS) * ROW_H + 40) + 30
parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif" role="img" '
         f'aria-label="EdgeScore versus Python baseline: events per second and request p99 latency">',
         f'<title>EdgeScore vs. Python baseline (same ONNX model)</title>',
         f'<rect width="{W}" height="{H}" rx="10" fill="{SURFACE}" stroke="{GRID}"/>',
         f'<text x="24" y="32" font-size="18" font-weight="700" fill="{INK}">EdgeScore (C++) vs. naive Python baseline</text>',
         f'<text x="24" y="52" font-size="12" fill="{MUTED}">Same ONNX model and API, 50 events/request, 2 clients, 2-vCPU VM shared with the load generator</text>']
parts += panel(TOP + 10, "Throughput, events per second", "higher is better", eps, lambda v: f"{v:,.0f}  ({v / eps[0]:.0f}x)" if v != eps[0] else f"{v:,.0f}")
parts += panel(TOP + 10 + 34 + len(ROWS) * ROW_H + 40, "Request p99 latency, ms (50 events per request)", "lower is better", p99, lambda v: f"{v:.1f} ms")
parts.append(f'<text x="24" y="{H - 14}" font-size="11" fill="{MUTED}">* HEC row deliberately overloads the queue; events/s counts accepted events until drained, excess got 503 (backpressure).</text>')
parts.append("</svg>")
out = Path(__file__).resolve().parents[1] / "docs" / "benchmark.svg"
out.write_text("\n".join(parts) + "\n")
print("wrote", out)
