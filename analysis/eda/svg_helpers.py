"""Small theme-aware SVG chart builders for the report page."""
from __future__ import annotations

FONT = 'font-family="IBM Plex Mono, ui-monospace, monospace"'


def nice_ticks(lo, hi, target=5):
    """Ticks on a round step, so no gridline reads as a rounded-off zero."""
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span/max(1, target)
    magnitude = 10**int(__import__("math").floor(__import__("math").log10(raw)))
    for multiple in (1, 2, 2.5, 5, 10):
        step = multiple*magnitude
        if span/step <= target + 1:
            break
    start = step*__import__("math").ceil(lo/step)
    ticks, value = [], start
    while value <= hi + step*1e-9:
        ticks.append(round(value, 10))
        value += step
    return ticks


def _fmt(value, step):
    decimals = max(0, -int(__import__("math").floor(__import__("math").log10(step))))
    text = f"{value:.{decimals}f}"
    return "0" if text.lstrip("-").strip("0.") == "" else text


def _scale(value, lo, hi, out_lo, out_hi):
    if hi == lo:
        return (out_lo + out_hi)/2
    return out_lo + (value - lo)*(out_hi - out_lo)/(hi - lo)


def slope_chart(series, *, ylim, title, ylabel, width=560, height=None,
                labels=("awake", "ket/xyl"), zero_line=False):
    """series: list of (name, colour_var, y_left, y_right).

    Series are identified in a right-hand key rather than by labels pinned to
    the line ends, which collide wherever series converge.
    """
    height = height or max(300, 96 + len(series)*22)
    left, right, top, bottom = 62, width-214, 34, height-42
    lo, hi = ylim
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="{title}" style="width:100%;height:auto">']
    parts.append(f'<text x="8" y="18" {FONT} font-size="11" '
                 f'fill="var(--ink-soft)">{title}</text>')
    for fraction in (0, .25, .5, .75, 1):
        value = lo + fraction*(hi-lo)
        y = _scale(value, lo, hi, bottom, top)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
                     f'stroke="var(--rule)" stroke-width="1"/>')
        parts.append(f'<text x="{left-7}" y="{y+3.5:.1f}" {FONT} font-size="9.5" '
                     f'text-anchor="end" fill="var(--ink-faint)">{value:+.2f}</text>')
    if zero_line:
        y = _scale(0, lo, hi, bottom, top)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
                     f'stroke="var(--ink-faint)" stroke-width="1.4" '
                     f'stroke-dasharray="4 3"/>')
    for index, text in enumerate(labels):
        x = left if index == 0 else right
        anchor = "start" if index == 0 else "end"
        parts.append(f'<text x="{x}" y="{bottom+20}" {FONT} font-size="10" '
                     f'text-anchor="{anchor}" fill="var(--ink-soft)">{text}</text>')
    for name, colour, a, b in series:
        ya, yb = _scale(a, lo, hi, bottom, top), _scale(b, lo, hi, bottom, top)
        parts.append(f'<line x1="{left}" y1="{ya:.1f}" x2="{right}" y2="{yb:.1f}" '
                     f'stroke="{colour}" stroke-width="2.2" opacity=".92"/>')
        parts.append(f'<circle cx="{left}" cy="{ya:.1f}" r="4.5" fill="{colour}"/>')
        parts.append(f'<circle cx="{right}" cy="{yb:.1f}" r="4.5" fill="{colour}"/>')
    key_x = right + 20
    for index, (name, colour, a, b) in enumerate(series):
        y = top + 6 + index*20
        parts.append(f'<rect x="{key_x}" y="{y-7}" width="9" height="9" rx="1.5" '
                     f'fill="{colour}"/>')
        parts.append(f'<text x="{key_x+14}" y="{y+1}" {FONT} font-size="8.8" '
                     f'fill="var(--ink)">{name}</text>')
        parts.append(f'<text x="{key_x+14}" y="{y+11}" {FONT} font-size="8.2" '
                     f'fill="var(--ink-faint)">{a:+.2f} &#8594; {b:+.2f}</text>')
    parts.append(f'<text x="8" y="{height-8}" {FONT} font-size="9.5" '
                 f'fill="var(--ink-faint)">{ylabel}</text>')
    parts.append("</svg>")
    return "".join(parts)


def grouped_bars(rows, *, xlim, title, xlabel, chance=None, width=560,
                 bar=13, gap=9, pad_top=40):
    """rows: list of (name, colour_var, value_a, value_b)."""
    left = 168
    right = width-58
    height = pad_top + len(rows)*(2*bar+gap) + 34
    lo, hi = xlim
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="{title}" style="width:100%;height:auto">']
    parts.append(f'<text x="8" y="18" {FONT} font-size="11" '
                 f'fill="var(--ink-soft)">{title}</text>')
    for fraction in (0, .25, .5, .75, 1):
        value = lo + fraction*(hi-lo)
        x = _scale(value, lo, hi, left, right)
        parts.append(f'<line x1="{x:.1f}" y1="{pad_top-8}" x2="{x:.1f}" '
                     f'y2="{height-30}" stroke="var(--rule)" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-16}" {FONT} font-size="9.5" '
                     f'text-anchor="middle" fill="var(--ink-faint)">{value:.0f}</text>')
    if chance is not None:
        x = _scale(chance, lo, hi, left, right)
        parts.append(f'<line x1="{x:.1f}" y1="{pad_top-8}" x2="{x:.1f}" '
                     f'y2="{height-30}" stroke="var(--ink-faint)" '
                     f'stroke-width="1.6" stroke-dasharray="4 3"/>')
        parts.append(f'<text x="{x+5:.1f}" y="{pad_top-12}" {FONT} font-size="9" '
                     f'fill="var(--ink-faint)">chance</text>')
    y = pad_top
    for name, colour, a, b in rows:
        parts.append(f'<text x="{left-10}" y="{y+bar-1}" {FONT} font-size="9.5" '
                     f'text-anchor="end" fill="var(--ink)">{name}</text>')
        for value, height_bar, opacity in ((a, bar-2, .95), (b, bar-2, .42)):
            x = _scale(value, lo, hi, left, right)
            parts.append(f'<rect x="{left}" y="{y}" width="{max(0, x-left):.1f}" '
                         f'height="{height_bar}" fill="{colour}" opacity="{opacity}" rx="1"/>')
            parts.append(f'<text x="{x+6:.1f}" y="{y+height_bar-1.5}" {FONT} '
                         f'font-size="9" fill="var(--ink-faint)">{value:.0f}</text>')
            y += bar
        y += gap
    parts.append(f'<text x="{left}" y="{height-2}" {FONT} font-size="9.5" '
                 f'fill="var(--ink-faint)">{xlabel}</text>')
    parts.append("</svg>")
    return "".join(parts)


def line_chart(series, x, *, ylim, title, ylabel, xlabel, width=540, height=326,
               shade=None, zero_line=True, legend=True):
    """series: list of (name, colour_var, values)."""
    left, right, top, bottom = 56, width-128, 52, height-42
    lo, hi = ylim
    xlo, xhi = min(x), max(x)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="{title}" style="width:100%;height:auto">']
    parts.append(f'<text x="8" y="18" {FONT} font-size="11" '
                 f'fill="var(--ink-soft)">{title}</text>')
    if shade:
        x0 = _scale(shade[0], xlo, xhi, left, right)
        x1 = _scale(shade[1], xlo, xhi, left, right)
        parts.append(f'<rect x="{x0:.1f}" y="{top}" width="{x1-x0:.1f}" '
                     f'height="{bottom-top}" fill="var(--rule)" opacity=".55"/>')
    ticks = nice_ticks(lo, hi)
    step = (ticks[1]-ticks[0]) if len(ticks) > 1 else (hi-lo)
    for value in ticks:
        y = _scale(value, lo, hi, bottom, top)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
                     f'stroke="var(--rule)" stroke-width="1"/>')
        parts.append(f'<text x="{left-7}" y="{y+3.5:.1f}" {FONT} font-size="9.5" '
                     f'text-anchor="end" fill="var(--ink-faint)">{_fmt(value, step)}</text>')
    if zero_line and lo < 0 < hi:
        y = _scale(0, lo, hi, bottom, top)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
                     f'stroke="var(--ink-faint)" stroke-width="1.3"/>')
    for value in x:
        if value % 2 == 0:
            px = _scale(value, xlo, xhi, left, right)
            parts.append(f'<text x="{px:.1f}" y="{bottom+18}" {FONT} font-size="9.5" '
                         f'text-anchor="middle" fill="var(--ink-faint)">{value:g}</text>')
    for offset, (name, colour, values) in enumerate(series):
        points = " ".join(
            f"{_scale(xv, xlo, xhi, left, right):.1f},{_scale(yv, lo, hi, bottom, top):.1f}"
            for xv, yv in zip(x, values) if yv is not None)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colour}" '
                     f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        if legend:
            ly = top + 4 + offset*15
            parts.append(f'<line x1="{right+10}" y1="{ly}" x2="{right+26}" y2="{ly}" '
                         f'stroke="{colour}" stroke-width="2.4"/>')
            parts.append(f'<text x="{right+30}" y="{ly+3.5}" {FONT} font-size="8.6" '
                         f'fill="var(--ink-soft)">{name}</text>')
    parts.append(f'<text x="{(left+right)/2:.0f}" y="{height-6}" {FONT} font-size="9.5" '
                 f'text-anchor="middle" fill="var(--ink-faint)">{xlabel}</text>')
    parts.append(f'<text x="8" y="32" {FONT} font-size="9.5" '
                 f'fill="var(--ink-faint)">{ylabel}</text>')
    parts.append("</svg>")
    return "".join(parts)
