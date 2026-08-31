"""Assemble the exploratory-pass report page."""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from .svg_helpers import grouped_bars, line_chart, slope_chart

SP = pathlib.Path(__file__).parent
IMG = json.loads((SP/"out/report_figures.json").read_text())

S = {"a": "var(--s1)", "b": "var(--s2)", "c": "var(--s3)",
     "d": "var(--s4)", "e": "var(--s5)", "f": "var(--s6)"}


def figure(name, caption, alt):
    """A themed raster figure, one image per theme."""
    return f'''<figure class="fig">
  <img class="only-light" src="data:image/png;base64,{IMG[name+'_light']}" alt="{alt}">
  <img class="only-dark" src="data:image/png;base64,{IMG[name+'_dark']}" alt="{alt}">
  <figcaption>{caption}</figcaption>
</figure>'''


def svg_figure(svg, caption):
    return f'<figure class="fig">{svg}<figcaption>{caption}</figcaption></figure>'


# ---------------------------------------------------------------- SVG charts
SUPPRESSION = svg_figure(slope_chart([
    ("TH 20x deep somas", S["a"], .483, .025),
    ("TH 20x sup somas", S["b"], .467, .067),
    ("DAT 20x sup somas", S["c"], .365, .067),
    ("DAT 10x units", S["d"], .274, .031),
    ("TH 10x units", S["e"], .183, .000),
    ("Thy1 10x units", S["f"], .084, .000),
], ylim=(0, .55), title="Suppression breadth collapses in every population",
   ylabel="fraction of odors suppressing the unit — mice as sampling units"),
   "Suppression breadth, awake against ket/xyl. Every cohort and both "
   "compartments move the same way, and this is a within-session, "
   "within-field contrast.")

BALANCE = svg_figure(slope_chart([
    ("DAT 20x superficial", S["c"], .673, .579),
    ("TH 20x superficial", S["b"], .339, .470),
    ("TH 20x deep", S["a"], -.410, .703),
], ylim=(-.6, .9), zero_line=True,
   title="E/S balance: a depth gradient awake, erased by anesthesia",
   ylabel="+1 all excitation · 0 balanced · -1 all suppression"),
   "Somatic E/S balance. Awake, the three cohorts order monotonically and TH "
   "deep is the only suppression-dominated population. Under ket/xyl all three "
   "converge on excitation, and TH deep moves furthest.")

ACCURACY = svg_figure(grouped_bars([
    ("Thy1 10x", S["f"], 93.2, 82.8),
    ("DAT 10x", S["d"], 89.4, 70.9),
    ("TH 10x", S["e"], 78.1, 73.0),
    ("DAT 20x sup somas", S["c"], 58.8, 33.5),
    ("TH 20x sup somas", S["b"], 32.1, 31.1),
    ("TH 20x deep somas", S["a"], 27.3, 24.1),
], xlim=(0, 100), chance=6.25,
   title="Sixteen-way odor classification accuracy",
   xlabel="% of trials classified correctly — solid awake, faded ket/xyl"),
   "Leave-one-trial-out nearest centroid across the whole panel, median across "
   "sessions. Every population is far above chance, including TH deep at four "
   "times chance; the 10x/20x gap is the largest structure here.")

RATIO = svg_figure(line_chart([
    ("Thy1 10x", S["f"], [1.84, 1.45, 2.73, 3.16, 1.97, 1.02, .81, .49]),
    ("DAT 10x", S["d"], [.58, .75, 1.11, 1.66, .81, 1.09, .35, .29]),
    ("TH 20x deep somas", S["a"], [.03, .49, .84, .87, 1.64, .36, .19, .24]),
    ("TH 10x", S["e"], [.12, .12, .29, .64, .01, -.05, -.04, -.06]),
    ("TH 20x sup somas", S["b"], [.79, .27, .25, .22, .18, .06, .16, .24]),
], list(range(8)), ylim=(-.5, 3.5), shade=(0, 4),
   title="Ratio separation for pair 39/40 builds late, awake",
   ylabel="crossvalidated distance", xlabel="window start, seconds from odor onset"),
   "The one reciprocal pair that separates above chance. Odor is on for the "
   "shaded 0-4 s. Separation peaks in the final second, and for TH deep somata "
   "after offset entirely, so a 0-4 s integral dilutes it.")

TIME = list(range(-5, 9))
RUNNING = svg_figure(line_chart([
    ("TH 20x sup somas", S["b"], [.01, 0, .03, -.01, -.03, -.02, .21, .31, .27, .27, .26, .23, .18, .13]),
    ("Thy1 10x", S["f"], [0, .04, .03, 0, -.01, .12, .18, .24, .23, .11, .07, .08, .07, .07]),
    ("TH 20x deep somas", S["a"], [.03, -.02, .03, -.02, 0, .07, .17, .23, .17, .12, .21, .13, .14, .08]),
    ("TH 10x", S["e"], [.02, 0, -.02, -.03, .04, .07, .20, .22, .17, .16, .03, .08, .09, .11]),
    ("DAT 10x", S["d"], [.02, -.04, .02, -.04, .02, 0, .13, .22, .21, .16, .07, .12, .09, .07]),
    ("DAT 20x sup somas", S["c"], [-.01, -.01, -.01, 0, .03, -.02, .09, .17, .13, .09, .01, .05, .08, .06]),
], TIME, ylim=(-.1, .35), shade=(0, 4),
   title="Running coupling is flat before the odor and peaks at 2 s",
   ylabel="median r", xlabel="time from odor onset (s)"),
   "Correlation between trial-to-trial running deviation and response "
   "deviation, odor identity removed. Zero across the whole pre-odor period in "
   "every cohort, which is why this is not motion artifact.")

PUPIL = svg_figure(line_chart([
    ("TH 20x sup somas", S["b"], [.01, .02, -.01, -.03, 0, .11, .21, .23, .24, .17, .15, .05, .02, .05]),
    ("DAT 10x", S["d"], [0, -.01, -.02, .01, .04, 0, .08, .15, .12, .08, .07, .01, 0, -.03]),
    ("TH 10x", S["e"], [.04, .06, -.03, -.12, .04, .10, .15, .11, .01, .05, .11, .04, .02, 0]),
    ("Thy1 10x", S["f"], [.03, -.01, -.01, -.02, .02, 0, .04, .07, .03, .02, .02, 0, -.01, -.03]),
    ("TH 20x deep somas", S["a"], [.05, -.03, -.06, .01, .06, .04, .05, .05, .03, .05, .03, .08, .07, .03]),
], TIME, ylim=(-.15, .35), shade=(0, 4),
   title="Pupil dilation: same shape, about three times weaker",
   ylabel="median r", xlabel="time from odor onset (s)"),
   "Pupil coupling, awake, unmasked trace. The same onset-locked shape as "
   "running but smaller, and absent under ket/xyl where locomotion is zero and "
   "pupil quality is best.")

BLANK = figure("blank",
    "Population median response to mineral oil against real odors. The "
    "pre-odor baseline is flat, so this is not drift. Awake, the blank drives "
    "TH glomeruli to roughly two thirds of the real-odor response; at 20x it "
    "sits near zero. Shaded band is the 4 s odor.",
    "Population median z traces for blank and real odors, awake and anesthetized")

BIPHASIC = figure("biphasic",
    "One real unit-odor pair, TH 10x. Within the 4 s odor the response goes "
    "+1.6, then -4.7, then +2.9 — and its signed four-second mean is -0.06 z. "
    "The shaded areas are the two components measured separately.",
    "A biphasic response whose signed four-second mean is near zero")

CONFUSION = figure("confusion",
    "Leave-one-trial-out confusion across all 16 odors, awake, odors ordered "
    "so each mixture sits beside its two components. In Thy1 the only "
    "off-diagonal mass is alpha against alpha-prime and epsilon against "
    "epsilon-prime; lambda/lambda-prime is fully resolved. Mixtures are "
    "confused with their reciprocal partner, never with their components.",
    "Confusion matrices for six populations across sixteen odors")

TRIALS = figure("trials",
    "Every awake trial against every other, one Thy1 session, sorted by odor. "
    "The three mixture families behave differently: alpha and alpha-prime form "
    "one block unlike either component; epsilon, epsilon-prime and both "
    "components form a single block; lambda and lambda-prime separate from "
    "each other and sit nearer acetophenone.",
    "Trial-by-trial population pattern correlation matrix for one Thy1 session")

pathlib.Path(SP/"report_parts.json").write_text(json.dumps({
    "suppression": SUPPRESSION, "balance": BALANCE, "accuracy": ACCURACY,
    "ratio": RATIO, "running": RUNNING, "pupil": PUPIL, "blank": BLANK,
    "biphasic": BIPHASIC, "confusion": CONFUSION, "trials": TRIALS,
}))
print("built", 10, "figure blocks")
