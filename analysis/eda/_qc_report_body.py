import json, pathlib
SP = pathlib.Path(__file__).parent
IMG = json.loads((SP/"qc_figures.json").read_text())
CSS = open(SP/"odyn_eda_report.html").read()
CSS = CSS[CSS.index("<style>"):CSS.index("</style>")+8]


def figure(name, caption, alt):
    return f'''<figure class="fig">
  <img class="only-light" src="data:image/png;base64,{IMG[name+'_light']}" alt="{alt}">
  <img class="only-dark" src="data:image/png;base64,{IMG[name+'_dark']}" alt="{alt}">
  <figcaption>{caption}</figcaption>
</figure>'''


HTML = f"""<title>ODyn Acquisition QC</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
{CSS}

<div class="wrap">

<header class="masthead">
  <span class="eyebrow">Acquisition QC &middot; ket/xyl 16-odor dataset</span>
  <h1>Reagents, rig, and session quality</h1>
  <p class="lede">Findings about the preparation rather than the biology: how the mineral-oil blank behaves, which sessions look anomalous, and what is worth recording going forward. Separated from the analysis write-up so neither obscures the other.</p>
  <div class="runmeta">
    <span><b>44</b> usable sessions of 48</span>
    <span>odors made <b>7/6</b>, <b>7/22</b>, <b>8/6</b></span>
    <span>session length <b>224</b> to <b>160</b> at the 7/22 make</span>
    <span>run <b>2026-08-30</b></span>
  </div>
</header>

<section id="summary">
  <div class="sechead"><span class="secnum">SUM</span><h2>What to act on</h2></div>
  <div class="board">
    <div class="finding">
      <span class="chip warn">Reagent</span>
      <h3>The mineral-oil blank got cleaner with every remake</h3>
      <p>The blank response falls 35-fold across successive odor batches, and every within-mouse batch transition in the dataset declines. It behaves like a contamination readout, which makes it unsuitable as any kind of analysis reference and worth tracking as a reagent-quality number.</p>
    </div>
    <div class="finding">
      <span class="chip stop">Inspect</span>
      <h3>2026-07-17 looks like a bad delivery day</h3>
      <p>Both TH sessions that day, in different animals, have every ROI responding to mineral oil, and in one the blank out-drives the real odors. Worth pulling the raw traces and checking the vial and valve logs before those sessions are used.</p>
    </div>
    <div class="finding">
      <span class="chip warn">Rig</span>
      <h3>Awake pupil is censored exactly where it matters</h3>
      <p>Coverage is lost mainly to clipping, up to 41% of awake samples, and clipping happens when the pupil is large. Only 18 of 37 awake sessions clear a 50% coverage bar. The ROI or fit bounds want widening before pupil carries weight in a figure.</p>
    </div>
    <div class="finding">
      <span class="chip good">Note</span>
      <h3>Session length changed once, at the 7/22 odor make</h3>
      <p>Sessions ran 224 trials through 7/21 and 160 from 7/23 &mdash; 7 repeats per odor per state down to 5 &mdash; with a single 224-trial session on 8/21. Because that step coincides with an odor make, batch and session length cannot be separated across it.</p>
    </div>
  </div>
</section>

<section id="batch">
  <div class="sechead"><span class="secnum">01</span><h2>The mineral-oil batch effect</h2></div>
  <div class="stack">
    <p class="prose">Odors were made on 7/6, 7/22 and 8/6. Ordering sessions by batch rather than by calendar date produces a clean monotone decline in how much the blank drives the population.</p>
    <div class="tablewrap">
      <table>
        <caption>10&times; awake, median across sessions. Sessions before 7/6 ran on older stock of unknown age.</caption>
        <thead><tr><th>Odor batch</th><th>Sessions</th><th>Mineral oil (z)</th><th>Responder fraction</th></tr></thead>
        <tbody>
          <tr><td>before 7/6 &nbsp;<span class="muted">older stock</span></td><td class="num">2</td><td class="num hi-e">1.23</td><td class="num">0.98</td></tr>
          <tr><td>made 7/6</td><td class="num">8</td><td class="num">0.40</td><td class="num">0.77</td></tr>
          <tr><td>made 7/22</td><td class="num">6</td><td class="num">0.22</td><td class="num">0.63</td></tr>
          <tr><td>made 8/6</td><td class="num">7</td><td class="num hi-s">0.035</td><td class="num">0.45</td></tr>
        </tbody>
      </table>
    </div>
    {figure("batch",
      "Left: every within-mouse batch transition present in the data, with line and animal held fixed. All four decline. "
      "Right: mineral-oil response against days since the odors were made, with the Thy1 fit shown.",
      "Mineral oil response by odor batch, within mouse, and against days since the odors were made")}
    <div class="tablewrap">
      <table>
        <caption>Within-mouse transitions &mdash; the batch comparison with animal held fixed.</caption>
        <thead><tr><th>Line</th><th>Mouse</th><th>Transition</th><th>Mineral oil before</th><th>after</th></tr></thead>
        <tbody>
          <tr><td>DAT</td><td>m472</td><td>older stock &rarr; 7/6</td><td class="num">1.23</td><td class="num">0.41</td></tr>
          <tr><td>DAT</td><td>m496</td><td>7/22 &rarr; 8/6</td><td class="num">0.26</td><td class="num">0.035</td></tr>
          <tr><td>Thy1</td><td>m392</td><td>7/22 &rarr; 8/6</td><td class="num">0.33</td><td class="num">0.09</td></tr>
          <tr><td>Thy1</td><td>m426</td><td>7/22 &rarr; 8/6</td><td class="num">0.11</td><td class="num">0.007</td></tr>
        </tbody>
      </table>
    </div>
    <ul class="clean">
      <li><strong>All four transitions decline</strong> (sign test p = 0.062, the floor for n = 4). Median change &minus;0.229 z for the blank against only &minus;0.123 z for the real odors, so it is disproportionately the blank rather than a general drop in signal.</li>
      <li><strong>Thy1 is the clean test</strong> &mdash; the only line spanning a remake with the same two animals throughout. Days since make against blank response gives &rho; = <strong>+0.73, p = 0.039</strong>; the batch medians are 0.22 then 0.034. Over the same change its odor response moved only &minus;20% against the blank's &minus;85%.</li>
      <li><strong>TH runs the other way</strong> (7/6 batch 0.39, 8/6 batch 0.67), but TH cannot test this: it has no sessions on the 7/22 batch and every August TH session is a different animal from every July one.</li>
      <li><strong>It is batch order, not simple ageing.</strong> Days-since-make within a batch is weak across all lines (&rho; = +0.20, p = 0.39). The effect is mostly a step at each remake, with Thy1 the one line also showing the within-batch gradient.</li>
    </ul>
  </div>
</section>

<section id="blank">
  <div class="sechead"><span class="secnum">02</span><h2>How the blank behaves, by scale</h2></div>
  <div class="stack">
    <p class="prose">The blank response is a 10&times; phenomenon. At 20&times; the median cell shows essentially nothing, which is why it is not visible in the QC images.</p>
    {figure("trace",
      "Population median response to mineral oil against real odors. The pre-odor baseline is flat, so this is not drift. "
      "Awake, the blank drives TH glomeruli to roughly two thirds of the real-odor response; at 20x it sits near zero.",
      "Population median z traces for blank and real odors, awake and anesthetized")}
    <div class="tablewrap">
      <table>
        <caption>Sustained-window (1&ndash;4 s) median response in z, awake. Responder fraction is a separate quantity from magnitude.</caption>
        <thead><tr><th>Population</th><th>Blank (z)</th><th>Real odors (z)</th><th>Blank as % of odor</th><th>Blank responder fraction</th></tr></thead>
        <tbody>
          <tr><td>TH 10&times; units</td><td class="num hi-e">0.62</td><td class="num">1.01</td><td class="num">61%</td><td class="num">0.85</td></tr>
          <tr><td>DAT 10&times; units</td><td class="num">0.33</td><td class="num">0.69</td><td class="num">47%</td><td class="num">0.60</td></tr>
          <tr><td>Thy1 10&times; units</td><td class="num">0.10</td><td class="num">0.61</td><td class="num">17%</td><td class="num">0.39</td></tr>
          <tr><td>20&times; somas</td><td class="num hi-s">0.005</td><td class="num">0.065</td><td class="num muted">&mdash;</td><td class="num">0.40</td></tr>
          <tr><td>20&times; processes</td><td class="num hi-s">&minus;0.04</td><td class="num">&minus;0.01</td><td class="num muted">&mdash;</td><td class="num">0.26</td></tr>
        </tbody>
      </table>
    </div>
    <div class="callout">
      <span class="eyebrow">Fraction is not magnitude</span>
      <p>A responder call asks whether an excursion exceeds that ROI's own pre-odor excursions. A small but consistent deflection passes easily on a median over five trials, which is why 40% of 20&times; ROIs register a blank response whose typical size is 0.005 z. The detection procedure itself is calibrated: run on two windows inside the pre-odor period it returns 0.035&ndash;0.078 against a nominal 0.05.</p>
    </div>
  </div>
</section>

<section id="outliers">
  <div class="sechead"><span class="secnum">03</span><h2>Sessions worth inspecting</h2></div>
  <div class="stack">
    <div class="tablewrap">
      <table>
        <caption>Sessions where mineral oil out-drove the real odors, or where every ROI responded to it.</caption>
        <thead><tr><th>Date</th><th>Group</th><th>Line / mouse</th><th>Mineral oil (z)</th><th>Odor (z)</th><th>MO / odor</th><th>Responders</th></tr></thead>
        <tbody>
          <tr><td>2026-07-17</td><td class="num">231</td><td>TH m462</td><td class="num hi-e">2.19</td><td class="num">1.36</td><td class="num hi-e">1.61</td><td class="num hi-e">1.00</td></tr>
          <tr><td>2026-07-17</td><td class="num">232</td><td>TH m465</td><td class="num hi-e">1.25</td><td class="num">2.27</td><td class="num">0.55</td><td class="num hi-e">1.00</td></tr>
          <tr><td>2026-08-03</td><td class="num">215</td><td>DAT m496</td><td class="num">0.44</td><td class="num">0.24</td><td class="num hi-e">1.80</td><td class="num">0.69</td></tr>
          <tr><td>2026-06-26</td><td class="num">193</td><td>DAT m472</td><td class="num">1.27</td><td class="num">0.82</td><td class="num hi-e">1.55</td><td class="num">0.99</td></tr>
        </tbody>
      </table>
    </div>
    <p class="prose"><strong>2026-07-17 is the clearest flag.</strong> Two different animals, same day, both with every ROI responding to the blank. That points at the vial or the valve rather than the animals. It is not explained by batch: those sessions sit 11 days into the 7/6 batch, and other 7/6-batch sessions at 7&ndash;11 days run 0.15&ndash;0.72. The DAT session that day cannot corroborate either way &mdash; it is the awake-only 20-trial session with a single blank trial.</p>
  </div>
</section>

<section id="pupil">
  <div class="sechead"><span class="secnum">04</span><h2>Pupil tracking</h2></div>
  <div class="stack">
    {figure("pupil",
      "Each point is one session. Awake coverage is far worse than anesthetized, and the loss is driven by clipping rather than blinks.",
      "Pupil coverage and clipped fraction per session, awake versus anesthetized")}
    <ul class="clean">
      <li><strong>The loss is clipping, not blinks.</strong> Up to 41% of awake samples are clipped against 0&ndash;18% blinks, and awake pupils are systematically larger (109&ndash;150 px against 80&ndash;128 anesthetized). The tracker is losing the dilations, which is the worst possible place to lose data for an arousal analysis.</li>
      <li><strong>Only 18 of 37 awake sessions clear 50% coverage</strong> after quality masking; 14 clear 70%. The unmasked fit has 97&ndash;99% coverage everywhere except TH 20&times; superficial, which is a genuine tracking failure at 48&ndash;58%.</li>
      <li><strong>Coverage is bimodal</strong> &mdash; sessions are either near 1.00 or near 0.00, with only one in between. That pattern suggests a per-session setup difference (ROI placement, illumination) rather than gradual quality drift, so it should be fixable at acquisition.</li>
    </ul>
  </div>
</section>

<section id="protocol">
  <div class="sechead"><span class="secnum">05</span><h2>Session length and inventory</h2></div>
  <div class="stack">
    {figure("timeline",
      "Mineral-oil response with the three odor-make dates marked, above trials acquired per session. The session-length step "
      "falls between 7/21 and 7/23, bracketing the 7/22 make.",
      "Mineral oil response and trials acquired per session over the study, with odor make dates marked")}
    <ul class="clean">
      <li><strong>Session length steps once.</strong> Every session through 2026-07-21 acquired 224 trials; every session from 2026-07-23 acquired 160. That is 7 repeats per odor per state down to 5. The protocol itself is unchanged throughout &mdash; this is how many repeats were run.</li>
      <li><strong>The step coincides with the 7/22 odor make</strong>, so batch and session length change together and cannot be separated across that boundary. The one dissociation in the dataset is <strong>group 244 on 8/21</strong>: 224 trials on the 8/6 batch. A single session, but it is the only leverage available.</li>
      <li><strong>Fewer trials did not cost discriminability.</strong> The later, shorter sessions classify odors better, not worse, so the session-length change does not explain the acquisition-date trend and is not itself a concern for the analysis.</li>
      <li><strong>Four sessions have no grouped product</strong>: groups 176, 191, 198 and 234. Two of them cost TH 20&times; superficial an entire mouse. Group 230 is awake-only with 20 trials.</li>
      <li><strong>Repeats per odor vary from 1 to 11 within a single session</strong>, which is what forces trial-count stratification in every downstream threshold.</li>
    </ul>
  </div>
</section>

<section id="record">
  <div class="sechead"><span class="secnum">06</span><h2>Worth recording per session</h2></div>
  <div class="stack">
    <ul class="clean">
      <li><strong>Odor batch date</strong>, so batch can be a covariate rather than something reconstructed from a calendar afterwards.</li>
      <li><strong>Blank responder fraction and median blank response.</strong> It is the most session-variable number measured here (0.34 to 1.00 awake at 10&times;) and it tracks reagent state.</li>
      <li><strong>Pupil coverage before and after quality masking</strong>, with the clipped fraction separated from blinks &mdash; the two have different fixes.</li>
      <li><strong>Repeats per odor</strong>, not just total trials. More mineral-oil repeats would help most: at 4&ndash;8 blank trials any blank-referenced correction is noise-limited, and the penalty falls as one over the square root of that count.</li>
    </ul>
  </div>
</section>

<footer>
  <p>Acquisition QC for the ODyn ket/xyl 16-odor dataset &middot; 44 sessions &middot; 2026-08-30. Companion to the exploratory-pass write-up, which covers the analysis findings.</p>
</footer>

</div>
"""
(SP/"odyn_qc_report.html").write_text(HTML)
print("QC report:", len(HTML), "chars")
