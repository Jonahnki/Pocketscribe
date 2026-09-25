#!/usr/bin/env python3
"""Build the static Pocketscribe site.

Generates `site/index.html` plus the live example reports in `site/demo/`, then
Vercel serves the directory as-is. There is no build step on Vercel's side and no
runtime: fpocket and a scientific Python stack do not belong in a serverless
function, and more importantly this project's whole claim is that structures are
analysed locally and never uploaded anywhere. A static site is the only kind that
does not contradict the README.

The figures on the landing page are extracted from a real generated report rather
than drawn by hand, so the page cannot drift away from what the tool actually
produces.

Usage:
    python site/build.py

The generated files are committed, because Vercel needs them present at deploy time.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

SITE = Path(__file__).parent
ROOT = SITE.parent
DEMO = SITE / "demo"


def interpreter() -> str:
    """The Python that has Pocketscribe installed.

    Prefer the project's own virtualenv over whatever interpreter happened to launch
    this script: running `python site/build.py` from an activated conda base is the
    normal case, and that interpreter does not have the dependencies.
    """
    venv = ROOT / ".venv" / "bin" / "python"
    if venv.is_file():
        return str(venv)
    return sys.executable


def generate_demo_reports() -> None:
    """Run the CLI to produce the example reports the site links to."""
    DEMO.mkdir(parents=True, exist_ok=True)
    python = interpreter()
    print(f"  using {python}")

    for args, name in (
        ([], "report.html"),
        (["--consensus"], "consensus.html"),
    ):
        print(f"  generating demo/{name}")
        result = subprocess.run(
            [python, "-m", "pocketscribe.cli", "demo", *args,
             "--output", str(DEMO / name)],
            cwd=ROOT, capture_output=True, text=True,
        )
        if result.returncode != 0:
            hint = ""
            if "ModuleNotFoundError" in result.stderr:
                hint = (
                    "\n\nHint: Pocketscribe is not installed for this interpreter. "
                    "Create the project venv and install it:\n"
                    "    python3 -m venv .venv && ./.venv/bin/pip install -e '.[dev]'"
                )
            raise SystemExit(
                f"failed to generate {name}:\n{result.stdout}\n{result.stderr}{hint}"
            )


def extract_figure(html: str, pattern: str) -> str:
    """Pull one inline SVG out of a generated report."""
    match = re.search(pattern, html, re.S)
    if not match:
        raise SystemExit(f"could not find a figure matching {pattern!r} in the report")
    return match.group(0)


# --------------------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------------------

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pocketscribe — binding-pocket triage for predicted protein structures</title>
<meta name="description" content="Open-source tool that takes a predicted protein structure from AlphaFold2, OpenFold, ColabFold, ESMFold or Boltz and produces a confidence-caveated binding-pocket report with ready-to-run GROMACS inputs. Runs locally; nothing is uploaded. Research use only.">
<meta property="og:title" content="Pocketscribe">
<meta property="og:description" content="Confidence-aware binding-pocket triage and MD setup for predicted protein structures.">
<meta property="og:type" content="website">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><text y='26' font-size='26'>&#129516;</text></svg>">
<style>
:root {
  --page: #f9f9f7; --surface: #ffffff; --sunken: #f4f3ef;
  --ink: #17170f; --ink-2: #52514e; --ink-3: #898781;
  --rule: #e1e0d9; --rule-2: #c3c2b7; --accent: #0b4fbf;
  --figure: #fcfcfb;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
  --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
  --serif: ui-serif, Georgia, "Iowan Old Style", "Times New Roman", serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --page: #0d0d0d; --surface: #1a1a19; --sunken: #222220;
    --ink: #f4f3ef; --ink-2: #c3c2b7; --ink-3: #898781;
    --rule: #2c2c2a; --rule-2: #383835; --accent: #6ca4f0;
  }
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font-family: var(--sans); font-size: 16px; line-height: 1.65;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 940px; margin: 0 auto; padding: 0 16px; }

/* ---------- nav ---------- */
nav {
  position: sticky; top: 0; z-index: 10;
  background: color-mix(in srgb, var(--page) 88%, transparent);
  backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--rule);
}
nav .wrap { display: flex; align-items: center; gap: 22px; height: 54px; }
.brand {
  font-family: var(--serif); font-weight: 600; font-size: 16px;
  letter-spacing: .01em; margin-right: auto; text-decoration: none; color: var(--ink);
}
nav a { color: var(--ink-2); text-decoration: none; font-size: 14px; }
nav a:hover { color: var(--accent); }
@media (max-width: 620px) { .nav-hide { display: none; } }

/* ---------- hero ---------- */
header.hero { padding: 72px 0 8px; }
h1 {
  font-family: var(--serif); font-weight: 600;
  font-size: clamp(32px, 6vw, 50px); line-height: 1.12;
  margin: 0 0 18px; letter-spacing: -.015em; max-width: 19ch;
}
.lede { font-size: clamp(17px, 2.2vw, 19px); color: var(--ink-2); max-width: 62ch; margin: 0 0 26px; }
.cta { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 22px; }
.btn {
  display: inline-block; padding: 10px 18px; border-radius: 7px;
  font-size: 14.5px; font-weight: 550; text-decoration: none;
  border: 1px solid var(--rule-2); color: var(--ink); background: var(--surface);
}
.btn:hover { border-color: var(--accent); color: var(--accent); }
.btn-primary { background: var(--accent); border-color: var(--accent); color: #fff; }
.btn-primary:hover { opacity: .9; color: #fff; }
.chip {
  display: inline-block; font-size: 12px; letter-spacing: .05em;
  text-transform: uppercase; color: var(--ink-3);
  border: 1px solid var(--rule); border-radius: 100px; padding: 3px 11px;
}

/* ---------- sections ---------- */
section { padding: 52px 0; border-top: 1px solid var(--rule); scroll-margin-top: 62px; }
section:first-of-type { border-top: none; }
h2 {
  font-family: var(--serif); font-weight: 600;
  font-size: clamp(22px, 3.4vw, 28px); margin: 0 0 6px; letter-spacing: -.01em;
}
.sub { color: var(--ink-2); margin: 0 0 26px; max-width: 64ch; }
h3 { font-size: 15.5px; font-weight: 650; margin: 0 0 6px; }
p { margin: 0 0 15px; }

.grid { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
.card {
  background: var(--surface); border: 1px solid var(--rule);
  border-radius: 9px; padding: 17px 19px;
}
.card p { font-size: 14.5px; color: var(--ink-2); margin: 0; }

figure { margin: 0 0 18px; background: var(--figure); border: 1px solid var(--rule); border-radius: 9px; padding: 16px; }
figure svg { display: block; width: 100%; height: auto; }
figcaption { font-size: 13.5px; color: #52514e; margin-top: 12px; }
figcaption strong { color: #17170f; }

table { width: 100%; border-collapse: collapse; font-size: 14px; }
.scroll { overflow-x: auto; }
th, td { text-align: left; padding: 9px 11px; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-size: 11.5px; letter-spacing: .05em; text-transform: uppercase; color: var(--ink-3); border-bottom: 1px solid var(--rule-2); white-space: nowrap; }
tbody tr:last-child td { border-bottom: none; }
.t1 { color: var(--accent); font-weight: 600; }
.t2 { color: var(--ink-3); }

pre {
  background: var(--sunken); border: 1px solid var(--rule); border-radius: 8px;
  padding: 14px 16px; overflow-x: auto; font-family: var(--mono); font-size: 13px;
  line-height: 1.6; margin: 0 0 14px;
}
code { font-family: var(--mono); font-size: .92em; }
p code, li code, td code { background: var(--sunken); padding: 1px 5px; border-radius: 4px; }

ul.plain { margin: 0 0 15px; padding-left: 20px; color: var(--ink-2); font-size: 15px; }
ul.plain li { margin-bottom: 7px; }
a { color: var(--accent); }

.callout {
  border: 1px solid var(--rule-2); border-left: 4px solid var(--accent);
  border-radius: 7px; background: var(--sunken); padding: 16px 18px; font-size: 14.5px;
}
.callout p:last-child { margin-bottom: 0; }

.refs { font-size: 13.5px; color: var(--ink-2); }
.refs li { margin-bottom: 9px; }

footer { border-top: 1px solid var(--rule); padding: 34px 0 60px; font-size: 13.5px; color: var(--ink-3); }
footer p { margin-bottom: 10px; }
footer strong { color: var(--ink-2); }
</style>
</head>
<body>

<nav><div class="wrap">
  <a class="brand" href="#">Pocketscribe</a>
  <a class="nav-hide" href="#what">What it does</a>
  <a class="nav-hide" href="#consensus">Consensus</a>
  <a class="nav-hide" href="#sources">Sources</a>
  <a href="#install">Install</a>
  <a href="https://github.com/Jonahnki/Pocketscribe">GitHub</a>
</div></nav>

<div class="wrap">

<header class="hero">
  <span class="chip">Open source · Research use only</span>
  <h1>Binding-pocket triage for predicted structures</h1>
  <p class="lede">
    Pocketscribe takes a protein structure predicted by AlphaFold2, OpenFold, ColabFold,
    ESMFold or Boltz and produces one report: ranked binding pockets, each caveated by how
    far the prediction actually supports it, plus ready-to-run molecular dynamics inputs.
    It runs on your machine. Nothing is uploaded.
  </p>
  <div class="cta">
    <a class="btn btn-primary" href="demo/report.html">See a real report</a>
    <a class="btn" href="demo/consensus.html">Cross-model consensus example</a>
    <a class="btn" href="https://github.com/Jonahnki/Pocketscribe">GitHub</a>
  </div>
</header>

<section id="problem">
  <h2>The gap it fills</h2>
  <p class="sub">
    A group with a predicted structure and no crystallographic data has to decide whether
    any site on that model justifies a docking campaign, a simulation, or bench time. The
    existing tools each solve one piece and leave you to join them up.
  </p>
  <div class="grid">
    <div class="card">
      <h3>Pocket finders ignore confidence</h3>
      <p>fpocket, DoGSiteScorer and P2Rank were built for experimental structures. Handed
      an AlphaFold model, they score a cavity formed by a disordered loop at pLDDT&nbsp;35
      exactly as confidently as one in a well-resolved core.</p>
    </div>
    <div class="card">
      <h3>Web tools want your structure</h3>
      <p>For an unpublished target, uploading the model to a third party is often simply
      not an option — which rules out most of the convenient routes.</p>
    </div>
    <div class="card">
      <h3>Family profilers don't generalise</h3>
      <p>Tools trained on kinases or GPCRs are sharp inside their family and quiet outside
      it, which is exactly where a novel target sits.</p>
    </div>
    <div class="card">
      <h3>Nothing reaches simulation</h3>
      <p>After a promising pocket, the step to a running simulation still means assembling
      a GROMACS setup by hand from tutorials — a reliable source of quiet errors.</p>
    </div>
  </div>
</section>

<section id="what">
  <h2>Confidence is carried through, not bolted on</h2>
  <p class="sub">
    Every pocket claim is cross-referenced against the prediction's own per-residue
    confidence. The figure below is taken from the live demo report — not a mock-up.
  </p>

  <figure>
    __CONFIDENCE_PLOT__
    <figcaption>
      <strong>Per-residue pLDDT along the sequence.</strong> The trough in the middle is
      a stretch the model is not confident about. A cavity lined by those residues can
      still score respectably on geometry — and the report says so, while flagging that
      its shape and volume are not supported by the prediction. Separating those two
      statements is the whole point of the tool.
    </figcaption>
  </figure>

  <div class="grid">
    <div class="card">
      <h3>Structure QC</h3>
      <p>Per-residue confidence mapped onto the sequence, low-confidence regions flagged,
      chain breaks and numbering gaps reported.</p>
    </div>
    <div class="card">
      <h3>Pocket detection</h3>
      <p>fpocket does the cavity geometry and druggability scoring. Each pocket is then
      annotated with what the prediction confidence permits you to conclude from it.</p>
    </div>
    <div class="card">
      <h3>MD setup</h3>
      <p>A cleaned structure, CHARMM36m parameters with the correct non-bonded settings,
      equilibration and production <code>.mdp</code> files, and a written protocol.</p>
    </div>
    <div class="card">
      <h3>One shareable file</h3>
      <p>A single self-contained HTML report. No scripts, no CDN, no external assets — it
      opens offline and still renders years later.</p>
    </div>
  </div>
</section>

<section id="consensus">
  <h2>Cross-model consensus</h2>
  <p class="sub">
    Supply predictions of the same protein from more than one tool and Pocketscribe matches
    pockets across them — then reports agreement between <em>independent</em> methods
    separately from agreement between close relatives.
  </p>

  <figure>__CONSENSUS_DIAGRAM__
    <figcaption>
      <strong>Not all agreement is equal.</strong> OpenFold is a reimplementation of
      AlphaFold2, so the two share an inference pathway and can reproduce each other's
      errors. ESMFold predicts from a single sequence with no MSA, so its agreement is
      genuinely independent evidence. Pocketscribe never pools the two into one
      consensus score.
    </figcaption>
  </figure>

  <p>
    Two pockets from different models count as the same pocket only when <strong>both</strong>
    criteria hold: their lining-residue sets overlap above a threshold once mapped through
    a sequence alignment, <strong>and</strong> their centroids fall within a distance
    tolerance after Kabsch superposition. Requiring both avoids matching distant cavities
    that share a few residues, and nearby cavities that share none.
  </p>

  <div class="callout">
    <p>
      The principle — that agreement between independent methods is stronger evidence than
      agreement between close relatives — is not new; structural biologists already reason
      this way about independent crystal forms. What Pocketscribe contributes is packaging
      it as an automatic pipeline step for predicted-structure pocket triage, with an
      identity gate, cross-model residue mapping, superposition and family-aware scoring.
    </p>
  </div>
</section>

<section id="sources">
  <h2>Structure sources</h2>
  <p class="sub">
    Every prediction tool encodes confidence differently, so each has its own adapter. If
    confidence cannot be found or parsed, Pocketscribe fails with a specific error rather
    than guessing — a silently wrong confidence value would poison every pocket caveat
    downstream.
  </p>
  <div class="scroll">
  <table>
    <thead><tr><th>Source</th><th>Tier</th><th>Confidence</th><th>Method family</th></tr></thead>
    <tbody>
      <tr><td><strong>AlphaFold2</strong></td><td class="t1">Tier 1</td><td>pLDDT in B-factor column</td><td>MSA / co-evolution</td></tr>
      <tr><td><strong>OpenFold</strong></td><td class="t1">Tier 1</td><td>pLDDT in B-factor column</td><td>MSA / co-evolution</td></tr>
      <tr><td><strong>ColabFold</strong></td><td class="t1">Tier 1</td><td>pLDDT in B-factor column</td><td>MSA / co-evolution</td></tr>
      <tr><td><strong>ESMFold</strong></td><td class="t1">Tier 1</td><td>pLDDT in B-factor column</td><td>Single-sequence LM</td></tr>
      <tr><td><strong>Boltz-1 / Boltz-2</strong></td><td class="t1">Tier 2 · implemented</td><td>sidecar JSON, 0–1 scale</td><td>MSA / co-evolution</td></tr>
      <tr><td>Protenix</td><td class="t2">Tier 2 · planned</td><td>atom-level, sidecar JSON</td><td>MSA / co-evolution</td></tr>
      <tr><td>AlphaFold3</td><td class="t2">Tier 2 · planned</td><td>AF3 mmCIF conventions</td><td>MSA / co-evolution</td></tr>
      <tr><td>RoseTTAFold</td><td class="t2">Tier 2 · planned</td><td>predicted LDDT, <code>.npz</code></td><td>MSA / co-evolution</td></tr>
    </tbody>
  </table>
  </div>
  <p style="font-size:14.5px;color:var(--ink-2);margin-top:14px">
    Adding a source is the most useful contribution you can make, and the adapter interface
    exists for it — there's a step-by-step walkthrough in
    <a href="https://github.com/Jonahnki/Pocketscribe/blob/main/CONTRIBUTING.md">CONTRIBUTING.md</a>.
  </p>
</section>

<section id="install">
  <h2>Install and run</h2>
  <p class="sub">Python 3.10+, plus fpocket as an external binary.</p>
  <pre><code># fpocket is a C program, not a pip package
conda install -c conda-forge fpocket

git clone https://github.com/Jonahnki/Pocketscribe.git
cd Pocketscribe &amp;&amp; pip install -e .

# try it on a bundled synthetic example — no network needed
pocketscribe demo --output demo_report.html

# a real run
pocketscribe run --pdb model.pdb --source alphafold2 --md-setup --output report.html

# cross-model consensus: 2+ predictions of the SAME protein
pocketscribe run \
    --pdb af2_model.pdb:alphafold2 \
    --pdb esm_model.pdb:esmfold \
    --output consensus.html</code></pre>
  <p style="font-size:14.5px;color:var(--ink-2)">
    Source auto-detection is available with <code>--source auto</code>; it always prints the
    evidence behind its guess so a wrong one is visible and correctable.
  </p>
</section>

<section id="built-on">
  <h2>Built on</h2>
  <p class="sub">
    Pocketscribe is an integration layer. It introduces no new pocket-detection algorithm,
    no new druggability model and no new force field. The science below is other people's —
    please cite it, not only this tool.
  </p>
  <ul class="plain refs">
    <li><strong>fpocket</strong> — Le Guilloux, Schmidtke &amp; Tuffery, <em>BMC Bioinformatics</em> 10, 168 (2009). Cavity detection.</li>
    <li><strong>fpocket druggability</strong> — Schmidtke &amp; Barril, <em>J. Med. Chem.</em> 53, 5858–5867 (2010).</li>
    <li><strong>GROMACS</strong> — Abraham <em>et al.</em>, <em>SoftwareX</em> 1–2, 19–25 (2015).</li>
    <li><strong>CHARMM36m</strong> — Huang <em>et al.</em>, <em>Nature Methods</em> 14, 71–73 (2017).</li>
    <li><strong>AlphaFold2</strong> — Jumper <em>et al.</em>, <em>Nature</em> 596, 583–589 (2021).</li>
    <li><strong>OpenFold</strong> — Ahdritz <em>et al.</em>, <em>Nature Methods</em> 21, 1514–1524 (2024).</li>
    <li><strong>ColabFold</strong> — Mirdita <em>et al.</em>, <em>Nature Methods</em> 19, 679–682 (2022).</li>
    <li><strong>ESMFold</strong> — Lin <em>et al.</em>, <em>Science</em> 379, 1123–1130 (2023).</li>
    <li><strong>Boltz-1</strong> — Wohlwend <em>et al.</em>, <em>bioRxiv</em> (2024).</li>
    <li><strong>Biopython</strong> — Cock <em>et al.</em>, <em>Bioinformatics</em> 25, 1422–1423 (2009). Parsing and alignment.</li>
    <li><strong>Kabsch superposition</strong> — Kabsch, <em>Acta Cryst. A</em> 32, 922–923 (1976).</li>
  </ul>
</section>

<footer>
  <p>
    <strong>Research use only.</strong> Pocketscribe is a first-pass triage tool. It is not
    a medical device, not a diagnostic, and carries no clinical or therapeutic claim. Every
    pocket it describes exists in a computational model, not an experimental structure. A
    high druggability score means a cavity has geometry and chemistry that tend to accompany
    ligandable sites — it is not evidence that any compound binds. It exists to help a group
    decide where to spend expert time, docking, simulation and experiments; never to replace
    them.
  </p>
  <p>
    MIT licensed ·
    <a href="https://github.com/Jonahnki/Pocketscribe">github.com/Jonahnki/Pocketscribe</a> ·
    <a href="https://github.com/Jonahnki/Pocketscribe/blob/main/CITATION.cff">How to cite</a>
  </p>
  <p>By John Adeyemo Adedeji · <a href="https://orcid.org/0009-0004-1257-4551">ORCID 0009-0004-1257-4551</a> · Osun State University, Osogbo, Nigeria</p>
</footer>

</div>
</body>
</html>
"""


CONSENSUS_DIAGRAM = r"""<svg viewBox="0 0 760 330" role="img" width="100%"
  aria-label="Schematic. Three predictions of one protein, grouped into two architecture families: AlphaFold2 and OpenFold are MSA-based, ESMFold is a single-sequence language model. A cavity detected by all three has cross-family support. A cavity detected only by AlphaFold2 and OpenFold, and missed by ESMFold, has same-family support only.">
  <rect width="760" height="330" fill="#fcfcfb"/>
  <g font-family="ui-sans-serif, system-ui, sans-serif">

  <text x="16" y="24" font-size="12" font-weight="600" fill="#52514e">THREE PREDICTIONS OF ONE PROTEIN</text>

  <!-- family group: MSA / co-evolution -->
  <rect x="16" y="38" width="250" height="118" rx="9" fill="none" stroke="#c3c2b7" stroke-dasharray="4 3"/>
  <text x="28" y="58" font-size="11.5" fill="#898781">MSA / co-evolution family</text>
  <rect x="30" y="68" width="106" height="34" rx="6" fill="#0B4FBF"/>
  <text x="83" y="90" font-size="13" fill="#fff" text-anchor="middle" font-weight="600">AlphaFold2</text>
  <rect x="146" y="68" width="106" height="34" rx="6" fill="#0B4FBF"/>
  <text x="199" y="90" font-size="13" fill="#fff" text-anchor="middle" font-weight="600">OpenFold</text>
  <text x="30" y="126" font-size="11.5" fill="#898781">shares an inference pathway —</text>
  <text x="30" y="142" font-size="11.5" fill="#898781">can reproduce the same errors</text>

  <!-- family group: single-sequence language model -->
  <rect x="286" y="38" width="160" height="118" rx="9" fill="none" stroke="#c3c2b7" stroke-dasharray="4 3"/>
  <text x="298" y="58" font-size="11.5" fill="#898781">Single-sequence LM</text>
  <rect x="300" y="68" width="106" height="34" rx="6" fill="#B8860B"/>
  <text x="353" y="90" font-size="13" fill="#fff" text-anchor="middle" font-weight="600">ESMFold</text>
  <text x="300" y="126" font-size="11.5" fill="#898781">no MSA — independent</text>
  <text x="300" y="142" font-size="11.5" fill="#898781">route to the same fold</text>

  <text x="470" y="96" font-size="12" fill="#898781">Pockets are matched across</text>
  <text x="470" y="113" font-size="12" fill="#898781">the models, then scored by</text>
  <text x="470" y="130" font-size="12" fill="#898781">which families agree.</text>

  <path d="M 230 168 L 230 186" stroke="#c3c2b7" stroke-width="1.5" fill="none"/>
  <path d="M 226 180 L 230 188 L 234 180 Z" fill="#c3c2b7"/>
  <path d="M 530 168 L 530 186" stroke="#c3c2b7" stroke-width="1.5" fill="none"/>
  <path d="M 526 180 L 530 188 L 534 180 Z" fill="#c3c2b7"/>

  <!-- outcome: cross-family -->
  <rect x="16" y="194" width="360" height="118" rx="8" fill="#fff" stroke="#0B4FBF" stroke-width="1.5"/>
  <text x="32" y="218" font-size="13.5" font-weight="650" fill="#0B4FBF">Found by both families</text>
  <text x="32" y="238" font-size="11" fill="#898781">DETECTED BY</text>
  <rect x="32" y="246" width="80" height="22" rx="4" fill="#0B4FBF"/>
  <text x="72" y="261" font-size="11" fill="#fff" text-anchor="middle">AlphaFold2</text>
  <rect x="118" y="246" width="74" height="22" rx="4" fill="#0B4FBF"/>
  <text x="155" y="261" font-size="11" fill="#fff" text-anchor="middle">OpenFold</text>
  <rect x="198" y="246" width="72" height="22" rx="4" fill="#B8860B"/>
  <text x="234" y="261" font-size="11" fill="#fff" text-anchor="middle">ESMFold</text>
  <text x="32" y="288" font-size="12" fill="#52514e">Both families agree — the strongest robustness</text>
  <text x="32" y="303" font-size="12" fill="#52514e">signal short of an experiment.</text>

  <!-- outcome: same-family only -->
  <rect x="396" y="194" width="348" height="118" rx="8" fill="#fff" stroke="#B8860B" stroke-width="1.5"/>
  <text x="412" y="218" font-size="13.5" font-weight="650" fill="#B8860B">Found by one family only</text>
  <text x="412" y="238" font-size="11" fill="#898781">DETECTED BY</text>
  <rect x="412" y="246" width="80" height="22" rx="4" fill="#0B4FBF"/>
  <text x="452" y="261" font-size="11" fill="#fff" text-anchor="middle">AlphaFold2</text>
  <rect x="498" y="246" width="74" height="22" rx="4" fill="#0B4FBF"/>
  <text x="535" y="261" font-size="11" fill="#fff" text-anchor="middle">OpenFold</text>
  <rect x="578" y="246" width="88" height="22" rx="4" fill="none" stroke="#c3c2b7" stroke-dasharray="3 2"/>
  <text x="622" y="261" font-size="11" fill="#898781" text-anchor="middle">not ESMFold</text>
  <text x="412" y="288" font-size="12" fill="#52514e">One family only — the two models can share an</text>
  <text x="412" y="303" font-size="12" fill="#52514e">error, so this is not independent confirmation.</text>

  </g>
</svg>"""


def main() -> None:
    # Note for future edits: never hardcode a pocket rank, score or volume into the
    # page copy. Those depend on which backend produced the report -- fpocket and the
    # built-in fallback rank pockets differently, and fpocket volumes vary between runs
    # -- so a specific claim here silently goes stale. Keep the prose about *kinds* of
    # finding and let the linked report carry the numbers.
    print("Building the Pocketscribe site")
    generate_demo_reports()

    report = (DEMO / "report.html").read_text()
    confidence_plot = extract_figure(report, r'<svg class="viz" role="img".*?</svg>')

    page = PAGE.replace("__CONFIDENCE_PLOT__", confidence_plot)
    page = page.replace("__CONSENSUS_DIAGRAM__", CONSENSUS_DIAGRAM)

    (SITE / "index.html").write_text(page, encoding="utf-8")
    size = (SITE / "index.html").stat().st_size
    print(f"  wrote index.html ({size // 1024} KB, figures embedded from the real report)")
    print("\nPreview locally:")
    print("  python -m http.server -d site 8000   →  http://localhost:8000")


if __name__ == "__main__":
    main()
