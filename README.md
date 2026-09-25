# Pocketscribe

**Confidence-aware binding-pocket triage and molecular dynamics setup for predicted
protein structures.**

Pocketscribe takes a structure predicted by AlphaFold2, OpenFold, ColabFold, ESMFold or
Boltz and produces one self-contained report covering structure quality control,
binding-pocket detection, druggability scoring that is caveated by the prediction's own
per-residue confidence, optional cross-model consensus, and ready-to-run GROMACS input
files.

Research use only. It is a first-pass triage tool, not a substitute for expert
structural biology judgement, experimental validation, or a real docking or MD study.

---

## Statement of need

An academic drug-discovery group with a predicted structure and no crystallographic
data faces a practical problem: deciding whether any site on that model is worth the
cost of a docking campaign, a simulation, or a wet-lab experiment. The existing tools
each solve one piece of it and leave the group to join them up.

**Pocket detection tools are bare scoring engines.** fpocket, DoGSiteScorer and P2Rank
find cavities and score them well, but they were designed for experimental structures
and treat every input the same way. Handed an AlphaFold model, they will score a cavity
formed by a disordered loop with pLDDT of 35 exactly as confidently as one in a
well-resolved core. The per-residue confidence that the prediction came with — the
single most important piece of context for interpreting a predicted structure — is
simply not part of their output.

**Web platforms require uploading unpublished structures to a third party.** For a
group working on an unpublished target, that is often not an option, and it rules out
the interactive tools that would otherwise be the easiest route.

**Family-specific druggability profilers do not generalise.** Tools trained on kinases
or GPCRs give sharp answers inside their family and little outside it, which is
precisely the situation a group exploring a novel or poorly-characterised target is in.

**Nothing carries the analysis through to simulation.** Even after a promising pocket is
identified, the step from "we found a pocket" to "we are simulating it" means assembling
a GROMACS setup by hand from tutorials — a reliable source of quiet errors, of which
running CHARMM36m on GROMACS' default non-bonded settings is only the most common.

Pocketscribe addresses these together. It treats per-residue confidence as a first-class
signal that caveats every downstream claim; it reads output from the prediction tools
academic groups actually use rather than assuming one source; it runs entirely locally,
so no unpublished structure leaves the machine; and it carries a run through to
ready-to-use MD input files and a written protocol in a single pass, ending in one
static report file rather than scattered outputs or a live web session.

### Cross-model pocket consensus

One component goes beyond integration. When a user supplies structures for the same
protein from more than one prediction tool, Pocketscribe superposes them, establishes
which pockets in one model correspond to which in another, and scores the agreement in
an **architecture-family-aware** way: agreement between an MSA-based model
(AlphaFold2, OpenFold, ColabFold, Boltz) and a single-sequence language model (ESMFold)
is reported separately from, and as stronger evidence than, agreement between two models
of the same family, which can reproduce each other's systematic errors. The two are
never pooled into one undifferentiated consensus score.

The underlying principle — that agreement between independent methods is stronger
evidence than agreement between close relatives — is not new to science; structural
biologists already reason this way about independent crystal forms. The contribution
here is packaging it as an automatic pipeline step for predicted-structure pocket
triage: sequence-identity gating, cross-model residue mapping, Kabsch superposition,
two-criteria pocket correspondence matching, and confidence-weighted family-aware
agreement scoring. As far as we could determine at the time of writing, no existing
open pocket-detection tool does this.

### What Pocketscribe is not

Pocketscribe is a **workflow and integration contribution**: research software whose
value is in removing real friction for a community, independently of whether any single
algorithm inside it is new.

It introduces **no new pocket-detection algorithm, no new druggability model and no new
force field**. fpocket does the cavity geometry and the druggability scoring, GROMACS and
CHARMM36m do the simulation physics, and the structure-prediction models do the folding.
Every one of them is cited in [Built on](#built-on--cite-these-too) and in every
generated report. If a report contributed to your work, cite those tools, not only this
one.

If you know of a tool that already combines these capabilities, please open an issue —
this section will be corrected.

---

## Install

```bash
pip install pocketscribe
```

> Not yet published to PyPI. Until it is, install from source — see
> [From source](#from-source) below.

### fpocket (external dependency)

Pocket detection is done by **fpocket**, which is a C program and is not installed by
pip:

```bash
# conda (recommended, works everywhere)
conda install -c conda-forge fpocket

# Debian / Ubuntu
sudo apt-get install fpocket

# macOS
brew install fpocket

# from source
# https://github.com/Discngine/fpocket
```

Check it is visible:

```bash
fpocket --version
```

**If fpocket is not installed**, `pocketscribe demo` and the test suite fall back to a
built-in grid-based cavity scan so that the pipeline can still be tried offline. That
fallback is clearly labelled everywhere it appears, and its scores are a documented
heuristic — **not** fpocket's trained and validated druggability score. A real analysis
run (`pocketscribe run`) never falls back silently: it stops and tells you to install
fpocket.

### Optional: narrative synthesis

```bash
pip install 'pocketscribe[narrative]'
export ANTHROPIC_API_KEY=...
```

This enables one optional report section, described under
[The optional narrative layer](#the-optional-narrative-layer). Everything else runs
offline, and the report is complete without it.

### From source

```bash
git clone https://github.com/jonahnki/pocketscribe.git
cd pocketscribe
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

---

## Quick start

```bash
# Try it on a bundled synthetic example -- no network, no fpocket needed
pocketscribe demo --output demo_report.html

# The cross-model consensus path, on three bundled models of one protein
pocketscribe demo --consensus --output demo_consensus_report.html

# A real run
pocketscribe run --pdb model.pdb --source alphafold2 --output report.html

# Let Pocketscribe work out where the model came from (it always says what it found)
pocketscribe run --pdb model.pdb --source auto --output report.html

# Add MD setup for the top-ranked pocket
pocketscribe run --pdb model.pdb --source colabfold --md-setup --output report.html

# Cross-model consensus: 2+ predictions of the SAME protein, each tagged with its source
pocketscribe run \
    --pdb af2_model.pdb:alphafold2 \
    --pdb esm_model.pdb:esmfold \
    --pdb boltz_model.pdb:boltz \
    --output consensus_report.html

# What can Pocketscribe read?
pocketscribe sources
```

---

## Supported structure sources

Confidence is encoded differently by every prediction tool, so each has its own adapter.
If a source's confidence cannot be found or parsed, Pocketscribe **fails with a specific
error** rather than guessing — a silently wrong confidence value would poison every
pocket caveat downstream.

| Source | Tier | Confidence | Method family | Status |
| --- | --- | --- | --- | --- |
| **AlphaFold2** | Tier 1 | pLDDT in B-factor column (0–100) | MSA / co-evolution | Fully supported |
| **OpenFold** | Tier 1 | pLDDT in B-factor column (0–100) | MSA / co-evolution | Fully supported |
| **ColabFold** | Tier 1 | pLDDT in B-factor column (0–100) | MSA / co-evolution | Fully supported |
| **ESMFold** | Tier 1 | pLDDT in B-factor column (0–100) | Single-sequence LM | Fully supported |
| **Boltz-1 / Boltz-2** | Tier 2 | per-residue pLDDT in sidecar JSON (0–1) | MSA / co-evolution | Reference adapter, implemented |
| **Protenix** | Tier 2 | atom-level pLDDT in sidecar JSON | MSA / co-evolution | Declared, not implemented |
| **AlphaFold3** | Tier 2 | per-atom pLDDT + AF3 mmCIF conventions | MSA / co-evolution | Declared, not implemented |
| **RoseTTAFold** | Tier 2 | predicted LDDT in sidecar `.npz` | MSA / co-evolution | Declared, not implemented |

**Tier 1** means parsed, auto-detected, individually unit tested with that source's own
header and filename conventions.

**Tier 2 "reference adapter"** means implemented and tested. Boltz is here because it is
the working example of a source that does *not* use the legacy B-factor convention.

**Tier 2 "declared, not implemented"** means the adapter exists and raises a clear,
source-specific error naming what is missing and why the AlphaFold2 parser cannot simply
be reused. It will never attempt a parse that would be wrong.

Adding a source is the most useful contribution you can make, and the architecture is
designed for it — see [CONTRIBUTING.md](CONTRIBUTING.md).

### Two things Pocketscribe refuses to do

**It will not read an experimental structure as a prediction.** Crystallographic B-factors
sit comfortably inside the 0–100 pLDDT range, so a range check is not enough. Pocketscribe
instead checks whether B-factors vary *within* a residue: pLDDT is a per-residue quantity
written identically to every atom of a residue, while a refined temperature factor is
not. A file that fails this check is rejected with an explanation.

**It will not guess a source.** `--source auto` prints the evidence behind its guess so
you can correct it, and raises if nothing matches rather than defaulting to AlphaFold2.

---

## What the report contains

1. **Executive summary** — pocket count, confidence headline, what the run was.
2. **Input structure** — detected source and its support tier, chains, residue counts,
   how the source was established.
3. **Structure quality and prediction confidence** — per-residue confidence plot, band
   counts as a table, flagged low-confidence regions, geometry checks.
4. **Ranked pockets** — volume, druggability, hydrophobicity, polarity, lining residues,
   and a confidence caveat on every pocket.
5. **Cross-model consensus** — only when 2+ structures are supplied.
6. **Pocket locations** — Cα trace with lining residues coloured by their own confidence.
7. **MD setup summary** — only with `--md-setup`.
8. **Interpretation** — only when the optional narrative layer is enabled.
9. **Methods and parameters** — everything needed to reproduce the run.
10. **Scope and limitations** — the disclaimer, in full.
11. **How to cite** — Pocketscribe and, more importantly, the tools it wrapped.

The report is a **single HTML file** with the CSS inlined and every figure as inline SVG.
No scripts, no CDN, no external assets: it opens offline, prints cleanly, and will still
render years from now on a machine that no longer has Pocketscribe installed.

---

## How confidence is used

Pocket claims are caveated, not merely annotated. Each pocket's lining residues are
cross-referenced against the confidence map, and the pocket is assigned a caveat level:

| Level | Roughly when | What it means |
| --- | --- | --- |
| `none` | Confidently predicted lining residues | Normal predicted-structure caveats apply |
| `low` | Some less certain residues at the periphery | Side-chain placement may shift |
| `moderate` | A meaningful minority below 70, or mean below 70 | Cavity may be real; shape and volume uncertain |
| `severe` | ≥25% below 50, or mean below 50 | Do not commit resources without independent evidence |

Escalation is deliberately conservative: a pocket is escalated on *either* a poor
average *or* a substantial minority of untrustworthy residues, because a binding site
needs only a few badly-placed lining residues to be geometrically wrong.

---

## Cross-model consensus

Supply two or more predictions of the same protein and Pocketscribe adds a consensus
stage. It is skipped entirely for a single structure, and single-structure reports are
unaffected by its existence.

**Method.**

1. **Identity gate.** Sequences are aligned and required to be near-identical (≥95% by
   default, `--min-identity`). Different proteins are rejected outright — a
   plausible-looking consensus table for two unrelated proteins is worse than an error.
2. **Residue mapping.** A residue-to-residue map is built from the alignment, so models
   differing in numbering, missing residues or construct boundaries can still be compared.
3. **Superposition.** Corresponding Cα atoms are Kabsch-superposed onto the first model.
4. **Pocket correspondence.** Two pockets from different models are the same pocket only
   if **both** hold: lining-residue Jaccard overlap (mapped through the alignment) above
   a threshold, **and** post-superposition centroid separation within tolerance.
   Requiring both avoids matching distant cavities that share a few residues, and nearby
   cavities that share none.
5. **Family-aware agreement.** Each model is tagged with an architecture family.
   Cross-family and same-family agreement are computed and reported separately, each
   normalised by how many pairs of that kind the input set could have produced, and
   optionally weighted by each model's own confidence in the pocket's lining residues.

**Reading the output.** A pocket labelled `cross-family` was found independently by
methods that do not share an inference pathway — the strongest robustness signal this
pipeline can offer short of experimental validation. A pocket labelled `same-family-only`
was found by models that may share systematic errors, and is explicitly *not* independent
confirmation. `single-model` means no consensus support at all.

To get anything out of this stage you need models from different families. Three
AlphaFold2-lineage models will tell you very little; adding one ESMFold prediction
changes that, and the report says so when you have not.

---

## MD simulation setup

`--md-setup` generates, but never runs, a complete GROMACS setup:

| File | Purpose |
| --- | --- |
| `protein_clean.pdb` | Protein-only coordinates, heteroatoms and hydrogens stripped |
| `run_setup.sh` | pdb2gmx → editconf → solvate → genion → minimisation → NVT → NPT → production |
| `minim.mdp` | Steepest-descent energy minimisation |
| `ions.mdp` | Parameters for building the `genion` `.tpr` |
| `nvt.mdp` | 100 ps constant-volume equilibration, protein restrained |
| `npt.mdp` | 100 ps constant-pressure equilibration, protein restrained |
| `md.mdp` | Unrestrained production run |
| `pocket_selection.txt` | `gmx make_ndx` selection for the target pocket's residues |
| `PROTOCOL.md` | Step-by-step protocol explaining every choice and when to change it |

Defaults are CHARMM36m with TIP3P water, a rhombic dodecahedron box with 1.0 nm padding,
neutralising ions plus 0.15 M NaCl, 310 K, 2 fs timestep with h-bond constraints. The
non-bonded settings follow the CHARMM36 recommendation (force-switched van der Waals
from 1.0 to 1.2 nm, PME at a matching 1.2 nm cut-off) and are stated explicitly in every
`.mdp` file, because running CHARMM36m on GROMACS' defaults is a common and consequential
error that survives copy-and-paste.

Everything a user should reconsider for their own system is commented in place.

---

## The optional narrative layer

One report section is written by a language model: a short scientific interpretation of
which pockets look most promising, what the confidence caveats mean for reading them, and
a recommended next step.

It is **genuinely optional and structurally isolated**:

- If `ANTHROPIC_API_KEY` is unset, or the `anthropic` package is not installed, or the
  call fails for any reason, the section is skipped and the report is complete without it.
- It runs last and receives only the finished analysis as JSON — pocket scores, confidence
  statistics, consensus results. **No coordinates, no file contents, no structure.** What
  leaves your machine is a table of numbers about pockets, not your unpublished structure.
- The prompt instructs the model to reinforce the report's caveats rather than smooth
  them over, and the rendered section is labelled as model-written.
- `--no-narrative` disables it outright.

No part of the deterministic pipeline imports it.

---

## Configuration

```yaml
# config.yaml
pockets:
  backend: fpocket        # fpocket | builtin
  top_pockets: 5
consensus:
  min_sequence_identity: 0.95
  centroid_tolerance: 8.0      # Angstrom
  min_residue_jaccard: 0.25
  confidence_weighting: true
md:
  enabled: true
  force_field: charmm36m
  temperature_K: 310.0
  production_ns: 100.0
narrative:
  enabled: true
```

```bash
pocketscribe run --pdb model.pdb --config config.yaml --output report.html
```

CLI options override the file. Every parameter ends up in the report's methods appendix.

---

## Built on — cite these too

Pocketscribe is an integration layer. The science in any report it produces was done by
the tools below. **Citing Pocketscribe without them misattributes the work.**

**Pocket detection and druggability**

- Le Guilloux, V., Schmidtke, P. & Tuffery, P. Fpocket: an open source platform for
  ligand pocket detection. *BMC Bioinformatics* **10**, 168 (2009).
  doi:[10.1186/1471-2105-10-168](https://doi.org/10.1186/1471-2105-10-168)
- Schmidtke, P. & Barril, X. Understanding and predicting druggability. A
  high-throughput method for detection of drug binding sites. *Journal of Medicinal
  Chemistry* **53**, 5858–5867 (2010).
  doi:[10.1021/jm100574m](https://doi.org/10.1021/jm100574m)

**Molecular dynamics**

- Abraham, M. J. *et al.* GROMACS: High performance molecular simulations through
  multi-level parallelism from laptops to supercomputers. *SoftwareX* **1–2**, 19–25
  (2015). doi:[10.1016/j.softx.2015.06.001](https://doi.org/10.1016/j.softx.2015.06.001)
- Huang, J. *et al.* CHARMM36m: an improved force field for folded and intrinsically
  disordered proteins. *Nature Methods* **14**, 71–73 (2017).
  doi:[10.1038/nmeth.4067](https://doi.org/10.1038/nmeth.4067)

**Structure prediction sources**

- **AlphaFold2** — Jumper, J. *et al.* Highly accurate protein structure prediction with
  AlphaFold. *Nature* **596**, 583–589 (2021).
  doi:[10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2)
- **AlphaFold DB** — Varadi, M. *et al.* AlphaFold Protein Structure Database in 2024.
  *Nucleic Acids Research* **52**, D368–D375 (2024).
  doi:[10.1093/nar/gkad1011](https://doi.org/10.1093/nar/gkad1011)
- **OpenFold** — Ahdritz, G. *et al.* OpenFold: retraining AlphaFold2 yields new insights
  into its learning mechanisms and capacity for generalization. *Nature Methods* **21**,
  1514–1524 (2024). doi:[10.1038/s41592-024-02272-z](https://doi.org/10.1038/s41592-024-02272-z)
- **ColabFold** — Mirdita, M. *et al.* ColabFold: making protein folding accessible to
  all. *Nature Methods* **19**, 679–682 (2022).
  doi:[10.1038/s41592-022-01488-1](https://doi.org/10.1038/s41592-022-01488-1)
- **ESMFold / ESM-2** — Lin, Z. *et al.* Evolutionary-scale prediction of atomic-level
  protein structure with a language model. *Science* **379**, 1123–1130 (2023).
  doi:[10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574)
- **Boltz-1** — Wohlwend, J. *et al.* Boltz-1: democratizing biomolecular interaction
  modeling. *bioRxiv* (2024).
  doi:[10.1101/2024.11.19.624167](https://doi.org/10.1101/2024.11.19.624167)

Declared but not yet implemented (cited here so the adapter stubs point somewhere):
**Protenix** (doi:[10.1101/2025.01.08.631967](https://doi.org/10.1101/2025.01.08.631967)),
**AlphaFold3** (Abramson, J. *et al. Nature* **630**, 493–500, 2024,
doi:[10.1038/s41586-024-07487-w](https://doi.org/10.1038/s41586-024-07487-w)),
**RoseTTAFold** (Baek, M. *et al. Science* **373**, 871–876, 2021,
doi:[10.1126/science.abj8754](https://doi.org/10.1126/science.abj8754)).

**Supporting libraries and methods**

- Cock, P. J. A. *et al.* Biopython. *Bioinformatics* **25**, 1422–1423 (2009).
  doi:[10.1093/bioinformatics/btp163](https://doi.org/10.1093/bioinformatics/btp163)
- Kabsch, W. A solution for the best rotation to relate two sets of vectors. *Acta
  Crystallographica A* **32**, 922–923 (1976).
  doi:[10.1107/S0567739476001873](https://doi.org/10.1107/S0567739476001873)
- Hendlich, M., Rippmann, F. & Barnickel, G. LIGSITE. *Journal of Molecular Graphics and
  Modelling* **15**, 359–363 (1997) — the grid buriedness scan the offline fallback
  backend follows. Relevant only when fpocket is unavailable.

Please verify each reference against its source before it goes into a manuscript.

---

## Limitations

- **Predicted structures are models.** They place side chains wrongly, close cavities
  that are open in reality, and open cavities that are not. Apo predictions in particular
  often miss the conformational change that accompanies binding.
- **Druggability scores are heuristics.** A high score means a cavity has the geometry
  and chemistry that tend to accompany ligandable sites. It is not evidence that anything
  binds, and says nothing about affinity or selectivity.
- **Consensus is not truth.** Independent methods can be independently wrong, especially
  for proteins with few homologues or unusual folds.
- **No simulation is run.** The MD stage generates inputs with reasonable defaults, not a
  protocol validated for your system.
- **Multi-chain consensus pairs chains in sorted order.** Structures whose chains are
  named or ordered differently may need renaming first.
- **The offline fallback backend is not fpocket** and its scores are not comparable to
  fpocket's. Install fpocket for anything you intend to act on.

---

## How to cite

If Pocketscribe contributed to published work, please cite it **together with** the tools
above. A machine-readable [`CITATION.cff`](CITATION.cff) ships with the repository.

> Adedeji, J. A. Pocketscribe: confidence-aware binding-pocket triage and molecular
> dynamics setup for predicted protein structures (version 0.1.0).
> https://github.com/jonahnki/pocketscribe

---

## Contributing

Contributions are welcome, and the structure-source adapter architecture exists
specifically to make one kind of contribution easy. See
[CONTRIBUTING.md](CONTRIBUTING.md) for a walkthrough that adds a working adapter
step by step, and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

MIT — see [LICENSE](LICENSE).
