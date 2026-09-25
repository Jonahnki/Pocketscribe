# Contributing to Pocketscribe

Thank you for considering a contribution. This document is meant to be usable rather
than ceremonial: the first section is a complete walkthrough of the contribution the
project most needs, written so you can follow it start to finish.

## Getting set up

```bash
git clone https://github.com/jonahnki/pocketscribe.git
cd pocketscribe
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

The full test suite runs offline and does not need fpocket, GROMACS or network access.
If `pytest` passes on a clean checkout, your environment is ready.

---

## Adding a new structure-source adapter

This is the primary contribution path. New structure-prediction tools appear constantly,
each encoding confidence its own way, and the adapter interface exists so that supporting
one does not mean touching the rest of the pipeline.

The worked example below implements **Protenix**, which currently ships as a Tier 2 stub
in `pocketscribe/sources.py`. Substitute your own source throughout.

### Step 0 — understand how the source stores confidence

This is the part that matters, and the part you cannot skip. Find out, from the tool's own
repository or paper:

- Where per-residue confidence lives: the PDB B-factor column, a sidecar JSON, an `.npz`
  array, or an mmCIF category.
- What scale it is on: 0–100 (pLDDT convention) or 0–1.
- Whether it is per-residue or per-atom. If per-atom, it must be aggregated to per-residue
  (Pocketscribe uses the mean over a residue's atoms).
- How the tool's output can be recognised: filename patterns, header lines, the presence
  of a sidecar file.

Write this down — it goes into the adapter's docstring, where the next person will need it.

### Step 1 — implement the adapter

Adapters live in `pocketscribe/sources.py` and satisfy the `StructureSource` protocol:
an `info` attribute, a `detect()` method and an `extract_confidence()` method.

Replace the `ProtenixSource` stub with a real implementation:

```python
class ProtenixSource:
    """Protenix (ByteDance, Apache-2.0) -- an open AlphaFold3 reproduction.

    Confidence convention: Protenix writes predictions as mmCIF with a per-sample
    ``*_confidence.json`` carrying atom-level pLDDT on a 0-100 scale. Atom-level values
    are averaged per residue here. The mmCIF B-factor column is NOT read: for a
    diffusion model it is not guaranteed to carry pLDDT at all.
    """

    info = SourceInfo(
        id="protenix",
        display_name="Protenix",
        tier=SupportTier.TIER_2_REFERENCE,      # promoted from TIER_2_STUB
        family=ArchitectureFamily.MSA_COEVOLUTION,
        metric=ConfidenceMetric.PLDDT,
        uses_msa=True,
        notes=(
            "Protenix is an all-atom diffusion model reproducing AlphaFold3. Its "
            "atom-level pLDDT is averaged per residue by Pocketscribe."
        ),
        citation_key="protenix2025",
    )

    def detect(self, path: Path, text_head: str) -> tuple[bool, list[str]]:
        evidence: list[str] = []
        if _head_contains(text_head, "PROTENIX"):
            evidence.append("header mentions Protenix")
        if self._find_sidecar(path) is not None:
            evidence.append("found a Protenix confidence sidecar next to the structure")
        return bool(evidence), evidence

    @staticmethod
    def _find_sidecar(path: Path) -> Path | None:
        for candidate in (
            path.with_name(f"{path.stem}_confidence.json"),
            path.with_name(f"confidence_{path.stem}.json"),
        ):
            if candidate.is_file():
                return candidate
        return None

    def extract_confidence(self, path: Path, residues: list[dict]):
        sidecar = self._find_sidecar(path)
        if sidecar is None:
            raise ConfidenceExtractionError(
                "Protenix stores per-residue confidence in a sidecar JSON, not in the "
                f"B-factor column, and none was found next to '{path.name}'. Expected "
                f"{path.stem}_confidence.json. Pocketscribe will not read the B-factor "
                "column for Protenix, because the result would be wrong."
            )
        ...  # parse, aggregate per residue, return (metric, {residue_key: value})
```

Four rules the implementation must follow. They are what the rest of the pipeline relies
on, and a pull request that breaks any of them will be asked to change:

1. **Return confidence on a 0–100 scale.** Every threshold downstream (70, 50) assumes it.
   Normalise a 0–1 source, as `BoltzSource` does.
2. **Never fall back to the B-factor column** unless the source genuinely writes pLDDT
   there. Reading the wrong column produces a confident-looking report built on a wrong
   number, which is worse than any error message.
3. **Raise `ConfidenceExtractionError` with a specific, actionable message** when
   confidence cannot be found. Name the file you looked for and what the user should do.
4. **Make `detect()` explain itself.** The evidence strings are printed to the user so a
   wrong guess is visible and correctable.

Then register it. The adapter is already in the registry loop at the bottom of
`sources.py`; check the detection order in `_DETECTION_ORDER` — more specific matchers
must come before more permissive ones, and `alphafold2` stays last because it is the most
permissive.

### Step 2 — add a fixture showing the source's real output format

Fixtures are generated, not hand-edited, so their provenance is auditable. Add your
source to `scripts/make_example_structures.py`, following the `boltz_*` example:

```python
residues_protenix = build_residues(perturb(ca, 0.45, seed=15), SEQUENCE_A, axes)
write_pdb(
    OUTPUT_DIR / "protenix_synth01_sample_0.pdb",
    residues_protenix,
    np.zeros(n_residues),          # confidence is NOT in the B-factor column
    [
        "HEADER    PREDICTED STRUCTURE                     01-JAN-24   SYN1",
        "TITLE     PROTENIX PREDICTION FOR SYNTHETIC TEST PROTEIN 1",
        "REMARK   1 SYNTHETIC STRUCTURE GENERATED BY POCKETSCRIBE FOR TESTING.",
    ],
)
(OUTPUT_DIR / "protenix_synth01_sample_0_confidence.json").write_text(...)
```

Then regenerate:

```bash
python scripts/make_example_structures.py
```

The fixture must reflect the source's **actual** header, filename and confidence
conventions. A relabelled copy of the AlphaFold2 file proves nothing about detection,
which is precisely where sources differ.

If your source uses a format the generator cannot produce, a small checked-in sample is
acceptable — put it in `tests/fixtures/` with a README noting that it is a hand-written
format sample, as `tests/fixtures/fpocket_out/` does.

### Step 3 — add tests

Add a fixture path to `tests/conftest.py`:

```python
@pytest.fixture(scope="session")
def protenix_path(example_dir: Path) -> Path:
    return example_dir / "protenix_synth01_sample_0.pdb"
```

Then in `tests/test_sources.py`, at minimum:

```python
def test_protenix_confidence(protenix_path):
    parsed = parse_structure(protenix_path)
    metric, values = ProtenixSource().extract_confidence(protenix_path, parsed.residues)
    assert len(values) == parsed.qc.n_residues
    assert 0.0 <= min(values.values()) <= max(values.values()) <= 100.0


def test_protenix_does_not_fall_back_to_bfactors(tmp_path, protenix_path):
    """Without its sidecar, the adapter must raise rather than read the wrong column."""
    copied = tmp_path / protenix_path.name
    copied.write_text(protenix_path.read_text())
    parsed = parse_structure(copied)
    with pytest.raises(ConfidenceExtractionError, match="sidecar"):
        ProtenixSource().extract_confidence(copied, parsed.residues)
```

Also add your source to the auto-detection parametrisation:

```python
@pytest.mark.parametrize(
    "fixture_name,expected_id",
    [
        ...,
        ("protenix_path", "protenix"),
    ],
)
```

And remove it from the Tier 2 stub parametrisations
(`test_tier_2_stub_fails_with_a_source_specific_message` and
`test_tier_2_stub_never_returns_numbers`), since it is no longer a stub.

The suite already asserts that every registered source declares a citation, so
`test_every_source_declares_a_citation` will fail until you complete step 5.

### Step 4 — update the README table

Move your source from Tier 2 to the appropriate row of the supported-sources table in
`README.md`, with its confidence convention and method family.

### Step 5 — add or verify the citation

Every source must be correctly citable. Check `pocketscribe/citations.py`, and pull the
reference from the tool's own repository or publication rather than from memory. If the
source already has a stub entry, verify it is still correct.

Add the reference to the "Built on" section of `README.md` too.

**Getting a citation wrong is a research-integrity problem, not a formatting detail.**
These reports are meant to be cited in other people's methods sections.

### Step 6 — decide the architecture family carefully

`ArchitectureFamily` drives the entire cross-model consensus module. Getting it wrong
means Pocketscribe will present non-independent agreement as independent evidence, which
is the one conclusion the module exists to prevent.

- `MSA_COEVOLUTION` — uses a multiple sequence alignment or co-evolutionary signal.
- `SINGLE_SEQUENCE_LM` — a protein language model on a single sequence, no MSA.
- `UNKNOWN` — you are not sure. This is the **correct** choice when in doubt: an unknown
  family never counts as cross-family agreement, so it fails safe.

If your source is a close derivative of an existing one — a reimplementation, a fine-tune,
a wrapper around the same structure module — it belongs in the same family as its parent,
however different the packaging is. OpenFold and ColabFold both sit in AlphaFold2's family
for exactly this reason.

### Step 7 — open the pull request

Describe the source, where its confidence lives, and how you verified the adapter against
real output from it. If you tested only against a synthetic fixture, say so — that is fine
and worth knowing.

---

## Other contributions

**Bug reports** are valuable, especially ones involving real output from a real prediction
tool. Use the bug report template and include the source, a minimal structure file if you
can share one, and the exact command.

**A pocket-detection backend** (DoGSiteScorer, P2Rank, ...) would be welcome. Follow the
shape of `FpocketBackend`: a `name`, `is_fpocket`, `version()` and `detect()`. Any score
that is not fpocket's validated druggability score must be tagged with its own
`ScoreProvenance` so it can never be presented as equivalent.

**Documentation** improvements, particularly clearer explanations of what the confidence
caveats mean in practice, are genuinely useful.

**Please open an issue before starting large refactors**, so effort is not wasted on a
direction the project cannot take.

---

## Code style and what must pass

Before a pull request can be merged:

```bash
ruff check .        # must pass clean
pytest              # must pass in full
```

Plus:

- **No coverage regression.** New code paths need tests. New behaviour that can fail
  silently needs a test that makes it fail loudly.
- **Type hints on public functions.** The codebase uses `from __future__ import
  annotations` throughout.
- **Docstrings that explain why, not what.** The reader can see what the code does. What
  they cannot see is why a threshold is 0.2, or why a check exists at all.
- **Errors must be specific and actionable.** Name the file, the expected format, and what
  the user should do. "Invalid input" is not an acceptable message anywhere in this
  codebase.
- **Determinism.** The same inputs must produce the same report. Anything using randomness
  needs a fixed seed.

---

## Project boundaries — non-negotiable

Contributions must not:

- **Add clinical-use claims.** Pocketscribe is research-use-only software. No contribution
  may introduce language suggesting diagnostic, therapeutic or clinical applicability, and
  the disclaimer section of the report may not be weakened or made optional.
- **Introduce pricing or commercial language into this repository.** No "Pro" tiers, no
  upsells, no hosted-service scaffolding, no feature gating. This repository is
  MIT-licensed open research software and stays that way.
- **Weaken the confidence-caveat behaviour for any source.** Every pocket gets a caveat.
  Confidence that cannot be established is reported as unknown and escalated, never
  silently treated as good. An adapter may not read a column that does not hold
  confidence, and may not guess.
- **Present a heuristic score as a validated one.** Any score that is not fpocket's
  druggability score must carry its own provenance tag and be labelled wherever it appears.
- **Make the narrative layer a dependency of the core pipeline.** Pocketscribe must run to
  completion, and produce a complete report, with narrative synthesis entirely disabled.

These are architectural commitments, enforced by tests. If a change you want to make seems
to require breaking one, open an issue and let us talk about it first.

---

## Code of conduct

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
