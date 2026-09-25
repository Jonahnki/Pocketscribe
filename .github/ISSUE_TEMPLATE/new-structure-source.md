---
name: New structure source
about: Request or propose support for a structure-prediction tool Pocketscribe cannot read yet
title: "[source] Support <tool name>"
labels: ["new-source", "enhancement"]
---

## Which tool?

**Name:**
**Repository / paper:**
**License:**

## How does it store per-residue confidence?

This is the part that decides whether an adapter is a small change or a large one, so
please be as specific as you can.

- [ ] pLDDT in the PDB B-factor column (the AlphaFold2 convention)
- [ ] A sidecar file (JSON, `.npz`, ...) — which file, and what is it called?
- [ ] An mmCIF category — which one?
- [ ] Something else / not sure

**Scale:** 0–100, 0–1, or other?

**Granularity:** per residue, or per atom (needing aggregation)?

**Anything else about the format** — chain naming, multiple samples per run, files that
must be kept together:

## How can its output be recognised?

Filename patterns, header lines, the presence of a particular sidecar — whatever would
let `--source auto` identify it without guessing.

## Architecture family

This determines how the tool is treated in cross-model consensus. Agreement between
families is reported as stronger evidence than agreement within one, so getting it wrong
would present non-independent agreement as independent.

- [ ] MSA / co-evolution based
- [ ] Single-sequence protein language model
- [ ] Not sure

Is it a derivative, reimplementation or fine-tune of an existing model? If so, which one?

## Sample output

A small example file (or a link to one) makes this enormously easier to implement and
test. Please make sure anything you attach is yours to share.

## Are you planning to implement it?

- [ ] Yes, I'd like to open a PR — see the adapter walkthrough in CONTRIBUTING.md
- [ ] No, I'm requesting it
