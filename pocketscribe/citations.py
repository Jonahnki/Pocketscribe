"""Citations for every external tool and method Pocketscribe wraps.

Pocketscribe invents no science. fpocket does the cavity geometry and druggability
scoring, GROMACS and CHARMM36m do the simulation physics, and the structure-prediction
models do the folding. A report that gets cited in someone's methods section must point
at those primary sources, so they live here in one place, shared by the README, the
rendered report and ``pocketscribe sources``.

Keeping this list correct is a research-integrity obligation, not a formatting detail.
If you add an adapter, add its citation here in the same commit.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    """One primary reference."""

    key: str
    name: str
    reference: str
    url: str = ""
    role: str = ""


CITATIONS: dict[str, Citation] = {
    "fpocket": Citation(
        key="fpocket",
        name="fpocket",
        reference=(
            "Le Guilloux, V., Schmidtke, P. & Tuffery, P. Fpocket: an open source "
            "platform for ligand pocket detection. BMC Bioinformatics 10, 168 (2009). "
            "doi:10.1186/1471-2105-10-168"
        ),
        url="https://github.com/Discngine/fpocket",
        role="Cavity detection and alpha-sphere descriptors.",
    ),
    "fpocket_druggability": Citation(
        key="fpocket_druggability",
        name="fpocket druggability score",
        reference=(
            "Schmidtke, P. & Barril, X. Understanding and predicting druggability. A "
            "high-throughput method for detection of drug binding sites. Journal of "
            "Medicinal Chemistry 53, 5858-5867 (2010). doi:10.1021/jm100574m"
        ),
        url="https://doi.org/10.1021/jm100574m",
        role="The trained druggability model reported in the pocket table.",
    ),
    "gromacs": Citation(
        key="gromacs",
        name="GROMACS",
        reference=(
            "Abraham, M. J., Murtola, T., Schulz, R., Pall, S., Smith, J. C., Hess, B. "
            "& Lindahl, E. GROMACS: High performance molecular simulations through "
            "multi-level parallelism from laptops to supercomputers. SoftwareX 1-2, "
            "19-25 (2015). doi:10.1016/j.softx.2015.06.001"
        ),
        url="https://www.gromacs.org/",
        role="The MD engine the generated input files target.",
    ),
    "charmm36m": Citation(
        key="charmm36m",
        name="CHARMM36m force field",
        reference=(
            "Huang, J., Rauscher, S., Nawrocki, G., Ran, T., Feig, M., de Groot, B. L., "
            "Grubmuller, H. & MacKerell, A. D. Jr. CHARMM36m: an improved force field "
            "for folded and intrinsically disordered proteins. Nature Methods 14, "
            "71-73 (2017). doi:10.1038/nmeth.4067"
        ),
        url="https://doi.org/10.1038/nmeth.4067",
        role="Default force field in the generated MD setup.",
    ),
    "jumper2021": Citation(
        key="jumper2021",
        name="AlphaFold2",
        reference=(
            "Jumper, J., Evans, R., Pritzel, A. et al. Highly accurate protein structure "
            "prediction with AlphaFold. Nature 596, 583-589 (2021). "
            "doi:10.1038/s41586-021-03819-2"
        ),
        url="https://github.com/google-deepmind/alphafold",
        role="Tier 1 structure source.",
    ),
    "varadi2024": Citation(
        key="varadi2024",
        name="AlphaFold Protein Structure Database",
        reference=(
            "Varadi, M., Bertoni, D., Magana, P. et al. AlphaFold Protein Structure "
            "Database in 2024: providing structure coverage for over 214 million protein "
            "sequences. Nucleic Acids Research 52, D368-D375 (2024). "
            "doi:10.1093/nar/gkad1011"
        ),
        url="https://alphafold.ebi.ac.uk/",
        role="Source of AlphaFold DB models, if that is where the input came from.",
    ),
    "ahdritz2024": Citation(
        key="ahdritz2024",
        name="OpenFold",
        reference=(
            "Ahdritz, G., Bouatta, N., Floristean, C. et al. OpenFold: retraining "
            "AlphaFold2 yields new insights into its learning mechanisms and capacity "
            "for generalization. Nature Methods 21, 1514-1524 (2024). "
            "doi:10.1038/s41592-024-02272-z"
        ),
        url="https://github.com/aqlaboratory/openfold",
        role="Tier 1 structure source.",
    ),
    "mirdita2022": Citation(
        key="mirdita2022",
        name="ColabFold",
        reference=(
            "Mirdita, M., Schutze, K., Moriwaki, Y., Heo, L., Ovchinnikov, S. & "
            "Steinegger, M. ColabFold: making protein folding accessible to all. "
            "Nature Methods 19, 679-682 (2022). doi:10.1038/s41592-022-01488-1"
        ),
        url="https://github.com/sokrypton/ColabFold",
        role="Tier 1 structure source.",
    ),
    "lin2023": Citation(
        key="lin2023",
        name="ESMFold / ESM-2",
        reference=(
            "Lin, Z., Akin, H., Rao, R. et al. Evolutionary-scale prediction of "
            "atomic-level protein structure with a language model. Science 379, "
            "1123-1130 (2023). doi:10.1126/science.ade2574"
        ),
        url="https://github.com/facebookresearch/esm",
        role="Tier 1 structure source; the single-sequence family in consensus.",
    ),
    "wohlwend2024": Citation(
        key="wohlwend2024",
        name="Boltz-1 / Boltz-2",
        reference=(
            "Wohlwend, J., Corso, G., Passaro, S. et al. Boltz-1: democratizing "
            "biomolecular interaction modeling. bioRxiv (2024). "
            "doi:10.1101/2024.11.19.624167"
        ),
        url="https://github.com/jwohlwend/boltz",
        role="Tier 2 reference structure source.",
    ),
    "protenix2025": Citation(
        key="protenix2025",
        name="Protenix",
        reference=(
            "ByteDance AML AI4Science Team. Protenix: advancing structure prediction "
            "through a comprehensive AlphaFold3 reproduction. bioRxiv (2025). "
            "doi:10.1101/2025.01.08.631967"
        ),
        url="https://github.com/bytedance/Protenix",
        role="Tier 2 stub; adapter not yet implemented.",
    ),
    "abramson2024": Citation(
        key="abramson2024",
        name="AlphaFold3",
        reference=(
            "Abramson, J., Adler, J., Dunger, J. et al. Accurate structure prediction of "
            "biomolecular interactions with AlphaFold 3. Nature 630, 493-500 (2024). "
            "doi:10.1038/s41586-024-07487-w"
        ),
        url="https://github.com/google-deepmind/alphafold3",
        role="Tier 2 stub; adapter not yet implemented.",
    ),
    "baek2021": Citation(
        key="baek2021",
        name="RoseTTAFold",
        reference=(
            "Baek, M., DiMaio, F., Anishchenko, I. et al. Accurate prediction of protein "
            "structures and interactions using a three-track neural network. Science "
            "373, 871-876 (2021). doi:10.1126/science.abj8754"
        ),
        url="https://github.com/RosettaCommons/RoseTTAFold",
        role="Tier 2 stub; deprioritised relative to newer models.",
    ),
    "biopython": Citation(
        key="biopython",
        name="Biopython",
        reference=(
            "Cock, P. J. A., Antao, T., Chang, J. T. et al. Biopython: freely available "
            "Python tools for computational molecular biology and bioinformatics. "
            "Bioinformatics 25, 1422-1423 (2009). doi:10.1093/bioinformatics/btp163"
        ),
        url="https://biopython.org/",
        role="Structure parsing and pairwise sequence alignment.",
    ),
    "kabsch1976": Citation(
        key="kabsch1976",
        name="Kabsch superposition",
        reference=(
            "Kabsch, W. A solution for the best rotation to relate two sets of vectors. "
            "Acta Crystallographica A 32, 922-923 (1976). doi:10.1107/S0567739476001873"
        ),
        url="https://doi.org/10.1107/S0567739476001873",
        role="Structural superposition in the consensus module.",
    ),
    "hendlich1997": Citation(
        key="hendlich1997",
        name="LIGSITE",
        reference=(
            "Hendlich, M., Rippmann, F. & Barnickel, G. LIGSITE: automatic and efficient "
            "detection of potential small molecule-binding sites in proteins. Journal of "
            "Molecular Graphics and Modelling 15, 359-363 (1997). "
            "doi:10.1016/S1093-3263(98)00002-3"
        ),
        url="https://doi.org/10.1016/S1093-3263(98)00002-3",
        role=(
            "The grid buriedness scan the built-in offline fallback backend follows. "
            "Only relevant when fpocket is unavailable."
        ),
    ),
}


def core_citations() -> list[Citation]:
    """The references every run depends on, in reading order."""
    return [
        CITATIONS["fpocket"],
        CITATIONS["fpocket_druggability"],
        CITATIONS["gromacs"],
        CITATIONS["charmm36m"],
        CITATIONS["biopython"],
    ]


def citations_for_report(
    source_keys: list[str], used_fpocket: bool, used_consensus: bool
) -> list[Citation]:
    """Assemble the 'Built on' list for one report, without duplicates."""
    selected: list[Citation] = []

    def add(citation: Citation | None) -> None:
        if citation is not None and citation not in selected:
            selected.append(citation)

    if used_fpocket:
        add(CITATIONS["fpocket"])
        add(CITATIONS["fpocket_druggability"])
    else:
        add(CITATIONS["hendlich1997"])

    for key in source_keys:
        add(CITATIONS.get(key))

    add(CITATIONS["gromacs"])
    add(CITATIONS["charmm36m"])
    add(CITATIONS["biopython"])
    if used_consensus:
        add(CITATIONS["kabsch1976"])
    return selected
