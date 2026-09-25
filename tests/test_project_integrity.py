"""Project-level guarantees.

These tests cover commitments that live in prose rather than in a function: the
repository's non-commercial boundary, the accuracy of the citation metadata, and the
documentation staying in step with the code. They are the reason a contributor can trust
CONTRIBUTING.md enough to follow it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pocketscribe import __version__
from pocketscribe.citations import CITATIONS, core_citations
from pocketscribe.models import SupportTier
from pocketscribe.sources import all_sources

ROOT = Path(__file__).resolve().parent.parent

TEXT_SUFFIXES = {".py", ".md", ".toml", ".cff", ".yml", ".yaml", ".html", ".css", ".txt"}
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache",
             "build", "dist", "example_structures"}


def _repository_text_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in path.parts):
            continue
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            files.append(path)
    return files


# --------------------------------------------------------------------------------------
# Non-negotiable project boundaries
# --------------------------------------------------------------------------------------

#: Phrases that would indicate pricing, tiering or hosted-service framing leaking into
#: this repository. The boundary is architectural: that lives in a separate codebase.
#: Files that legitimately name these phrases in order to forbid or test for them.
#: Everything else in the repository is scanned.
GUARD_FILES = {
    "CONTRIBUTING.md",              # states the prohibition
    "tests/test_report.py",         # asserts the phrases are absent from reports
    "tests/test_project_integrity.py",
}


def _is_guard_file(path: Path) -> bool:
    return str(path.relative_to(ROOT)) in GUARD_FILES


COMMERCIAL_PHRASES = [
    "pricing",
    "subscription",
    "free tier",
    "paid tier",
    "paid plan",
    "pro plan",
    "pro tier",
    "premium",
    "upgrade to",
    "per month",
    "per seat",
    "billing",
    "buy now",
    "start your trial",
    "contact sales",
    "enterprise plan",
]


def test_repository_contains_no_commercial_language():
    offences: list[str] = []
    for path in _repository_text_files():
        if _is_guard_file(path):
            continue
        lowered = path.read_text(errors="replace").lower()
        for phrase in COMMERCIAL_PHRASES:
            if phrase in lowered:
                offences.append(f"{path.relative_to(ROOT)}: {phrase!r}")
    assert not offences, "commercial language found:\n" + "\n".join(offences)


#: Claims this software is not entitled to make about a predicted structure.
CLINICAL_PHRASES = [
    "diagnose",
    "diagnostic use",
    "treat patients",
    "therapeutic indication",
    "clinically proven",
    "medically approved",
    "cure for",
]


def test_repository_makes_no_clinical_claim():
    offences: list[str] = []
    for path in _repository_text_files():
        if _is_guard_file(path):
            continue
        text = path.read_text(errors="replace").lower()
        for phrase in CLINICAL_PHRASES:
            if phrase in text:
                # "not a diagnostic" and similar disclaimers are the point, not a breach.
                for line in text.splitlines():
                    if phrase in line and "not " not in line and "never " not in line:
                        offences.append(f"{path.relative_to(ROOT)}: {line.strip()[:80]}")
    assert not offences, "clinical claims found:\n" + "\n".join(offences)


def test_license_is_mit_and_names_the_author():
    text = (ROOT / "LICENSE").read_text()
    assert "MIT License" in text
    assert "John Adeyemo Adedeji" in text


# --------------------------------------------------------------------------------------
# Citation integrity
# --------------------------------------------------------------------------------------


def test_citation_cff_is_valid_yaml_and_current():
    data = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    assert data["cff-version"] == "1.2.0"
    assert data["license"] == "MIT"
    assert str(data["version"]) == __version__, "CITATION.cff version is out of date"
    assert data["authors"][0]["orcid"]


def test_citation_cff_lists_the_tools_this_software_wraps():
    """A citation file that omits fpocket and GROMACS misattributes the science."""
    data = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    dois = {reference.get("doi", "") for reference in data["references"]}
    for required in (
        "10.1186/1471-2105-10-168",  # fpocket
        "10.1021/jm100574m",  # fpocket druggability
        "10.1016/j.softx.2015.06.001",  # GROMACS
        "10.1038/nmeth.4067",  # CHARMM36m
    ):
        assert required in dois, f"CITATION.cff is missing {required}"


def test_citation_cff_states_that_the_tools_must_be_cited_too():
    data = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    assert "cite it together with the tools it integrates" in data["message"]


def test_every_citation_has_a_doi_or_url():
    for key, citation in CITATIONS.items():
        assert citation.reference.strip(), key
        assert "doi:" in citation.reference or citation.url, key


def test_readme_built_on_section_covers_the_core_citations():
    readme = (ROOT / "README.md").read_text()
    assert "## Built on" in readme
    for citation in core_citations():
        # Match on the first author's surname, which is stable across formatting.
        surname = citation.reference.split(",")[0].split()[-1]
        assert surname in readme, f"README does not cite {citation.name}"


def test_readme_has_no_placeholder_citations():
    readme = (ROOT / "README.md").read_text().lower()
    for placeholder in ("et al. (year)", "todo", "tbd", "xxxx", "citation needed"):
        assert placeholder not in readme


# --------------------------------------------------------------------------------------
# Documentation stays in step with the code
# --------------------------------------------------------------------------------------


def test_readme_source_table_lists_every_registered_source():
    """A user must be able to see what is supported without reading the code."""
    readme = (ROOT / "README.md").read_text()
    table_section = readme.split("## Supported structure sources")[1].split("\n## ")[0]
    for adapter in all_sources():
        assert adapter.info.display_name.split(" (")[0] in table_section, (
            f"{adapter.info.id} is missing from the README source table"
        )


def test_readme_marks_tier_2_stubs_as_unimplemented():
    readme = (ROOT / "README.md").read_text()
    table_section = readme.split("## Supported structure sources")[1].split("\n## ")[0]
    for adapter in all_sources():
        if adapter.info.tier is SupportTier.TIER_2_STUB:
            name = adapter.info.display_name.split(" (")[0]
            row = next(line for line in table_section.splitlines() if name in line)
            assert "not implemented" in row.lower(), (
                f"{adapter.info.id} is a stub but the README does not say so"
            )


@pytest.mark.parametrize(
    "symbol",
    [
        "StructureSource",
        "SourceInfo",
        "SupportTier",
        "ArchitectureFamily",
        "ConfidenceMetric",
        "ConfidenceExtractionError",
        "_head_contains",
        "detect",
        "extract_confidence",
        "_DETECTION_ORDER",
    ],
)
def test_contributing_walkthrough_references_real_symbols(symbol):
    """CONTRIBUTING.md must be followable.

    Every symbol its adapter walkthrough names has to exist, or a contributor following
    it step by step hits a wall.
    """
    contributing = (ROOT / "CONTRIBUTING.md").read_text()
    assert symbol in contributing, f"CONTRIBUTING.md no longer mentions {symbol}"
    sources = (ROOT / "pocketscribe" / "sources.py").read_text()
    assert symbol in sources, f"{symbol} is referenced in CONTRIBUTING.md but not in sources.py"


@pytest.mark.parametrize(
    "path",
    [
        "scripts/make_example_structures.py",
        "tests/conftest.py",
        "tests/test_sources.py",
        "pocketscribe/citations.py",
        "pocketscribe/sources.py",
    ],
)
def test_contributing_points_at_files_that_exist(path):
    contributing = (ROOT / "CONTRIBUTING.md").read_text()
    assert path in contributing
    assert (ROOT / path).is_file()


def test_contributing_states_the_project_boundaries():
    contributing = (ROOT / "CONTRIBUTING.md").read_text().lower()
    assert "clinical" in contributing
    assert "pricing or commercial language" in contributing
    assert "weaken the confidence-caveat behaviour" in contributing


def test_contributing_states_what_must_pass_before_merge():
    contributing = (ROOT / "CONTRIBUTING.md").read_text()
    assert "ruff check ." in contributing
    assert "pytest" in contributing
    assert "coverage regression" in contributing


def test_issue_templates_exist_and_ask_the_right_questions():
    new_source = (ROOT / ".github/ISSUE_TEMPLATE/new-structure-source.md").read_text()
    assert "confidence" in new_source.lower()
    assert "repository" in new_source.lower()
    assert "architecture family" in new_source.lower()
    assert "sample output" in new_source.lower()

    bug = (ROOT / ".github/ISSUE_TEMPLATE/bug-report.md").read_text()
    assert "pocketscribe version" in bug.lower()
    assert "fpocket" in bug.lower()
    assert "which tool produced it" in bug.lower()


def test_code_of_conduct_exists():
    text = (ROOT / "CODE_OF_CONDUCT.md").read_text()
    assert "Contributor Covenant" in text


# --------------------------------------------------------------------------------------
# Packaging
# --------------------------------------------------------------------------------------


def test_example_structures_ship_as_package_data():
    """pocketscribe demo must work after a plain pip install."""
    import tomllib

    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    from pocketscribe.data import EXAMPLE_DIR

    assert EXAMPLE_DIR.is_dir()
    assert list(EXAMPLE_DIR.glob("*.pdb"))
    # The directory sits inside the package, so setuptools' package discovery includes it.
    assert "pocketscribe" in str(EXAMPLE_DIR)
    assert config["project"]["name"] == "pocketscribe"


def test_report_templates_are_declared_as_package_data():
    import tomllib

    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    declared = config["tool"]["setuptools"]["package-data"]["pocketscribe"]
    assert any("templates" in pattern and pattern.endswith(".html") for pattern in declared)
    assert any("templates" in pattern and pattern.endswith(".css") for pattern in declared)


def test_version_is_consistent_across_the_project():
    import tomllib

    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert config["project"]["version"] == __version__


def test_fpocket_is_documented_as_an_external_dependency():
    """It is not a pip dependency, so the install instructions have to cover it."""
    import re

    readme = (ROOT / "README.md").read_text()
    assert "conda install -c conda-forge fpocket" in readme
    # The README wraps prose, so whitespace is collapsed before matching.
    assert "not installed by pip" in re.sub(r"\s+", " ", readme)
