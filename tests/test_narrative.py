"""The optional narrative layer.

Two properties matter more than anything the model writes:

1. With no API key, the pipeline completes and the report is still complete.
2. Nothing but derived, structured numbers ever crosses the API boundary -- no
   coordinates, no file contents, no unpublished structure.
"""

from __future__ import annotations

import json

import pytest

from pocketscribe.config import RunConfig
from pocketscribe.narrative import SYSTEM_PROMPT, build_payload, synthesize
from pocketscribe.pipeline import StructureInput, run_pipeline


@pytest.fixture(scope="module")
def analysis(tmp_path_factory, request):
    tmp_path = tmp_path_factory.mktemp("narrative")
    af2_path = request.getfixturevalue("af2_path")
    config = RunConfig()
    config.narrative.enabled = False
    result, _ = run_pipeline(
        inputs=[
            StructureInput(path=af2_path, source_id="alphafold2", label=af2_path.name)
        ],
        config=config,
        md_output_dir=tmp_path,
        allow_backend_fallback=True,
    )
    return result


# --------------------------------------------------------------------------------------
# Skipping cleanly
# --------------------------------------------------------------------------------------


def test_skips_without_an_api_key(analysis, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    outcome = synthesize(analysis)
    assert outcome.enabled is False
    assert outcome.text == ""
    assert "ANTHROPIC_API_KEY" in outcome.skipped_reason


def test_skipped_reason_reassures_that_the_rest_is_intact(analysis, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    outcome = synthesize(analysis)
    assert "deterministic" in outcome.skipped_reason


def test_skips_when_explicitly_disabled(analysis):
    outcome = synthesize(analysis, enabled=False)
    assert outcome.enabled is False
    assert "disabled" in outcome.skipped_reason


def test_an_api_failure_never_costs_the_user_the_report(analysis, monkeypatch):
    """Any exception must be swallowed into a skipped result."""

    class _Exploding:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("network on fire")

    import sys
    import types

    fake = types.ModuleType("anthropic")
    fake.Anthropic = _Exploding
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    outcome = synthesize(analysis)
    assert outcome.enabled is False
    assert "network on fire" in outcome.skipped_reason


def test_missing_anthropic_package_is_handled(analysis, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "anthropic":
            raise ImportError("no module named anthropic")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    outcome = synthesize(analysis)
    assert outcome.enabled is False
    assert "pocketscribe[narrative]" in outcome.skipped_reason


# --------------------------------------------------------------------------------------
# The payload boundary
# --------------------------------------------------------------------------------------


def test_payload_is_json_serialisable(analysis):
    payload = build_payload(analysis)
    json.dumps(payload)  # must not raise


def test_payload_carries_no_coordinates(analysis):
    """The unpublished structure itself must never leave the machine."""
    payload = build_payload(analysis)
    serialised = json.dumps(payload)
    assert "coord" not in serialised.lower()

    structure = analysis.primary_structure
    for pocket in analysis.primary_pockets.pockets:
        if pocket.centroid:
            for value in pocket.centroid:
                assert f"{value:.3f}" not in serialised
    assert structure.path not in serialised


def test_payload_carries_the_confidence_caveats(analysis):
    """Without these the model could not write an honest interpretation."""
    payload = build_payload(analysis)
    pocket = payload["structures"][0]["pockets"][0]
    assert "confidence" in pocket
    assert "caveat" in pocket["confidence"]
    assert "caveat_level" in pocket["confidence"]
    assert payload["structures"][0]["confidence"]["fraction_below_70"] is not None


def test_payload_flags_a_non_fpocket_backend(analysis):
    payload = build_payload(analysis)
    if not analysis.primary_pockets.backend_is_fpocket:
        assert payload["pocket_backend"]["warning"]
        assert "NOT fpocket" in payload["pocket_backend"]["warning"]


def test_payload_includes_consensus_when_present(tmp_path, af2_path, esmfold_path):
    config = RunConfig()
    config.narrative.enabled = False
    result, _ = run_pipeline(
        inputs=[
            StructureInput(path=af2_path, source_id="alphafold2", label=af2_path.name),
            StructureInput(path=esmfold_path, source_id="esmfold", label=esmfold_path.name),
        ],
        config=config,
        md_output_dir=tmp_path,
        allow_backend_fallback=True,
    )
    payload = build_payload(result)
    assert "consensus" in payload
    assert payload["consensus"]["cross_family_comparison_possible"] is True
    assert payload["consensus"]["pocket_groups"][0]["evidence"]


# --------------------------------------------------------------------------------------
# The prompt must reinforce the report's caveats, not contradict them
# --------------------------------------------------------------------------------------


def test_prompt_forbids_overclaiming():
    lowered = SYSTEM_PROMPT.lower()
    assert "do not overclaim" in lowered
    assert "predicted" in lowered
    assert "confirmed binding site" in lowered
    assert "clinical" in lowered
    assert "do not invent numbers" in lowered


def test_prompt_encodes_the_consensus_evidence_hierarchy():
    """The narrative must not present same-family agreement as independent confirmation."""
    import re

    # The prompt is written with line continuations, so whitespace is collapsed.
    lowered = re.sub(r"\s+", " ", SYSTEM_PROMPT.lower())
    assert "architecture families" in lowered
    assert "never present same-family agreement as independent confirmation" in lowered


def test_prompt_requires_flagging_heuristic_scores():
    assert "heuristic fallback backend" in SYSTEM_PROMPT


# --------------------------------------------------------------------------------------
# Mocked success path
# --------------------------------------------------------------------------------------


def test_successful_synthesis_returns_the_text(analysis, monkeypatch):
    import sys
    import types

    captured: dict = {}

    class _Block:
        type = "text"
        text = "Pocket 1 is the stronger candidate.\n\nTreat pocket 2 cautiously."

    class _Response:
        content = [_Block()]

    class _Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Response()

    class _Client:
        def __init__(self, api_key=None):
            captured["api_key"] = api_key
            self.messages = _Messages()

    fake = types.ModuleType("anthropic")
    fake.Anthropic = _Client
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    outcome = synthesize(analysis, model="claude-sonnet-4-5")
    assert outcome.enabled is True
    assert "Pocket 1 is the stronger candidate." in outcome.text
    assert outcome.model == "claude-sonnet-4-5"

    assert captured["model"] == "claude-sonnet-4-5"
    assert captured["system"] == SYSTEM_PROMPT
    # What was sent is the JSON summary, not the structure.
    sent = captured["messages"][0]["content"]
    assert "```json" in sent
    assert "ATOM" not in sent


def test_empty_model_output_is_treated_as_a_skip(analysis, monkeypatch):
    import sys
    import types

    class _Response:
        content = []

    class _Messages:
        def create(self, **kwargs):
            return _Response()

    class _Client:
        def __init__(self, api_key=None):
            self.messages = _Messages()

    fake = types.ModuleType("anthropic")
    fake.Anthropic = _Client
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    outcome = synthesize(analysis)
    assert outcome.enabled is False
    assert "no text" in outcome.skipped_reason


def test_narrative_text_appears_in_the_rendered_report(analysis, tmp_path, monkeypatch):
    import sys
    import types

    from pocketscribe.parsing import parse_structure
    from pocketscribe.report.render import render_report

    class _Block:
        type = "text"
        text = "The first pocket is the stronger candidate for a docking campaign."

    class _Response:
        content = [_Block()]

    class _Messages:
        def create(self, **kwargs):
            return _Response()

    class _Client:
        def __init__(self, api_key=None):
            self.messages = _Messages()

    fake = types.ModuleType("anthropic")
    fake.Anthropic = _Client
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    analysis.narrative = synthesize(analysis)
    parsed = [parse_structure(analysis.primary_structure.path)]
    html = render_report(analysis, parsed, tmp_path / "r.html").read_text()

    assert "stronger candidate for a docking campaign" in html
    # And it must be labelled as model-written, not presented as a finding.
    assert "Written by an AI language model" in html
