"""CM-1 (VIE01-404): the moderation taxonomy -- its model and built-in
default (app/models/moderation.py, app/services/default_moderation.py), the
mapping from guard-model output to moderation categories
(app/services/moderation.py), the policy object form that carries it
(app/models/policy.py, app/routers/guardrail_policies.py), and the
`modality` option on the image guards (app/routers/infer_*.py)."""
from __future__ import annotations

import copy
import json

import pytest
from app.adapters.base import CATEGORY_NAMES
from app.models.moderation import (
    LLAMA_GUARD_CODES,
    MODALITIES,
    SHIELDGEMMA2_POLICIES,
    ModerationPolicy,
    parse_moderation,
)
from app.models.policy import parse_standard_policies
from app.routers import guardrail_policies, infer_llama_guard_vision, infer_shieldgemma2
from app.services import moderation, policy_check_service, policy_translator
from app.services.default_moderation import (
    DEFAULT_MODERATION,
    DEFAULT_MODERATION_STANDARD_POLICY,
)
from app.services.default_policy import DEFAULT_STANDARD_POLICY
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tests.unit.conftest import make_policy

DEFAULT = moderation.ResolvedModeration(moderation.DEFAULT_MODERATION_POLICY)
SG = {"dangerous": 0.01, "sexual": 0.02, "violence": 0.03}      # a clean ShieldGemma 2 result


def _sg(**scores: float) -> dict[str, float]:
    return {**SG, **scores}


# ── The built-in taxonomy ─────────────────────────────────────────────────────


def test_default_taxonomy_has_the_four_ticket_categories_and_validates():
    policy = ModerationPolicy.model_validate(DEFAULT_MODERATION)
    assert list(policy.categories) == ["nsfw", "violence", "hate", "self_harm"]
    for cat in policy.categories.values():
        assert set(cat.modalities) == set(MODALITIES)


def test_default_mapping():
    cats = moderation.DEFAULT_MODERATION_POLICY.categories
    assert (cats["nsfw"].llama_guard, cats["nsfw"].shieldgemma2) == (["S12", "S3", "S4"], ["sexual"])
    assert (cats["violence"].llama_guard, cats["violence"].shieldgemma2) == (["S1", "S9"], ["violence", "dangerous"])
    assert (cats["hate"].llama_guard, cats["hate"].shieldgemma2) == (["S10"], [])
    assert (cats["self_harm"].llama_guard, cats["self_harm"].shieldgemma2) == (["S11"], ["dangerous"])


def test_every_shieldgemma_policy_is_mapped():
    mapped = {p for c in moderation.DEFAULT_MODERATION_POLICY.categories.values() for p in c.shieldgemma2}
    assert mapped == set(SHIELDGEMMA2_POLICIES)


def test_mapped_s_codes_and_the_rest_are_left_to_the_guardrail_decision():
    mapped = {c for cat in moderation.DEFAULT_MODERATION_POLICY.categories.values() for c in cat.llama_guard}
    assert mapped == {"S1", "S3", "S4", "S9", "S10", "S11", "S12"}
    assert set(LLAMA_GUARD_CODES) - mapped == {"S2", "S5", "S6", "S7", "S8", "S13", "S14"}


def test_llama_guard_codes_match_the_adapter_taxonomy():
    assert set(LLAMA_GUARD_CODES) == set(CATEGORY_NAMES)


def test_default_thresholds_keep_the_old_image_guard_cut_off():
    for cat in moderation.DEFAULT_MODERATION_POLICY.categories.values():
        assert cat.modalities["image"].threshold == 0.5
        assert cat.modalities["video_frame"].threshold == 0.5
        assert cat.modalities["text"].threshold is None and cat.modalities["transcript"].threshold is None


def test_text_decisions_match_the_s_code_decisions_of_the_shipped_policy():
    """Activating the shipped moderation policy must not change text decisions:
    each category's text decision is the worst decision of its S-codes."""
    rank = {"allow": 0, "warn": 1, "review": 2, "block": 3}
    by_id = {c["id"]: c["decision"] for c in DEFAULT_STANDARD_POLICY}
    for cat in moderation.DEFAULT_MODERATION_POLICY.categories.values():
        worst = max((by_id[c] for c in cat.llama_guard), key=rank.__getitem__)
        assert cat.modalities["text"].decision == worst


# ── Model validation ──────────────────────────────────────────────────────────


def _taxonomy(**changes) -> dict:
    data = copy.deepcopy(DEFAULT_MODERATION)
    data["categories"]["hate"].update(changes)
    return data


@pytest.mark.parametrize("changes, message", [
    ({"llama_guard": ["S15"]}, "unknown Llama Guard code"),
    ({"shieldgemma2": ["hate"]}, "unknown ShieldGemma 2 policy"),
    ({"modalities": {"text": {"decision": "block"}}}, "missing modality"),
    ({"modalities": {**DEFAULT_MODERATION["categories"]["hate"]["modalities"],
                     "image": {"decision": "block"}}}, "needs a threshold"),
    ({"modalities": {**DEFAULT_MODERATION["categories"]["hate"]["modalities"],
                     "text": {"decision": "block", "threshold": 0.5}}}, "takes no threshold"),
    ({"modalities": {**DEFAULT_MODERATION["categories"]["hate"]["modalities"],
                     "image": {"decision": "block", "threshold": 1.5}}}, "less than or equal to 1"),
    ({"modalities": {**DEFAULT_MODERATION["categories"]["hate"]["modalities"],
                     "image": {"decision": "deny", "threshold": 0.5}}}, "decision"),
])
def test_invalid_taxonomy_is_rejected(changes, message):
    with pytest.raises(ValidationError, match=message):
        ModerationPolicy.model_validate(_taxonomy(**changes))


def test_empty_taxonomy_is_rejected():
    with pytest.raises(ValidationError, match="must not be empty"):
        ModerationPolicy.model_validate({"categories": {}})


def test_parse_moderation_reads_the_object_form_only():
    assert parse_moderation(json.dumps(DEFAULT_MODERATION_STANDARD_POLICY)).version == "1.0"
    assert parse_moderation(json.dumps(DEFAULT_STANDARD_POLICY)) is None           # plain array
    assert parse_moderation("rules: []") is None                                     # YAML
    assert parse_moderation(json.dumps({"categories": [], "moderation": {"bad": 1}})) is None
    assert parse_moderation(None) is None


# ── ShieldGemma 2 scores -> moderation ────────────────────────────────────────


def test_clean_image_is_allowed_and_hate_is_not_covered():
    v = moderation.moderate_scores(SG, "image", DEFAULT)
    assert (v["decision"], v["flagged"], v["not_covered"]) == ("allow", [], ["hate"])
    assert v["categories"]["hate"]["score"] is None
    assert (v["source"], v["policy_name"], v["modality"]) == ("default", None, "image")


def test_sexual_score_flags_nsfw():
    v = moderation.moderate_scores(_sg(sexual=0.97), "video_frame", DEFAULT)
    assert (v["decision"], v["flagged"]) == ("block", ["nsfw"])
    assert v["categories"]["nsfw"] == {"covered": True, "flagged": True, "decision": "block",
                                       "score": 0.97, "threshold": 0.5, "matched": ["sexual"]}


def test_dangerous_flags_both_violence_and_self_harm():
    v = moderation.moderate_scores(_sg(dangerous=0.8), "image", DEFAULT)
    assert v["flagged"] == ["violence", "self_harm"]
    assert v["categories"]["violence"]["score"] == 0.8      # highest of violence/dangerous


def test_threshold_is_inclusive():
    assert moderation.moderate_scores(_sg(violence=0.5), "image", DEFAULT)["flagged"] == ["violence"]
    assert moderation.moderate_scores(_sg(violence=0.499), "image", DEFAULT)["flagged"] == []


def _custom(**rules) -> moderation.ResolvedModeration:
    """The default taxonomy with nsfw's image/video_frame rules replaced."""
    data = copy.deepcopy(DEFAULT_MODERATION)
    data["categories"]["nsfw"]["modalities"].update(rules)
    return moderation.ResolvedModeration(ModerationPolicy.model_validate(data), "p", "2.0")


def test_thresholds_are_per_modality():
    resolved = _custom(image={"decision": "block", "threshold": 0.9},
                       video_frame={"decision": "block", "threshold": 0.6})
    assert moderation.moderate_scores(_sg(sexual=0.7), "image", resolved)["flagged"] == []
    assert moderation.moderate_scores(_sg(sexual=0.7), "video_frame", resolved)["flagged"] == ["nsfw"]


def test_worst_decision_wins():
    resolved = _custom(image={"decision": "warn", "threshold": 0.5})
    v = moderation.moderate_scores(_sg(sexual=0.9), "image", resolved)
    assert v["decision"] == "warn"
    v = moderation.moderate_scores(_sg(sexual=0.9, violence=0.9), "image", resolved)
    assert v["decision"] == "block"
    assert (v["source"], v["policy_name"], v["policy_version"]) == ("active_policy", "p", "2.0")


# ── Llama Guard S-codes -> moderation ─────────────────────────────────────────


@pytest.mark.parametrize("code, category", [
    ("S12", "nsfw"), ("S3", "nsfw"), ("S4", "nsfw"), ("S1", "violence"), ("S9", "violence"),
    ("S10", "hate"), ("S11", "self_harm"),
])
def test_each_mapped_code_flags_its_category(code, category):
    for modality in MODALITIES:
        v = moderation.moderate_codes([code], modality, DEFAULT)
        assert v["flagged"] == [category] and v["decision"] == "block"
        assert v["categories"][category]["matched"] == [code]


@pytest.mark.parametrize("code", ["S2", "S5", "S6", "S7", "S8", "S13", "S14"])
def test_unmapped_codes_flag_nothing(code):
    v = moderation.moderate_codes([code], "text", DEFAULT)
    assert (v["decision"], v["flagged"], v["not_covered"]) == ("allow", [], [])


# ── Resolving the taxonomy in force ───────────────────────────────────────────


@pytest.fixture
def active(monkeypatch):
    state = {"policy": None, "fail": None}

    async def fake_branch():
        if state["fail"]:
            raise state["fail"]
        return state["policy"]

    monkeypatch.setattr(policy_check_service, "get_policy_for_branch", fake_branch)
    return state


@pytest.mark.asyncio
async def test_active_policy_moderation_block_is_used(active):
    data = copy.deepcopy(DEFAULT_MODERATION_STANDARD_POLICY)
    data["moderation"]["version"] = "2.0"
    active["policy"] = make_policy(json.dumps(data), version="7")
    resolved = await moderation.resolve_moderation()
    assert (resolved.source, resolved.policy_name, resolved.policy_version) == ("active_policy", "p", "7")
    assert resolved.moderation.version == "2.0"


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", [None, make_policy(json.dumps(DEFAULT_STANDARD_POLICY))])
async def test_no_policy_or_no_block_uses_the_default(active, policy):
    active["policy"] = policy
    resolved = await moderation.resolve_moderation()
    assert resolved.source == "default" and resolved.moderation is moderation.DEFAULT_MODERATION_POLICY


@pytest.mark.asyncio
async def test_failed_lookup_uses_the_default(active):
    active["fail"] = ConnectionError("redis down")
    assert (await moderation.resolve_moderation()).source == "default"


# ── The policy object form ────────────────────────────────────────────────────


def test_object_form_parses_to_the_same_categories_as_the_array():
    as_object = parse_standard_policies(json.dumps(DEFAULT_MODERATION_STANDARD_POLICY))
    assert as_object == parse_standard_policies(json.dumps(DEFAULT_STANDARD_POLICY))


def test_object_form_with_other_keys_is_not_a_standard_policy():
    assert parse_standard_policies(json.dumps({"categories": DEFAULT_STANDARD_POLICY, "name": "x"})) is None


def test_object_form_composes_the_same_guard_prompts():
    """Activation composes per-model prompts from the categories; the
    moderation block must not change them."""
    assert (policy_translator.translate_policy_to_prompts(json.dumps(DEFAULT_MODERATION_STANDARD_POLICY))
            == policy_translator.translate_policy_to_prompts(json.dumps(DEFAULT_STANDARD_POLICY)))


def test_upload_validation_accepts_the_shipped_policy():
    guardrail_policies._validate_if_standard_format(json.dumps(DEFAULT_MODERATION_STANDARD_POLICY), "original")


@pytest.mark.parametrize("content, detail", [
    ({"categories": DEFAULT_STANDARD_POLICY, "moderation": {"categories": {}}}, "Invalid moderation block"),
    ({"categories": DEFAULT_STANDARD_POLICY, "moderation": DEFAULT_MODERATION, "x": 1}, "unexpected key"),
    ({"categories": [{"id": "S1"}], "moderation": DEFAULT_MODERATION}, "Invalid Standard Policy Format"),
])
def test_upload_validation_rejects_a_bad_object_form(content, detail):
    with pytest.raises(HTTPException) as exc:
        guardrail_policies._validate_if_standard_format(json.dumps(content), "original")
    assert exc.value.status_code == 422 and detail in exc.value.detail


def test_upload_validation_leaves_other_objects_alone():
    guardrail_policies._validate_if_standard_format(json.dumps({"name": "x", "rules": []}), "original")


@pytest.mark.asyncio
async def test_moderation_template_is_the_shipped_policy():
    t = await guardrail_policies.get_policy_template(format="moderation")
    assert json.loads(t.example_json) == DEFAULT_MODERATION_STANDARD_POLICY


# ── modality on the image guards ──────────────────────────────────────────────


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(infer_shieldgemma2.router)
    app.include_router(infer_llama_guard_vision.router)
    return TestClient(app)


FILES = {"image": ("a.png", b"\x89PNG fake", "image/png")}


def _sg_result(**scores: float) -> dict:
    return {"overall_verdict": "unsafe" if any(s > 0.5 for s in scores.values()) else "safe",
            "policies": {k: {"unsafe_probability": v, "safe_probability": 1 - v,
                             "verdict": "unsafe" if v > 0.5 else "safe", "policy_definition": "d"}
                         for k, v in _sg(**scores).items()}}


@pytest.fixture
def default_moderation(monkeypatch):
    async def fake_resolve():
        return DEFAULT
    monkeypatch.setattr(moderation, "resolve_moderation", fake_resolve)


def test_shieldgemma_without_modality_is_unchanged(monkeypatch, default_moderation):
    monkeypatch.setattr(infer_shieldgemma2, "run_shieldgemma2_inference", lambda *_a: _sg_result(sexual=0.9))
    body = _client().post("/infer/google-shieldgemma-2", files=FILES).json()
    assert "moderation" not in body and body["overall_verdict"] == "unsafe"


def test_shieldgemma_with_modality_adds_the_verdict(monkeypatch, default_moderation):
    monkeypatch.setattr(infer_shieldgemma2, "run_shieldgemma2_inference", lambda *_a: _sg_result(sexual=0.9))
    body = _client().post("/infer/google-shieldgemma-2", files=FILES, data={"modality": "video_frame"}).json()
    assert body["overall_verdict"] == "unsafe"                       # old fields kept
    m = body["moderation"]
    assert (m["modality"], m["decision"], m["flagged"], m["not_covered"]) == ("video_frame", "block", ["nsfw"], ["hate"])


def test_llama_guard_vision_with_modality_adds_the_verdict(monkeypatch, default_moderation):
    monkeypatch.setattr(infer_llama_guard_vision, "run_llama_guard_vision_inference",
                        lambda *_a: "unsafe\nS10")
    body = _client().post("/infer/Llama-Guard-3-11B-Vision", files=FILES, data={"modality": "image"}).json()
    assert body["categories"] == ["S10"]
    assert (body["moderation"]["flagged"], body["moderation"]["not_covered"]) == (["hate"], [])


@pytest.mark.parametrize("path", ["/infer/google-shieldgemma-2", "/infer/Llama-Guard-3-11B-Vision"])
def test_text_modalities_are_rejected_on_the_image_guards(path):
    assert _client().post(path, files=FILES, data={"modality": "text"}).status_code == 422
