"""GD-9: unit tests for the Article 30 RoPA endpoints (app/routers/records.py)."""
from app.routers.records import _build_ccpa, _build_ropa


def _find(activities: list[dict], name: str) -> dict:
    [activity] = [a for a in activities if a["name"] == name]
    return activity


def test_content_moderation_review_is_declared():
    """CM-5 / VIE01-408: a human reviewer looking at a flagged asset is a distinct
    processing activity from the automated guardrail check already listed under
    "AI/ML inference and chat completions" -- it needs its own RoPA entry, not a
    footnote on that one."""
    ropa = _build_ropa()
    activity = _find(ropa["record_of_processing_activities"], "Content moderation review")
    assert activity["legal_basis"].startswith("Legitimate interest")
    assert "Moderation administrators (reviewers)" in activity["data_subjects"]
    assert activity["data_categories"]  # not an empty declaration


def test_content_moderation_review_has_a_ccpa_equivalent():
    ccpa = _build_ccpa()
    activity = _find(ccpa["categories_of_personal_information_collected"], "Content moderation review")
    assert activity["business_purpose"].startswith("Legitimate interest")
    assert "Moderation administrators (reviewers)" in activity["consumers"]
