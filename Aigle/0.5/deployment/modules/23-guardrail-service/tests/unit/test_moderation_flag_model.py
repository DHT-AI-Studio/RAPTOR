"""Unit tests for the CM-5 (VIE01-408) request models (app/models/moderation_flag.py)."""
import pytest
from app.models.moderation_flag import (
    FLAG_STATUSES,
    ModerationFlagCreate,
    ModerationFlagStatusUpdate,
)
from pydantic import ValidationError


def test_at_most_one_sub_reference_is_allowed():
    with pytest.raises(ValidationError):
        ModerationFlagCreate(branch_id="b-1", version_id="v-1", asset_path="a", moment_id="m-1", chunk_id="c",
                              category="nsfw", model="shieldgemma-2")   # both given
    ModerationFlagCreate(branch_id="b-1", version_id="v-1", asset_path="a", moment_id="m-1",
                          category="nsfw", model="shieldgemma-2")   # one: ok
    ModerationFlagCreate(branch_id="b-1", version_id="v-1", asset_path="a", chunk_id="c",
                          category="nsfw", model="shieldgemma-2")   # the other: also ok
    ModerationFlagCreate(branch_id="b-1", version_id="v-1", asset_path="a",
                          category="nsfw", model="shieldgemma-2")   # neither: whole-asset flag, also ok


def test_branch_id_is_required():
    with pytest.raises(ValidationError):
        ModerationFlagCreate(version_id="v-1", asset_path="a", category="nsfw", model="shieldgemma-2")


def test_version_id_is_required():
    with pytest.raises(ValidationError):
        ModerationFlagCreate(branch_id="b-1", asset_path="a", category="nsfw", model="shieldgemma-2")


def test_asset_path_is_required_even_for_a_moment_or_chunk_flag():
    """The whole point of this field -- an admin reviewing a moment/chunk flag still
    needs the real asset it belongs to, not just a one-way hash or a bare UUID."""
    with pytest.raises(ValidationError):
        ModerationFlagCreate(branch_id="b-1", version_id="v-1", moment_id="m-1",
                              category="nsfw", model="shieldgemma-2")


def test_flag_statuses_has_no_separate_reviewed_state():
    """'reviewed' was dropped: it named no distinct outcome ('cleared' and
    'removed' already are the review's outcome), so an admin transitions
    straight from 'flagged' to one of those two."""
    assert FLAG_STATUSES == ("flagged", "cleared", "removed")
    with pytest.raises(ValidationError):
        ModerationFlagStatusUpdate(status="reviewed")
