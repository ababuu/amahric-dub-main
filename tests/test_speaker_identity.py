"""Tests for :mod:`app.pipeline.speaker_identity`.

The registry decides whether a voice heard in one scene is a character the film has
already named, so these tests are about the decisions, not about torch: no model, no
audio, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.pipeline import speaker_identity as identity
from app.pipeline.speaker_identity import (
    AMBIGUOUS,
    CONFIRM_SCORE,
    CONFIRMED,
    IdentityEvidence,
    PROVISIONAL,
    assess_identity,
    MATCHED,
    UNKNOWN,
    GlobalSpeaker,
    InvalidVoiceprintError,
    SpeakerRegistry,
    UnreadableRegistryError,
    blend,
    cosine_similarity,
    normalise,
)


def test_normalise_scales_to_unit_length():
    assert normalise([3.0, 4.0]) == pytest.approx((0.6, 0.8))


def test_normalise_rejects_an_empty_vector():
    with pytest.raises(InvalidVoiceprintError):
        normalise([])


def test_normalise_rejects_an_all_zero_vector():
    with pytest.raises(InvalidVoiceprintError):
        normalise([0.0, 0.0])


def test_normalise_rejects_a_non_finite_value():
    with pytest.raises(InvalidVoiceprintError):
        normalise([1.0, float("nan")])


def test_cosine_similarity_of_a_vector_with_itself_is_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_of_different_widths_is_minus_one():
    """A stale registry entry from another embedding model must never match."""

    assert cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) == -1.0


def test_blend_weights_by_seconds_not_by_turn_count():
    """A character with 6 s outvotes one who says "yeah" ten times."""

    blended = blend([1.0, 0.0], 6.0, [0.0, 1.0], 2.0)
    assert blended[0] > blended[1]


def test_global_speaker_rejects_an_empty_id():
    with pytest.raises(InvalidVoiceprintError):
        GlobalSpeaker(speaker_id="", voiceprint=(1.0, 0.0))


def test_identify_matches_a_clearly_closer_identity():
    registry = SpeakerRegistry(
        {
            "SPEAKER_00": GlobalSpeaker("SPEAKER_00", (1.0, 0.0), evidence_seconds=5.0),
            "SPEAKER_01": GlobalSpeaker("SPEAKER_01", (0.0, 1.0), evidence_seconds=5.0),
        }
    )

    decision = registry.identify([0.98, 0.2])

    assert decision.outcome == MATCHED
    assert decision.speaker_id == "SPEAKER_00"


def test_identify_refuses_to_choose_between_two_similar_identities():
    """Two characters this close cannot be told apart, so neither is reused."""

    registry = SpeakerRegistry(
        {
            "SPEAKER_00": GlobalSpeaker("SPEAKER_00", (1.0, 0.0), evidence_seconds=5.0),
            "SPEAKER_01": GlobalSpeaker("SPEAKER_01", (0.9, 0.44), evidence_seconds=5.0),
        }
    )

    decision = registry.identify([1.0, 0.35])

    assert decision.outcome == AMBIGUOUS
    assert decision.margin < identity.DEFAULT_MATCH_MIN_MARGIN


def test_identify_reports_an_unknown_voice():
    registry = SpeakerRegistry(
        {"SPEAKER_00": GlobalSpeaker("SPEAKER_00", (1.0, 0.0), evidence_seconds=5.0)}
    )

    assert registry.identify([0.0, 1.0]).outcome == UNKNOWN


def test_identify_on_an_empty_registry_is_unknown():
    assert SpeakerRegistry().identify([1.0, 0.0]).outcome == UNKNOWN


def test_mint_uses_the_next_free_label():
    registry = SpeakerRegistry()
    registry.observe("SPEAKER_00", [1.0, 0.0], 2.0, turns=1)
    registry.mint([0.0, 1.0], 2.0, turns=1)

    assert registry.next_id() == "SPEAKER_02"
    assert "SPEAKER_01" in registry


def test_observe_keeps_adding_evidence_to_the_same_identity():
    registry = SpeakerRegistry()
    registry.observe("SPEAKER_00", [1.0, 0.0], 2.0, turns=2, scenes=1)
    registry.observe("SPEAKER_00", [1.0, 0.1], 3.0, turns=1, scenes=1)

    speaker = registry.get("SPEAKER_00")
    assert speaker is not None
    assert speaker.evidence_seconds == pytest.approx(5.0)
    assert speaker.turns == 3
    assert speaker.scenes == 2


def test_observe_keeps_the_uncertain_flag_once_raised():
    registry = SpeakerRegistry()
    registry.observe("SPEAKER_00", [1.0, 0.0], 2.0, uncertain=True)
    registry.observe("SPEAKER_00", [1.0, 0.0], 2.0)

    speaker = registry.get("SPEAKER_00")
    assert speaker is not None and speaker.uncertain is True


def test_merge_keeps_the_heavier_voiceprint_and_drops_the_other():
    registry = SpeakerRegistry(
        {
            "SPEAKER_00": GlobalSpeaker("SPEAKER_00", (1.0, 0.0), evidence_seconds=8.0),
            "SPEAKER_02": GlobalSpeaker("SPEAKER_02", (0.0, 1.0), evidence_seconds=2.0),
        }
    )

    survivor = registry.merge("SPEAKER_02", "SPEAKER_00")

    assert survivor is not None
    assert survivor.speaker_id == "SPEAKER_00"
    assert survivor.evidence_seconds == pytest.approx(10.0)
    assert "SPEAKER_02" not in registry
    assert survivor.voiceprint[0] > survivor.voiceprint[1]


def test_registry_round_trips_through_json(tmp_path: Path):
    registry = SpeakerRegistry()
    registry.observe("SPEAKER_00", [1.0, 0.0], 6.0, turns=3, scenes=2)
    registry.mint([0.0, 1.0], 4.0, turns=2, scenes=1, uncertain=True)

    path = registry.save(tmp_path / "speakers" / "registry.json")
    restored = SpeakerRegistry.load(path)

    assert set(restored) == {"SPEAKER_00", "SPEAKER_01"}
    assert restored.get("SPEAKER_01") is not None
    assert restored.get("SPEAKER_01").uncertain is True
    assert restored.as_dict() == registry.as_dict()


def test_a_missing_registry_is_empty_rather_than_an_error(tmp_path: Path):
    assert len(SpeakerRegistry.load(tmp_path / "absent.json")) == 0


def test_a_malformed_registry_is_reported(tmp_path: Path):
    target = tmp_path / "registry.json"
    target.write_text("{not json", encoding="utf-8")

    with pytest.raises(UnreadableRegistryError):
        SpeakerRegistry.load(target)


def test_a_registry_from_another_version_is_rejected(tmp_path: Path):
    target = tmp_path / "registry.json"
    target.write_text(
        json.dumps({"version": 99, "speakers": []}), encoding="utf-8"
    )

    with pytest.raises(UnreadableRegistryError):
        SpeakerRegistry.load(target)


def test_a_registry_with_a_broken_voiceprint_is_reported(tmp_path: Path):
    target = tmp_path / "registry.json"
    target.write_text(
        json.dumps(
            {
                "version": identity.REGISTRY_VERSION,
                "speakers": [{"speaker_id": "SPEAKER_00", "voiceprint": []}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(UnreadableRegistryError):
        SpeakerRegistry.load(target)


def test_the_persisted_form_carries_no_python_objects():
    registry = SpeakerRegistry()
    registry.observe("SPEAKER_00", [1.0, 0.0], 6.0, turns=3)

    payload = registry.as_dict()

    json.dumps(payload)  # must not raise
    assert payload["version"] == identity.REGISTRY_VERSION

# ---------------------------------------------------------------------------
# is this a character? evidence, not speaking time
# ---------------------------------------------------------------------------


def _assessment(**evidence):
    return assess_identity(IdentityEvidence("SPEAKER_00", **evidence))


def test_a_short_speaker_heard_twice_is_confirmed():
    """Two brief, cohesive, separated turns in different parts of the film."""

    result = _assessment(
        observations=2,
        turns=2,
        seconds=1.6,
        cohesion=0.85,
        margin=0.30,
        appearances=2,
        measured_fraction=1.0,
    )

    assert result.state == CONFIRMED


def test_a_longer_single_turn_is_not_confirmed_on_duration_alone():
    """The same total evidence, heard once, is not a character yet.

    This pair with the test above is the point of the change: 1.6s heard twice is
    confirmed while 2.5s heard once is not, so speaking time cannot be the criterion.
    """

    result = _assessment(
        observations=1,
        turns=1,
        seconds=2.5,
        cohesion=0.85,
        margin=0.30,
        appearances=1,
        measured_fraction=1.0,
    )

    assert result.state == PROVISIONAL
    assert any("heard once" in reason for reason in result.reasons)


def test_a_substantial_clean_single_turn_can_still_be_confirmed():
    """A real monologue is an identity, and the guard must not reject it."""

    result = _assessment(
        observations=1,
        turns=1,
        seconds=8.0,
        cohesion=0.90,
        margin=0.40,
        appearances=1,
        measured_fraction=1.0,
    )

    assert result.state == CONFIRMED


def test_an_emotionally_split_voice_stays_provisional():
    """A delivery change is not a new character: the voiceprints do not agree."""

    result = _assessment(
        observations=5,
        turns=5,
        seconds=6.0,
        cohesion=0.30,
        margin=0.60,
        appearances=2,
        measured_fraction=1.0,
    )

    assert result.state == PROVISIONAL
    assert any("not one voice" in reason for reason in result.reasons)


def test_a_noisy_fragment_is_not_a_character():
    result = _assessment(
        observations=1,
        turns=3,
        seconds=0.5,
        cohesion=0.40,
        margin=0.02,
        appearances=1,
        measured_fraction=0.33,
    )

    assert result.state == PROVISIONAL
    assert result.score < CONFIRM_SCORE


def test_a_voice_indistinguishable_from_a_neighbour_is_provisional():
    result = _assessment(
        observations=4,
        turns=4,
        seconds=9.0,
        cohesion=0.80,
        margin=0.0,
        appearances=3,
        measured_fraction=1.0,
    )

    assert result.state == PROVISIONAL
    assert any("not separated" in reason for reason in result.reasons)


def test_nothing_measurable_is_never_confirmed():
    result = _assessment(observations=0, turns=4, seconds=0.0, measured_fraction=0.0)

    assert result.state == PROVISIONAL
    assert any("nothing measurable" in reason for reason in result.reasons)


def test_the_same_evidence_scores_higher_as_observations_accumulate():
    """A provisional identity becomes confirmable by more evidence, not by more seconds."""

    early = _assessment(
        observations=1,
        turns=1,
        seconds=1.0,
        cohesion=0.80,
        margin=0.30,
        appearances=1,
        measured_fraction=1.0,
    )
    later = _assessment(
        observations=3,
        turns=3,
        seconds=3.0,
        cohesion=0.80,
        margin=0.30,
        appearances=2,
        measured_fraction=1.0,
    )

    assert early.state == PROVISIONAL
    assert later.state == CONFIRMED
    assert later.score > early.score


def test_duration_cannot_carry_a_score_on_its_own():
    """A long unmeasured block with no other signal stays provisional."""

    result = _assessment(
        observations=1,
        turns=1,
        seconds=30.0,
        cohesion=0.0,
        margin=-0.2,
        appearances=1,
        measured_fraction=0.1,
    )

    assert result.state == PROVISIONAL
