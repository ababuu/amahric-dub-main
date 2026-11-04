"""Tests for Stage 5: local vLLM dialogue adaptation.

The model is never reached: every test injects a :class:`CompletionClient` fake, which
is also what lets the concurrency ceiling be measured rather than assumed.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.config import Settings
from app.pipeline.script_parse import ScriptLine, parse_script
from app.pipeline.transcription import DialogueSentence, TranscriptSegment
from app.pipeline.translation import (
    EMOTIONS,
    INLINE_SOUND_TAGS,
    SYSTEM_PROMPT,
    AdaptedDialogue,
    ApiClientError,
    ConfigurationError,
    DialogueIdError,
    InvalidModelOutputError,
    InvalidSegmentError,
    InvalidTranscriptSegmentsError,
    MalformedResponseError,
    Window,
    WindowRecord,
    adapt_dialogue,
    deduplicate_records,
    describe_backend,
    normalize_emotion,
    overlapping_windows,
    parse_response,
    sanitize_inline_tags,
)

FIDEL = "ጃክ አትጫወትብኝም።"


def _settings(root, **overrides) -> Settings:
    base = Settings(
        input_dir=root / "input",
        work_dir=root / "work",
        output_dir=root / "output",
        model_cache_dir=root / "models",
    )
    values = {field: getattr(base, field) for field in base.__dataclass_fields__}
    values.update(overrides)
    return Settings(**values)


def _segment(start: float, text: str, speaker: str = "SPEAKER_00") -> TranscriptSegment:
    return TranscriptSegment(speaker_id=speaker, start=start, end=start + 1.5, text=text)


def _dialogue(start: float = 0.0, **overrides) -> AdaptedDialogue:
    values = dict(
        speaker_id="Jack",
        start=start,
        end=start + 1.5,
        source_text="Get in the car.",
        amharic=FIDEL,
        emotion="angry",
        intensity=0.85,
        delivery="forceful",
        pause_before=0.0,
        pause_after=0.0,
    )
    values.update(overrides)
    return AdaptedDialogue(**values)


class FakeClient:
    """A completion client that answers the window it was given."""

    def __init__(self, *, emotion="neutral", text=FIDEL, character="Jack", delay=0.0):
        self.requests: list[dict] = []
        self.emotion = emotion
        self.text = text
        self.character = character
        self.delay = delay
        self.in_flight = 0
        self.max_in_flight = 0

    async def complete(self, *, system: str, user: str) -> str:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            payload, _ = json.JSONDecoder().raw_decode(user[user.index("{"):])
            self.requests.append(payload)
            lines = [
                {
                    "id": line["id"],
                    "character": self.character,
                    "start_timestamp": line["start_timestamp"],
                    "end_timestamp": line["end_timestamp"],
                    "overall_emotion": self.emotion,
                    "amharic_text_with_tags": self.text,
                }
                for line in payload["lines"]
            ]
            return json.dumps({"lines": lines}, ensure_ascii=False)
        finally:
            self.in_flight -= 1


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


def test_adapted_dialogue_keeps_its_values() -> None:
    line = _dialogue(start=2.0)

    assert line.speaker_id == "Jack"
    assert line.character == "Jack"
    assert line.source_text == "Get in the car."
    assert line.amharic == FIDEL
    assert line.duration == pytest.approx(1.5)


def test_adapted_dialogue_rejects_an_empty_amharic_line() -> None:
    with pytest.raises(InvalidSegmentError, match="amharic"):
        _dialogue(amharic="   ")


def test_adapted_dialogue_rejects_an_inverted_window() -> None:
    with pytest.raises(InvalidSegmentError, match="greater than start"):
        _dialogue(start=2.0, end=1.0)


def test_sound_tags_are_reported() -> None:
    line = _dialogue(amharic="ጃክ [sigh] ዝም በል። [breath]")

    assert line.sound_tags == ("sigh", "breath")


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


def test_windows_overlap_forward() -> None:
    windows = overlapping_windows(50, size=24, overlap=5)

    assert [(w.start, w.end) for w in windows] == [(0, 24), (19, 43), (38, 50)]


def test_a_film_shorter_than_one_window_is_one_window() -> None:
    assert [(w.start, w.end) for w in overlapping_windows(3, size=24, overlap=5)] == [
        (0, 3)
    ]


def test_no_lines_means_no_windows() -> None:
    assert overlapping_windows(0, size=24, overlap=5) == ()


def test_overlap_must_be_smaller_than_the_window() -> None:
    with pytest.raises(ConfigurationError, match="smaller than the window"):
        overlapping_windows(10, size=5, overlap=5)


def test_window_size_must_be_positive() -> None:
    with pytest.raises(ConfigurationError, match="window size"):
        overlapping_windows(10, size=0, overlap=0)


def test_edge_distance_measures_context() -> None:
    window = Window(index=0, start=10, end=20)

    assert window.edge_distance(10) == 0
    assert window.edge_distance(15) == 4
    assert window.edge_distance(19) == 0


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _records(count: int, windows, *, characters=None):
    records = []
    for window in windows:
        for position in range(window.start, window.end):
            speaker = (characters or {}).get((window.index, position), "Jack")
            records.append(
                WindowRecord(
                    position=position,
                    window=window,
                    dialogue=_dialogue(start=position * 2.0, speaker_id=speaker),
                )
            )
    return records


def test_deduplication_returns_one_line_per_position_in_order() -> None:
    windows = overlapping_windows(50, size=24, overlap=5)

    merged = deduplicate_records(_records(50, windows), total=50)

    assert len(merged) == 50
    assert [line.start for line in merged] == sorted(line.start for line in merged)


def test_the_answer_with_most_context_wins() -> None:
    windows = overlapping_windows(50, size=24, overlap=5)
    records = _records(50, windows, characters={(0, 22): "Edge", (1, 22): "Middle"})

    merged = deduplicate_records(records, total=50)

    # Position 22 is near the end of window 0 but well inside window 1.
    assert merged[22].speaker_id == "Middle"


def test_a_tie_goes_to_the_earlier_window() -> None:
    first = Window(index=0, start=0, end=4)
    second = Window(index=1, start=1, end=5)
    records = [
        WindowRecord(position, first, _dialogue(start=position * 2.0))
        for position in (0, 1, 2, 3)
    ]
    records[2] = WindowRecord(2, first, _dialogue(start=4.0, speaker_id="Earlier"))
    records.append(
        WindowRecord(2, second, _dialogue(start=4.0, speaker_id="Later"))
    )

    merged = deduplicate_records(records, total=4)

    # Both answers had the same amount of context, so the earlier window wins.
    assert merged[2].speaker_id == "Earlier"


def test_a_missing_line_fails_the_merge() -> None:
    window = Window(index=0, start=0, end=2)
    records = [WindowRecord(0, window, _dialogue(start=0.0))]

    with pytest.raises(DialogueIdError, match="never answered"):
        deduplicate_records(records, total=2)


def test_a_window_answering_the_same_line_twice_fails() -> None:
    window = Window(index=0, start=0, end=2)
    records = [
        WindowRecord(0, window, _dialogue(start=0.0)),
        WindowRecord(0, window, _dialogue(start=0.0)),
        WindowRecord(1, window, _dialogue(start=2.0)),
    ]

    with pytest.raises(DialogueIdError, match="twice"):
        deduplicate_records(records, total=2)


# ---------------------------------------------------------------------------
# Model output
# ---------------------------------------------------------------------------


def test_only_physical_inline_tags_survive() -> None:
    cleaned = sanitize_inline_tags(
        "ጃክ [sigh] ዝም በል። [ANGRY] [voice breaking] [gasp]"
    )

    assert cleaned == "ጃክ [sigh] ዝም በል። [gasp]"
    for tag in ("angry", "voice breaking"):
        assert tag not in cleaned


def test_an_allowed_tag_is_lower_cased() -> None:
    assert sanitize_inline_tags("ጃክ [SIGH]") == "ጃክ [sigh]"


@pytest.mark.parametrize("emotion", EMOTIONS)
def test_every_closed_vocabulary_emotion_is_accepted(emotion: str) -> None:
    assert normalize_emotion(f"  {emotion.upper()} ") == emotion


def test_a_descriptive_emotion_is_rejected() -> None:
    with pytest.raises(InvalidModelOutputError, match="closed"):
        normalize_emotion("quietly furious")


def test_parse_response_reads_the_lines_array() -> None:
    content = json.dumps(
        {
            "lines": [
                {
                    "id": "dialogue_000001",
                    "character": "Jack",
                    "start_timestamp": 0.0,
                    "end_timestamp": 1.5,
                    "overall_emotion": "sad",
                    "amharic_text_with_tags": FIDEL,
                }
            ]
        }
    )

    entries = parse_response(content, ["dialogue_000001"])

    assert list(entries) == ["dialogue_000001"]


def test_a_non_json_reply_is_reported() -> None:
    with pytest.raises(MalformedResponseError, match="not JSON"):
        parse_response("Sure! Here is the JSON:", ["dialogue_000001"])


def test_an_empty_reply_is_reported() -> None:
    with pytest.raises(MalformedResponseError, match="empty"):
        parse_response("   ", ["dialogue_000001"])


def test_a_reply_without_lines_is_reported() -> None:
    with pytest.raises(MalformedResponseError, match="'lines'"):
        parse_response('{"result": []}', ["dialogue_000001"])


def test_an_unrequested_line_is_reported() -> None:
    content = json.dumps({"lines": [{"id": "dialogue_999999"}]})

    with pytest.raises(DialogueIdError, match="not asked for"):
        parse_response(content, ["dialogue_000001"])


def test_a_duplicated_line_is_reported() -> None:
    content = json.dumps({"lines": [{"id": "dialogue_000001"}, {"id": "dialogue_000001"}]})

    with pytest.raises(DialogueIdError, match="twice"):
        parse_response(content, ["dialogue_000001"])


def test_a_missing_line_is_reported() -> None:
    with pytest.raises(DialogueIdError, match="did not answer"):
        parse_response('{"lines": []}', ["dialogue_000001"])


# ---------------------------------------------------------------------------
# The prompt contract
# ---------------------------------------------------------------------------


def test_the_prompt_names_the_required_schema() -> None:
    for key in (
        "character",
        "start_timestamp",
        "end_timestamp",
        "overall_emotion",
        "amharic_text_with_tags",
    ):
        assert key in SYSTEM_PROMPT


def test_the_prompt_states_the_closed_vocabularies() -> None:
    for emotion in EMOTIONS:
        assert f'"{emotion}"' in SYSTEM_PROMPT
    for tag in INLINE_SOUND_TAGS:
        assert f"[{tag}]" in SYSTEM_PROMPT


def test_the_prompt_forbids_screenplay_extensions() -> None:
    for extension in ("(O.S.)", "(V.O.)", "(CONT'D)"):
        assert extension in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# adapt_dialogue
# ---------------------------------------------------------------------------


def test_every_line_is_adapted_in_film_order(tmp_path) -> None:
    settings = _settings(tmp_path, translation_window_lines=2, translation_window_overlap=1)
    segments = [_segment(index * 2.0, f"line {index}") for index in range(5)]
    client = FakeClient()

    adapted = adapt_dialogue(segments, settings=settings, client=client)

    assert len(adapted) == 5
    assert [line.start for line in adapted] == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert all(line.emotion == "neutral" for line in adapted)


def test_overlapping_windows_share_context(tmp_path) -> None:
    settings = _settings(tmp_path, translation_window_lines=4, translation_window_overlap=2)
    segments = [_segment(index * 2.0, f"line {index}") for index in range(8)]
    client = FakeClient()

    adapt_dialogue(segments, settings=settings, client=client)

    assert len(client.requests) == 3
    first_ids = [line["id"] for line in client.requests[0]["lines"]]
    second_ids = [line["id"] for line in client.requests[1]["lines"]]
    assert set(first_ids) & set(second_ids)


def test_the_prompt_carries_the_syllable_budget(tmp_path) -> None:
    settings = _settings(tmp_path)
    client = FakeClient()

    adapt_dialogue([_segment(0.0, "hello there")], settings=settings, client=client)

    line = client.requests[0]["lines"][0]
    assert line["duration_seconds"] == pytest.approx(1.5)
    assert line["syllable_budget"] >= 1


def test_the_screenplay_is_sent_as_context(tmp_path) -> None:
    settings = _settings(tmp_path)
    script = parse_script("JACK\nGet in the car.\n\nSARAH\nNo.")
    client = FakeClient()

    adapt_dialogue(
        [_segment(0.0, "Get in the car.")],
        script=script.lines,
        settings=settings,
        client=client,
    )

    assert client.requests[0]["script"][0]["character"] == "Jack"
    assert client.requests[0]["script"][0]["text"] == "Get in the car."


def test_the_character_is_normalised_from_the_reply(tmp_path) -> None:
    settings = _settings(tmp_path)
    client = FakeClient(character="JACK (V.O.)")

    adapted = adapt_dialogue([_segment(0.0, "hello")], settings=settings, client=client)

    assert adapted[0].speaker_id == "Jack"


def test_a_character_the_screenplay_does_not_name_is_replaced(tmp_path) -> None:
    settings = _settings(tmp_path)
    script = parse_script("JACK\nHello there.")
    client = FakeClient(character="Imaginary")

    adapted = adapt_dialogue(
        [_segment(0.0, "Hello there.", speaker="SPEAKER_03")],
        script=script.lines,
        settings=settings,
        client=client,
    )

    assert adapted[0].speaker_id == "SPEAKER_03"


def test_the_engine_keeps_the_emotion_it_was_given(tmp_path) -> None:
    settings = _settings(tmp_path)
    client = FakeClient(emotion="fearful")

    adapted = adapt_dialogue([_segment(0.0, "hello")], settings=settings, client=client)

    assert adapted[0].emotion == "fearful"
    assert adapted[0].intensity == 0.70
    assert adapted[0].delivery == "trembling"


def test_a_forbidden_inline_tag_never_reaches_the_engine(tmp_path) -> None:
    settings = _settings(tmp_path)
    client = FakeClient(text="ጃክ [sigh] [angry] ዝም በል።")

    adapted = adapt_dialogue([_segment(0.0, "hello")], settings=settings, client=client)

    assert adapted[0].amharic == "ጃክ [sigh] ዝም በል።"


def test_a_moved_timestamp_fails_the_window(tmp_path) -> None:
    settings = _settings(tmp_path)

    class Shifting(FakeClient):
        async def complete(self, *, system: str, user: str) -> str:
            payload, _ = json.JSONDecoder().raw_decode(user[user.index("{"):])
            lines = [
                {
                    "id": line["id"],
                    "character": "Jack",
                    "start_timestamp": line["start_timestamp"] + 5.0,
                    "end_timestamp": line["end_timestamp"],
                    "overall_emotion": "neutral",
                    "amharic_text_with_tags": FIDEL,
                }
                for line in payload["lines"]
            ]
            return json.dumps({"lines": lines}, ensure_ascii=False)

    with pytest.raises(InvalidModelOutputError, match="application owns timing"):
        adapt_dialogue([_segment(0.0, "hello")], settings=settings, client=Shifting())


def test_a_rejected_window_is_retried_once(tmp_path) -> None:
    settings = _settings(tmp_path)

    class Flaky(FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.attempts = 0

        async def complete(self, *, system: str, user: str) -> str:
            self.attempts += 1
            if self.attempts == 1:
                return "not json"
            return await super().complete(system=system, user=user)

    client = Flaky()

    adapted = adapt_dialogue([_segment(0.0, "hello")], settings=settings, client=client)

    assert client.attempts == 2
    assert len(adapted) == 1


def test_concurrency_is_bounded_by_the_setting(tmp_path) -> None:
    settings = _settings(
        tmp_path,
        translation_window_lines=1,
        translation_window_overlap=0,
        translation_max_concurrency=2,
    )
    segments = [_segment(index * 2.0, f"line {index}") for index in range(8)]
    client = FakeClient(delay=0.02)

    adapt_dialogue(segments, settings=settings, client=client)

    assert client.max_in_flight <= 2
    assert client.max_in_flight > 1


def test_empty_dialogue_returns_nothing(tmp_path) -> None:
    assert adapt_dialogue([], settings=_settings(tmp_path), client=FakeClient()) == ()


def test_a_bad_window_overlap_is_rejected(tmp_path) -> None:
    settings = _settings(
        tmp_path, translation_window_lines=5, translation_window_overlap=5
    )

    with pytest.raises(ConfigurationError, match="smaller than"):
        adapt_dialogue([_segment(0.0, "hello")], settings=settings, client=FakeClient())


def test_a_bad_concurrency_is_rejected(tmp_path) -> None:
    settings = _settings(tmp_path, translation_max_concurrency=0)

    with pytest.raises(ConfigurationError, match="CONCURRENCY"):
        adapt_dialogue([_segment(0.0, "hello")], settings=settings, client=FakeClient())


def test_non_dialogue_input_is_rejected(tmp_path) -> None:
    with pytest.raises(InvalidTranscriptSegmentsError):
        adapt_dialogue(["hello"], settings=_settings(tmp_path), client=FakeClient())


def test_a_sentence_is_adapted_as_one_line(tmp_path) -> None:
    settings = _settings(tmp_path)
    sentence = DialogueSentence(
        speaker_id="Jack",
        start=0.0,
        end=3.0,
        text="Get in the car and drive.",
        parts=(
            TranscriptSegment("Jack", 0.0, 1.5, "Get in the car"),
            TranscriptSegment("Jack", 1.6, 3.0, "and drive."),
        ),
    )

    adapted = adapt_dialogue([sentence], settings=settings, client=FakeClient())

    assert len(adapted) == 1
    assert adapted[0].duration == pytest.approx(3.0)


def test_legacy_backend_arguments_are_rejected(tmp_path) -> None:
    """A stale caller must not believe a retired setting still applies."""

    with pytest.raises(ConfigurationError, match="no longer accepts"):
        adapt_dialogue(
            [_segment(0.0, "hello")],
            settings=_settings(tmp_path),
            client=FakeClient(),
            enforce_budget=True,
        )


def test_an_unreachable_server_is_named(tmp_path) -> None:
    settings = _settings(tmp_path, translation_base_url="http://127.0.0.1:1/v1")

    with pytest.raises((ApiClientError, Exception)):
        adapt_dialogue([_segment(0.0, "hello")], settings=settings)


def test_progress_is_reported_per_window(tmp_path) -> None:
    settings = _settings(tmp_path, translation_window_lines=1, translation_window_overlap=0)
    messages: list[str] = []

    adapt_dialogue(
        [_segment(0.0, "a"), _segment(2.0, "b")],
        settings=settings,
        client=FakeClient(),
        on_progress=messages.append,
    )

    assert len(messages) == 3  # one plan, one per window
    assert any("concurrent" in message for message in messages)


def test_describe_backend_names_the_model_and_the_batching(tmp_path) -> None:
    description = describe_backend(_settings(tmp_path))

    assert "Qwen" in description
    assert "JSON mode" in description
    assert "concurrent" in description


def test_script_lines_drive_the_character_names(tmp_path) -> None:
    settings = _settings(tmp_path)
    script = parse_script("SARAH\nWe have to go now.\n\nJACK\nNot yet.")

    adapted = adapt_dialogue(
        [_segment(0.0, "We have to go now."), _segment(2.0, "Not yet.")],
        script=script.lines,
        settings=settings,
        client=FakeClient(character="Sarah"),
    )

    assert adapted[0].speaker_id == "Sarah"


def test_a_script_line_validates_its_values() -> None:
    with pytest.raises(ValueError):
        ScriptLine(index=0, character="", text="hello")
