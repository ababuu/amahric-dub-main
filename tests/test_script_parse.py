"""Tests for Stage 1: screenplay parsing and character-name resolution.

Parsing and alignment are deterministic text work, so everything here is real: no
model, no audio, no network.
"""

from __future__ import annotations

import pytest

from app.pipeline.diarization import SpeakerSegment
from app.pipeline.script_parse import (
    CharacterAssignment,
    ParsedScript,
    ScriptLine,
    align_characters,
    load_script,
    normalize_character_name,
    parse_script,
    text_similarity,
)
from app.pipeline.transcription import TranscriptSegment

SCREENPLAY = """\
INT. KITCHEN - NIGHT

Rain hammers the window.

JACK (V.O.)
I told you not to come here.
(sighs)
Now look at what happened.

MARY: Get in the car.

EXT. STREET - CONTINUOUS

CUT TO:

JACK
Not yet.

THE DOOR SLAMS OPEN.
"""


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("JACK", "Jack"),
        ("JACK (V.O.)", "Jack"),
        ("MARY-JANE (CONT'D)", "MaryJane"),
        ("mr. smith", "MrSmith"),
        ("D'ANGELO", "DAngelo"),
        ("MAN #2", "Man2"),
        ("iPhone", "iPhone"),
        ("  SARAH  ", "Sarah"),
    ],
)
def test_character_names_are_strictly_alphanumeric_and_title_cased(
    raw: str, expected: str
) -> None:
    assert normalize_character_name(raw) == expected


def test_a_name_without_a_letter_or_digit_is_rejected() -> None:
    with pytest.raises(ValueError, match="no letter or digit"):
        normalize_character_name("(...)")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_the_script_is_reduced_to_ordered_dialogue() -> None:
    script = parse_script(SCREENPLAY)

    assert [line.character for line in script.lines] == ["Jack", "Mary", "Jack"]
    assert [line.index for line in script.lines] == [0, 1, 2]


def test_scene_headings_are_dropped_and_counted() -> None:
    script = parse_script(SCREENPLAY)

    assert script.lines[0].scene == 2
    assert script.lines[-1].scene == 3


def test_a_stage_direction_is_kept_inside_its_line() -> None:
    """The parenthetical is the cue the adaptation prompt turns into an inline tag."""

    script = parse_script(SCREENPLAY)

    assert "(sighs)" in script.lines[0].text
    assert script.lines[0].text.startswith("I told you not to come here.")


def test_an_inline_cue_is_supported() -> None:
    script = parse_script("MARY: Get in the car.")

    assert script.lines[0].character == "Mary"
    assert script.lines[0].text == "Get in the car."


def test_action_lines_are_not_dialogue() -> None:
    script = parse_script(SCREENPLAY)

    assert all("Rain hammers" not in line.text for line in script.lines)
    assert all("DOOR" not in line.text for line in script.lines)


def test_continuation_lines_are_joined_within_one_cue() -> None:
    script = parse_script("JACK\nI told you\nnot to come here.\n")

    assert script.lines[0].text == "I told you not to come here."


def test_the_distinct_characters_keep_their_first_appearance() -> None:
    script = parse_script(SCREENPLAY)

    assert script.characters == ("Jack", "Mary")


def test_an_empty_script_parses_to_nothing() -> None:
    script = parse_script("\n\n   \n")

    assert script == ParsedScript(lines=())
    assert not script
    assert len(script) == 0


def test_a_script_line_validates_its_values() -> None:
    with pytest.raises(ValueError, match="character"):
        ScriptLine(index=0, character="", text="hello")
    with pytest.raises(ValueError, match="text"):
        ScriptLine(index=0, character="Jack", text="  ")
    with pytest.raises(ValueError, match="index"):
        ScriptLine(index=-1, character="Jack", text="hello")


def test_load_script_reads_utf8_text(tmp_path) -> None:
    path = tmp_path / "movie.txt"
    path.write_text("JACK\nሰላም\n", encoding="utf-8")

    script = load_script(path)

    assert script.lines[0].text == "ሰላም"


def test_a_missing_script_file_is_reported(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_script(tmp_path / "absent.txt")


def test_the_window_is_clamped_to_the_script() -> None:
    script = parse_script(SCREENPLAY)

    assert len(script.window(-5, 99)) == len(script.lines)
    assert script.window(1, 2) == (script.lines[1],)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def _segment(start: float, text: str, speaker: str = "SPEAKER_00") -> TranscriptSegment:
    return TranscriptSegment(speaker_id=speaker, start=start, end=start + 2.0, text=text)


def test_words_overlap_more_than_punctuation() -> None:
    assert text_similarity("I told you not to come here.", "I told you not to come here!") > 0.9
    assert text_similarity("The quick brown fox", "Something else entirely") < 0.2


def test_speaker_labels_are_overwritten_with_the_script_characters() -> None:
    script = parse_script(SCREENPLAY)
    segments = [
        _segment(0.0, "I told you not to come here."),
        _segment(3.0, "Now look at what happened."),
        _segment(6.0, "Get in the car.", speaker="SPEAKER_01"),
        _segment(9.0, "Not yet.", speaker="SPEAKER_01"),
    ]

    assignment = align_characters(segments, script)

    assert [line.speaker_id for line in assignment.segments] == [
        "Jack",
        "Jack",
        "Mary",
        "Jack",
    ]
    assert assignment.matched == 3
    # "Now look at what happened." is part of the same script line as the sentence
    # before it, so its word overlap is diluted by the (sighs) cue; it inherits the
    # cluster's character instead of matching directly.
    assert assignment.inherited == 1
    assert assignment.speakers == {"SPEAKER_00": "Jack", "SPEAKER_01": "Mary"}


def test_diarization_turns_are_relabelled_too() -> None:
    script = parse_script(SCREENPLAY)
    segments = [_segment(0.0, "I told you not to come here.")]
    turns = [
        SpeakerSegment(speaker_id="SPEAKER_00", start=0.0, end=2.0),
        SpeakerSegment(speaker_id="SPEAKER_07", start=5.0, end=7.0),
    ]

    assignment = align_characters(segments, script)
    relabelled = assignment.relabel_turns(turns)

    assert relabelled[0].speaker_id == "Jack"
    # A cluster the screenplay never matched keeps its honest label.
    assert relabelled[1].speaker_id == "SPEAKER_07"


def test_timing_and_text_are_never_changed() -> None:
    script = parse_script(SCREENPLAY)
    segments = [_segment(1.25, "Get in the car.")]

    assignment = align_characters(segments, script)

    line = assignment.segments[0]
    assert (line.start, line.end, line.text) == (1.25, 3.25, "Get in the car.")


def test_a_mismatched_line_inherits_its_speakers_character() -> None:
    script = parse_script(SCREENPLAY)
    segments = [
        _segment(0.0, "I told you not to come here."),
        _segment(3.0, "mumble mumble"),
    ]

    assignment = align_characters(segments, script)

    assert assignment.segments[1].speaker_id == "Jack"
    assert assignment.inherited == 1


def test_a_speaker_with_no_match_at_all_keeps_its_label() -> None:
    script = parse_script(SCREENPLAY)
    segments = [_segment(0.0, "completely unrelated words", speaker="SPEAKER_09")]

    assignment = align_characters(segments, script)

    assert assignment.segments[0].speaker_id == "SPEAKER_09"
    assert assignment.unmatched == 1
    assert assignment.speakers == {}


def test_no_script_leaves_everything_alone() -> None:
    segments = [_segment(0.0, "hello", speaker="SPEAKER_02")]

    assignment = align_characters(segments, None)

    assert assignment.segments == tuple(segments)
    assert assignment.speakers == {}


def test_an_empty_script_leaves_everything_alone() -> None:
    segments = [_segment(0.0, "hello")]

    assignment = align_characters(segments, ParsedScript(lines=()))

    assert assignment.segments == tuple(segments)


def test_no_segments_is_allowed() -> None:
    assert align_characters([], parse_script(SCREENPLAY)) == CharacterAssignment(
        segments=(), speakers={}
    )


def test_a_cluster_split_across_characters_is_reported() -> None:
    script = parse_script(SCREENPLAY)
    # One diarization cluster holding two characters: every line matches directly, so
    # each keeps its own name, and the run has to say the cluster was mixed.
    segments = [
        _segment(0.0, "I told you not to come here."),
        _segment(3.0, "Get in the car."),
    ]

    assignment = align_characters(segments, script)

    assert {line.speaker_id for line in assignment.segments} == {"Jack", "Mary"}
    assert any("more than one script character" in note for note in assignment.notes)


def test_the_assignment_is_json_safe() -> None:
    assignment = align_characters(
        [_segment(0.0, "Get in the car.")], parse_script(SCREENPLAY)
    )

    payload = assignment.as_dict()

    assert payload["characters"] == ["Mary"]
    assert payload["matched"] == 1
