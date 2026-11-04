"""Ground-truth screenplay parsing, and character names taken from it.

A film arrives with two descriptions of the same dialogue: the audio, which the
diarizer and transcriber turn into ``SPEAKER_00`` at ``12.4s``, and the screenplay,
which says that ``JACK (V.O.)`` says it. The audio is the only one with timing, and
the script is the only one with names, so this module joins them::

    script.txt ---------------------------> parse_script() -> [ScriptLine, ...]
    [SpeakerSegment] + [TranscriptSegment] -> align_characters() -> [TranscriptSegment]

The transcribed lines are treated as a raw, sloppy timestamp feed. Their labels are
*overwritten* with the character the script gives for the same place on the timeline,
so every later stage - voice profiles, dialogue adaptation, the manifest - speaks about
Jack and Sarah rather than about ``SPEAKER_02``.

How the alignment works
-----------------------
The screenplay is flat text with no timestamps, so the only thing the two sides share
is order and wording. Each transcribed line is therefore matched against a bounded run
of script lines starting at the last match, scoring how much of the wording overlaps.
Three properties matter:

* **It only moves forwards.** A film is spoken in script order, and allowing a match
  to jump backwards would let one repeated "yes" capture half the screenplay.
* **It can skip.** Good lines the recogniser misheard, action beats, and lines the
  screenplay cut all have to be stepped over without breaking the rest of the film.
* **It never guesses a name it cannot support.** A line below the similarity floor
  keeps its diarization speaker's majority character, and a speaker whose every line
  is below the floor keeps its ``SPEAKER_XX`` label. A wrong name is worse than an
  honest label, because it is the key a voice sample directory is looked up by.

Format support
--------------
Screenplays are not a format, they are a convention. The parser accepts the two
conventions that are actually shipped as flat ``.txt``: a character cue on its own
line (``JACK (CONT'D)``, indented or not) followed by that character's dialogue, and a
cue with the dialogue on the same line (``JACK: Get in the car.``). Scene headings,
transitions and action are dropped. Parentheticals are *kept inside the line they
annotate*, because ``(sighs)`` at the start of a line is exactly the cue the adaptation
prompt turns into an inline ``[sigh]`` tag at that position.

Deliberate non-goals
--------------------
* **No timestamp recovery.** A script has none, so none is invented.
* **No fuzzy character merging.** Two spellings that differ are two characters; that is
  a property of the screenplay, not something to guess about.
* **No model, no network.** Parsing and alignment are deterministic text work.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from app.pipeline.diarization import SpeakerSegment
from app.pipeline.transcription import TranscriptSegment

#: A character cue that is only a name, optionally carrying an extension such as
#: ``(V.O.)``, ``(O.S.)`` or ``(CONT'D)``.
_CUE_ONLY = re.compile(
    r"^(?P<name>[A-Z][A-Z0-9 .'\-]*(?:\([A-Z0-9 .'\-]*\))?"
    r"(?:\s*/\s*[A-Z][A-Z0-9 .'\-]*)?)\s*$"
)

#: A cue with its dialogue on the same line, e.g. ``JACK: Get in.``.
_CUE_INLINE = re.compile(
    r"^(?P<name>[A-Z][A-Z0-9 .'\-]*(?:\([A-Z0-9 .'\-]*\))?)\s*:\s*(?P<text>.+)$"
)

#: A stage direction on its own line, e.g. ``(sighs)`` or ``(laughs softly)``.
_PARENTHETICAL = re.compile(r"^\(.*\)$")

#: Scene headings: ``INT. KITCHEN - DAY``, ``EXT. STREET - NIGHT``, ``I/E CAR``.
_SCENE_HEADING = re.compile(
    r"^(?:INT|EXT|EST|INT\./EXT|I/E|INT/EXT)[\s.\-/]", re.IGNORECASE
)

#: Transitions: ``CUT TO:``, ``FADE OUT.``, ``DISSOLVE TO:``.
_TRANSITION = re.compile(
    r"^(?:CUT TO|FADE (?:IN|OUT|TO BLACK)|DISSOLVE TO|SMASH CUT TO|"
    r"MATCH CUT TO|WIPE TO|BACK TO|CONTINUED)\b.*$",
    re.IGNORECASE,
)

#: Sentence punctuation a cue never ends with.
_SENTENCE_TERMINALS = (".", "!", "?", "…", "-", ":", ";")

#: Extension suffixes a character cue carries and the name does not.
_EXTENSIONS = ("O.S.", "V.O.", "O.C.", "CONT'D", "CONT'D.", "CONTINUED", "PRE-LAP")

#: Words that carry no identity and would make every line look like every other line.
_STOPWORDS = frozenset(
    """
    a an the and or but if of to in on at by for with from as is are was were be been
    being am do does did doing have has had having i you he she it we they me him her
    us them my your his its our their this that these those there here what who whom
    whose which when where why how not no yes so then than just very really can could
    will would shall should may might must about into over under again more most some
    any all both each few other such only own same too s t don now
    """.split()
)

#: Characters per line below which a line is not worth aligning: "No." against an
#: entire screenplay is noise, and the majority vote already covers such lines.
_MIN_ALIGN_CHARS = 3

#: How far past the last match a line may still find its script entry.
DEFAULT_SEARCH_AHEAD = 14

#: Similarity below which a match is treated as no match at all.
DEFAULT_MINIMUM_SIMILARITY = 0.42


def normalize_character_name(raw: str) -> str:
    """Return ``raw`` as a strict alphanumeric, title-cased character name.

    The name is used as a filesystem component (``voice_samples/<Character>/``), as a
    JSON key and as an identity across stages, so a single spelling has to be enforced.
    Screenplay extensions are removed first, then every character that is not a letter
    or a digit, and the words that remain are capitalised and joined::

        "JACK (V.O.)"          -> "Jack"
        "MARY-JANE (CONT'D)"   -> "MaryJane"
        "mr. smith"            -> "MrSmith"
        "D'ANGELO"             -> "DAngelo"

    Raises
    ------
    ValueError
        ``raw`` holds no letter or digit.
    """

    if not isinstance(raw, str):
        raise ValueError("a character name must be a string")

    text = raw.strip()
    # Drop extensions wherever they sit, then any trailing parenthetical note.
    for extension in _EXTENSIONS:
        text = re.sub(
            r"\(\s*" + re.escape(extension.rstrip(".")) + r"\.?\s*\)",
            " ",
            text,
            flags=re.IGNORECASE,
        )
    text = re.sub(r"\([^)]*\)", " ", text)
    words = [word for word in re.split(r"[^A-Za-z0-9]+", text) if word]
    if not words:
        raise ValueError(f"{raw!r} holds no letter or digit, so it cannot name a character")
    # An all-upper or all-lower word is a screenplay's spelling of a name, so its case
    # carries no information and is normalised. A deliberately mixed word ("iPhone")
    # is left alone - the screenplay is the authority on it.
    titled = [
        word.capitalize() if word.isupper() or word.islower() else word
        for word in words
    ]
    return "".join(titled)


@dataclass(frozen=True, slots=True)
class ScriptLine:
    """One complete line of dialogue, as the screenplay writes it.

    ``index`` is the line's position among the script's *dialogue* lines - the order
    the film speaks them in - so it is what the alignment moves through. ``scene`` is
    one-based and only used to describe a window to the model.
    """

    index: int
    character: str
    text: str
    scene: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError(f"index must be a non-negative integer, got {self.index!r}")
        if not isinstance(self.character, str) or not self.character.strip():
            raise ValueError("character must be a non-empty string")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a non-empty string")
        if isinstance(self.scene, bool) or not isinstance(self.scene, int) or self.scene < 1:
            raise ValueError(f"scene must be a positive integer, got {self.scene!r}")
        object.__setattr__(self, "text", self.text.strip())

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe view of the line."""

        return {
            "index": self.index,
            "character": self.character,
            "text": self.text,
            "scene": self.scene,
        }


@dataclass(frozen=True, slots=True)
class ParsedScript:
    """A screenplay reduced to its ordered dialogue lines."""

    lines: tuple[ScriptLine, ...]

    def __bool__(self) -> bool:
        return bool(self.lines)

    def __len__(self) -> int:
        return len(self.lines)

    @property
    def characters(self) -> tuple[str, ...]:
        """The distinct characters, in order of first appearance."""

        return tuple(dict.fromkeys(line.character for line in self.lines))

    def window(self, start: int, end: int) -> tuple[ScriptLine, ...]:
        """Return the script lines in ``[start, end)``, clamped to the screenplay."""

        first = max(0, int(start))
        last = min(len(self.lines), int(end))
        return self.lines[first:last]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe view of the screenplay."""

        return {
            "lines": len(self.lines),
            "characters": list(self.characters),
        }


def _is_speaking_cue(candidate: str) -> bool:
    """Return ``True`` when ``candidate`` reads as a character cue."""

    text = candidate.strip()
    if not 1 <= len(text) <= 64:
        return False
    if text.endswith(_SENTENCE_TERMINALS):
        # ``THE DOOR SLAMS OPEN.`` is action; a cue does not end a sentence.
        return False
    if text.upper() != text:
        return False
    if not any(character.isalpha() for character in text):
        return False
    # A cue is a name, occasionally with one extension: JACK, JACK (V.O.),
    # MARY-JANE, MAN #2, YOUNG JACK.
    without_extension = re.sub(r"\([^)]*\)", "", text).strip()
    if not without_extension:
        return False
    words = without_extension.split()
    if len(words) > 6:
        return False
    return all(re.fullmatch(r"[A-Z0-9.'#\-]+", word) for word in words)


def parse_script(text: str) -> ParsedScript:
    """Return the dialogue lines of a flat screenplay ``text``.

    The parser is intentionally forgiving about layout: indentation is ignored, and a
    cue is recognised by shape rather than by column. That is what lets the same code
    read a screenplay saved from a PDF, a Final Draft text export, or hand-typed.
    """

    if not isinstance(text, str):
        raise ValueError("script text must be a string")

    lines: list[ScriptLine] = []
    scene = 1
    character: str | None = None
    parts: list[str] = []
    #: A cue follows an action beat or a blank line. Inside an open dialogue block the
    #: blank line is what distinguishes the next ``SARAH`` cue from a shouted word the
    #: current character is saying.
    blank_before = True

    def flush() -> None:
        nonlocal parts, character
        body = " ".join(part for part in parts if part).strip()
        if character is not None and body:
            lines.append(
                ScriptLine(
                    index=len(lines),
                    character=character,
                    text=body,
                    scene=scene,
                )
            )
        parts = []

    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = raw.strip()

        if not stripped:
            flush()
            character = None
            blank_before = True
            continue

        if _SCENE_HEADING.match(stripped):
            flush()
            character = None
            blank_before = True
            scene += 1
            continue

        if _TRANSITION.match(stripped) and stripped.upper() == stripped:
            flush()
            character = None
            blank_before = True
            continue

        inline = _CUE_INLINE.match(stripped)
        if inline and _is_speaking_cue(inline.group("name")):
            flush()
            character = normalize_character_name(inline.group("name"))
            parts = [inline.group("text").strip()]
            blank_before = False
            continue

        if (
            (character is None or blank_before or len(stripped.split()) > 1)
            and _is_speaking_cue(stripped)
        ):
            flush()
            character = normalize_character_name(stripped)
            parts = []
            blank_before = False
            continue

        if character is not None:
            if _PARENTHETICAL.match(stripped):
                # A stage direction inside a line: keep it where it was written, so
                # the adaptation prompt can place its physical sound tag there.
                parts.append(stripped)
            else:
                parts.append(stripped)
            blank_before = False
            continue

        # Action or description outside any cue: not dialogue, and not the script's
        # business here.
        blank_before = False

    flush()
    return ParsedScript(lines=tuple(lines))


def load_script(path: str | Path) -> ParsedScript:
    """Read a flat ``.txt`` screenplay from ``path``.

    Raises
    ------
    FileNotFoundError
        No file exists at ``path``.
    ValueError
        The file is not valid UTF-8 text.
    """

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source} is not UTF-8 text: {exc}") from exc
    except FileNotFoundError:
        raise
    return parse_script(text)


@dataclass(frozen=True, slots=True)
class CharacterAssignment:
    """The transcribed lines, relabelled with the screenplay's character names."""

    segments: tuple[TranscriptSegment, ...]
    #: Diarization speaker id -> the character it was heard as most often.
    speakers: Mapping[str, str]
    #: Lines matched directly to a script entry.
    matched: int = 0
    #: Lines left on their speaker's majority character, because no script line fit.
    inherited: int = 0
    #: Lines left with their original ``SPEAKER_XX`` label, because nothing fit.
    unmatched: int = 0
    notes: tuple[str, ...] = ()

    @property
    def characters(self) -> tuple[str, ...]:
        """The distinct characters actually spoken in the audio, in order."""

        return tuple(dict.fromkeys(segment.speaker_id for segment in self.segments))

    def relabel_turns(self, turns: Iterable[SpeakerSegment]) -> tuple[SpeakerSegment, ...]:
        """Return the diarization turns with their speaker ids renamed.

        Voice profiles are keyed by speaker id, so the diarization output has to carry
        the same names the dialogue does. A turn whose cluster was never matched keeps
        its ``SPEAKER_XX`` label rather than being guessed at.
        """

        renamed: list[SpeakerSegment] = []
        for turn in turns:
            character = self.speakers.get(turn.speaker_id)
            if character is None or character == turn.speaker_id:
                renamed.append(turn)
                continue
            renamed.append(
                SpeakerSegment(
                    speaker_id=character,
                    start=turn.start,
                    end=turn.end,
                )
            )
        return tuple(renamed)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe view of the assignment."""

        return {
            "speakers": dict(self.speakers),
            "characters": list(self.characters),
            "matched": self.matched,
            "inherited": self.inherited,
            "unmatched": self.unmatched,
            "notes": list(self.notes),
        }


def _tokens(text: str) -> tuple[str, ...]:
    """Return the identity-bearing words of ``text``, lower-cased."""

    words = re.findall(r"[a-z0-9']+", text.lower())
    return tuple(word for word in words if word not in _STOPWORDS)


def text_similarity(left: str, right: str) -> float:
    """Return how strongly two lines look like the same utterance, in ``0..1``.

    Wording overlap is weighted above character overlap: a recogniser changes spelling
    and punctuation far more often than it changes which words were said.
    """

    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    left_set, right_set = set(left_tokens), set(right_tokens)
    jaccard = len(left_set & right_set) / len(left_set | right_set)
    sequence = SequenceMatcher(None, " ".join(left_tokens), " ".join(right_tokens)).ratio()
    return 0.6 * jaccard + 0.4 * sequence


def align_characters(
    segments: Iterable[TranscriptSegment],
    script: ParsedScript | Sequence[ScriptLine] | None,
    *,
    search_ahead: int = DEFAULT_SEARCH_AHEAD,
    minimum_similarity: float = DEFAULT_MINIMUM_SIMILARITY,
) -> CharacterAssignment:
    """Overwrite ``SPEAKER_XX`` labels with the screenplay's character names.

    Parameters
    ----------
    segments:
        Transcribed lines in chronological order - the raw timestamp feed.
    script:
        The parsed screenplay, or its lines. ``None`` or an empty screenplay returns
        the segments unchanged, which is what a run without a script should do.
    search_ahead:
        How many script lines past the last match one line may look. Wide enough to
        step over a screenplay-only scene, narrow enough that a repeated line cannot
        reach into the next act.
    minimum_similarity:
        The floor a match must clear to be used.

    Returns
    -------
    CharacterAssignment
        New segments with character names, the cluster-level mapping, and what could
        not be resolved.
    """

    ordered = tuple(segments)
    lines = (
        script.lines
        if isinstance(script, ParsedScript)
        else tuple(script or ())
    )
    if not ordered:
        return CharacterAssignment(segments=(), speakers={})
    if not lines:
        return CharacterAssignment(segments=ordered, speakers={})

    cursor = 0
    matched_characters: list[str | None] = []
    hits = 0
    for segment in ordered:
        text = segment.text.strip()
        if len(text) < _MIN_ALIGN_CHARS:
            matched_characters.append(None)
            continue

        best_index: int | None = None
        best_score = 0.0
        limit = min(len(lines), cursor + max(1, search_ahead))
        for position in range(cursor, limit):
            score = text_similarity(text, lines[position].text)
            if score > best_score:
                best_score = score
                best_index = position

        if best_index is None or best_score < minimum_similarity:
            matched_characters.append(None)
            continue

        matched_characters.append(lines[best_index].character)
        cursor = best_index + 1
        hits += 1

    # A cluster's majority character is what an unmatched line inherits. Ties go to
    # the character seen first, because Counter.most_common is insertion-ordered for
    # equal counts and the first sighting is the same choice on every run.
    votes: dict[str, Counter[str]] = {}
    for segment, character in zip(ordered, matched_characters):
        if character is None:
            continue
        votes.setdefault(segment.speaker_id, Counter())[character] += 1
    speakers = {
        speaker_id: counter.most_common(1)[0][0]
        for speaker_id, counter in votes.items()
    }

    relabelled: list[TranscriptSegment] = []
    inherited = 0
    unmatched = 0
    for segment, character in zip(ordered, matched_characters):
        if character is None:
            character = speakers.get(segment.speaker_id)
            if character is None:
                unmatched += 1
                character = segment.speaker_id
            else:
                inherited += 1
        relabelled.append(
            TranscriptSegment(
                speaker_id=character,
                start=segment.start,
                end=segment.end,
                text=segment.text,
            )
        )

    notes: list[str] = []
    merged = [
        speaker_id
        for speaker_id, counter in votes.items()
        if len(counter) > 1 and counter.most_common(1)[0][1] <= 1.0 * sum(counter.values()) / 2
    ]
    for speaker_id in merged:
        seen = ", ".join(name for name, _ in votes[speaker_id].most_common())
        notes.append(
            f"{speaker_id} was heard as more than one script character ({seen}); each "
            "line keeps the name the screenplay gives at its own position"
        )
    clustered = {
        character: sorted(
            speaker_id for speaker_id, chosen in speakers.items() if chosen == character
        )
        for character in set(speakers.values())
    }
    for character, cluster_ids in sorted(clustered.items()):
        if len(cluster_ids) > 1:
            notes.append(
                f"{character} was split across {len(cluster_ids)} diarization "
                f"cluster(s) ({', '.join(cluster_ids)}); all of them now speak as "
                f"{character}"
            )

    return CharacterAssignment(
        segments=tuple(relabelled),
        speakers=speakers,
        matched=hits,
        inherited=inherited,
        unmatched=unmatched,
        notes=tuple(notes),
    )


__all__ = [
    "CharacterAssignment",
    "DEFAULT_MINIMUM_SIMILARITY",
    "DEFAULT_SEARCH_AHEAD",
    "ParsedScript",
    "ScriptLine",
    "align_characters",
    "load_script",
    "normalize_character_name",
    "parse_script",
    "text_similarity",
]
