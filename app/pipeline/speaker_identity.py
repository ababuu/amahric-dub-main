"""Cross-scene speaker identity: voiceprints, matching, and a persistent registry.

Diarization answers "who spoke when" inside one scene. This module answers the
question a film asks across all of them: *is the voice in this scene the same person
that was already named* ``SPEAKER_00`` *two scenes ago?*

It is deliberately free of torch, pyannote, and numpy, so identity decisions can be
tested without a GPU or model weights:

* a :class:`GlobalSpeaker` is an id, a normalised voiceprint, and how much evidence
  that voiceprint stands on;
* :meth:`SpeakerRegistry.identify` turns one voiceprint into a decision - reuse an
  existing id, refuse to decide because two ids are too close to separate, or mint a
  new id for a voice the film has not used before;
* the registry round-trips to JSON, so a re-run of the same film keeps the identities
  an earlier run established.

The rule that makes this safe for dubbing: an *ambiguous* voiceprint is never folded
into an existing character. It gets its own id, marked uncertain, and the run reports
it. Two characters sharing one cloned voice is the failure this exists to prevent, so
the cheaper mistake is deliberately the one taken.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

#: The file a run writes its identities to, under the working directory, when the
#: caller does not name one.
DEFAULT_REGISTRY_FILENAME = "speaker_registry.json"

#: Cosine similarity a voiceprint needs to reuse an existing identity. pyannote's
#: speaker embeddings put the same voice well above this and different voices below
#: it; the value is a floor, not a target, because the margin test below does the
#: discriminating.
DEFAULT_MATCH_MIN_SIMILARITY = 0.55

#: How far the best candidate must beat the runner-up before the registry will reuse
#: an id. Two characters this close cannot be told apart by an embedding, so the
#: registry refuses to pick one and a new identity is minted instead.
DEFAULT_MATCH_MIN_MARGIN = 0.08

#: Two identities this similar are almost certainly one person discovered twice (a
#: character seen only in two short scenes, say). Above this they are merged.
DEFAULT_MERGE_MIN_SIMILARITY = 0.90

#: Between this and the merge threshold a pair is reported as a suspected duplicate
#: but left alone: a wrong merge is audible, a wrong duplicate is not.
DEFAULT_SUSPECT_SIMILARITY = 0.75

#: Bumped when the persisted shape changes so an old file is rejected, not misread.
REGISTRY_VERSION = 1


class IdentityError(RuntimeError):
    """Base class for every error raised by this module."""


class InvalidVoiceprintError(IdentityError, ValueError):
    """A voiceprint is empty, non-finite, or all zeros."""


class UnreadableRegistryError(IdentityError):
    """A persisted registry cannot be read back as identities."""


def normalise(vector: Sequence[float], *, name: str = "voiceprint") -> tuple[float, ...]:
    """Return ``vector`` scaled to unit length.

    Raises:
        InvalidVoiceprintError: when it is empty, contains a non-finite value, or is
            all zeros - none of which can be compared to anything.
    """

    try:
        values = tuple(float(value) for value in vector)
    except (TypeError, ValueError) as exc:
        raise InvalidVoiceprintError(f"{name} is not a sequence of numbers: {exc}") from exc
    if not values:
        raise InvalidVoiceprintError(f"{name} must not be empty")
    if any(not math.isfinite(value) for value in values):
        raise InvalidVoiceprintError(f"{name} contains a non-finite value")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0:
        raise InvalidVoiceprintError(f"{name} is all zeros and carries no identity")
    return tuple(value / norm for value in values)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity of two vectors, or ``-1`` when they cannot be compared.

    Different widths mean different embedding models, and a stale registry entry must
    never match a current voiceprint by accident, so they score below any real
    similarity rather than raising.
    """

    if len(left) != len(right) or not left:
        return -1.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for first, second in zip(left, right):
        first_value = float(first)
        second_value = float(second)
        if not (math.isfinite(first_value) and math.isfinite(second_value)):
            return -1.0
        dot += first_value * second_value
        left_norm += first_value * first_value
        right_norm += second_value * second_value
    if left_norm <= 0.0 or right_norm <= 0.0:
        return -1.0
    similarity = dot / math.sqrt(left_norm * right_norm)
    return max(-1.0, min(1.0, similarity))


def blend(
    centroid: Sequence[float], weight: float, vector: Sequence[float], seconds: float
) -> tuple[float, ...]:
    """Return the duration-weighted mean of an identity's voiceprint and new evidence.

    Weighting by seconds rather than by turn count keeps a character who speaks twice
    for six seconds from being outvoted by one who says "yeah" ten times.
    """

    if weight <= 0.0:
        return normalise(vector)
    if seconds <= 0.0:
        return normalise(centroid)
    total = weight + seconds
    blended = [
        (float(previous) * weight + float(new) * seconds) / total
        for previous, new in zip(centroid, vector)
    ]
    try:
        return normalise(blended)
    except InvalidVoiceprintError:
        # Opposite directions cancelled out; keep the identity we already had.
        return normalise(centroid)


@dataclass(frozen=True, slots=True)
class GlobalSpeaker:
    """One character's identity across the whole film."""

    speaker_id: str
    voiceprint: tuple[float, ...]
    #: Seconds of speech the voiceprint stands on.
    evidence_seconds: float = 0.0
    turns: int = 0
    scenes: int = 0
    #: ``True`` when this identity was minted without being separable from the
    #: others: it is a distinct voice, but which character it is remains open.
    uncertain: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.speaker_id, str) or not self.speaker_id.strip():
            raise InvalidVoiceprintError("speaker_id must be a non-empty string")
        if self.evidence_seconds < 0 or self.turns < 0 or self.scenes < 0:
            raise InvalidVoiceprintError(
                "evidence_seconds, turns and scenes must not be negative"
            )
        object.__setattr__(self, "voiceprint", normalise(self.voiceprint))

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe view of the identity."""

        return {
            "speaker_id": self.speaker_id,
            "voiceprint": [round(value, 6) for value in self.voiceprint],
            "evidence_seconds": round(self.evidence_seconds, 3),
            "turns": self.turns,
            "scenes": self.scenes,
            "uncertain": self.uncertain,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "GlobalSpeaker":
        """Rebuild an identity from :meth:`as_dict` output."""

        try:
            speaker_id = str(payload["speaker_id"])
            voiceprint = tuple(float(value) for value in payload["voiceprint"])  # type: ignore[union-attr]
        except (KeyError, TypeError, ValueError) as exc:
            raise UnreadableRegistryError(
                f"a persisted speaker is missing its id or voiceprint: {exc}"
            ) from exc
        return cls(
            speaker_id=speaker_id,
            voiceprint=voiceprint,
            evidence_seconds=float(payload.get("evidence_seconds", 0.0)),  # type: ignore[arg-type]
            turns=int(payload.get("turns", 0)),  # type: ignore[arg-type]
            scenes=int(payload.get("scenes", 0)),  # type: ignore[arg-type]
            uncertain=bool(payload.get("uncertain", False)),
        )


#: Outcomes of :meth:`SpeakerRegistry.identify`.
MATCHED = "matched"
AMBIGUOUS = "ambiguous"
UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class IdentityDecision:
    """What the registry concluded about one voiceprint."""

    outcome: str
    speaker_id: str | None
    similarity: float
    runner_up: float
    margin: float

    @property
    def matched(self) -> bool:
        """``True`` when the voiceprint may reuse ``speaker_id``."""

        return self.outcome == MATCHED

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe view of the decision."""

        return {
            "outcome": self.outcome,
            "speaker_id": self.speaker_id,
            "similarity": round(self.similarity, 4),
            "runner_up": round(self.runner_up, 4),
            "margin": round(self.margin, 4),
        }


class SpeakerRegistry:
    """The film's named voices, and the rules for naming a new one."""

    def __init__(self, speakers: Mapping[str, GlobalSpeaker] | None = None) -> None:
        self._speakers: dict[str, GlobalSpeaker] = dict(speakers or {})

    def __len__(self) -> int:
        return len(self._speakers)

    def __contains__(self, speaker_id: object) -> bool:
        return speaker_id in self._speakers

    def __iter__(self):
        return iter(sorted(self._speakers))

    @property
    def speakers(self) -> dict[str, GlobalSpeaker]:
        """A copy of the identities, keyed by speaker id."""

        return dict(self._speakers)

    def get(self, speaker_id: str) -> GlobalSpeaker | None:
        """Return the identity stored under ``speaker_id``, if any."""

        return self._speakers.get(speaker_id)

    def ranks(self, vector: Sequence[float]) -> list[tuple[str, float]]:
        """Return every identity ordered from most to least similar to ``vector``."""

        scored = [
            (speaker_id, cosine_similarity(vector, identity.voiceprint))
            for speaker_id, identity in self._speakers.items()
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored

    def identify(
        self,
        vector: Sequence[float],
        *,
        min_similarity: float = DEFAULT_MATCH_MIN_SIMILARITY,
        min_margin: float = DEFAULT_MATCH_MIN_MARGIN,
    ) -> IdentityDecision:
        """Decide whether ``vector`` belongs to an identity already in the registry.

        ``matched`` when one identity is both similar enough and clearly ahead of the
        next; ``ambiguous`` when it is similar enough but not separable - the caller
        must mint a new identity rather than fold two characters together;
        ``unknown`` when nothing is close.
        """

        scored = self.ranks(vector)
        if not scored:
            return IdentityDecision(UNKNOWN, None, -1.0, -1.0, 0.0)
        best_id, best = scored[0]
        runner_up = scored[1][1] if len(scored) > 1 else -1.0
        margin = best - runner_up if len(scored) > 1 else 1.0
        if best < min_similarity:
            return IdentityDecision(UNKNOWN, None, best, runner_up, margin)
        if len(scored) > 1 and margin < min_margin:
            return IdentityDecision(AMBIGUOUS, best_id, best, runner_up, margin)
        return IdentityDecision(MATCHED, best_id, best, runner_up, margin)

    def next_id(self, prefix: str = "SPEAKER") -> str:
        """Return the first ``PREFIX_NN`` label not already in use."""

        highest = -1
        for speaker_id in self._speakers:
            head, _, tail = speaker_id.rpartition("_")
            if head == prefix and tail.isdigit():
                highest = max(highest, int(tail))
        candidate = highest + 1
        while f"{prefix}_{candidate:02d}" in self._speakers:
            candidate += 1
        return f"{prefix}_{candidate:02d}"

    def observe(
        self,
        speaker_id: str,
        vector: Sequence[float],
        seconds: float,
        *,
        turns: int = 0,
        scenes: int = 0,
        uncertain: bool | None = None,
    ) -> GlobalSpeaker:
        """Add evidence for ``speaker_id``, creating the identity if it is new."""

        current = self._speakers.get(speaker_id)
        if current is None:
            identity = GlobalSpeaker(
                speaker_id=speaker_id,
                voiceprint=normalise(vector),
                evidence_seconds=max(0.0, float(seconds)),
                turns=max(0, int(turns)),
                scenes=max(0, int(scenes)),
                uncertain=bool(uncertain),
            )
            self._speakers[speaker_id] = identity
            return identity
        identity = GlobalSpeaker(
            speaker_id=speaker_id,
            voiceprint=blend(
                current.voiceprint, current.evidence_seconds, vector, float(seconds)
            ),
            evidence_seconds=current.evidence_seconds + max(0.0, float(seconds)),
            turns=current.turns + max(0, int(turns)),
            scenes=current.scenes + max(0, int(scenes)),
            uncertain=(
                current.uncertain if uncertain is None else (current.uncertain or uncertain)
            ),
        )
        self._speakers[speaker_id] = identity
        return identity

    def mint(
        self,
        vector: Sequence[float],
        seconds: float,
        *,
        turns: int = 0,
        scenes: int = 0,
        uncertain: bool = False,
        prefix: str = "SPEAKER",
    ) -> GlobalSpeaker:
        """Create a new identity for a voice the registry cannot place."""

        speaker_id = self.next_id(prefix)
        return self.observe(
            speaker_id, vector, seconds, turns=turns, scenes=scenes, uncertain=uncertain
        )

    def rename(self, old: str, new: str) -> None:
        """Move the identity stored under ``old`` to ``new`` (used by merges)."""

        if old not in self._speakers:
            return
        identity = self._speakers.pop(old)
        self._speakers[new] = GlobalSpeaker(
            speaker_id=new,
            voiceprint=identity.voiceprint,
            evidence_seconds=identity.evidence_seconds,
            turns=identity.turns,
            scenes=identity.scenes,
            uncertain=identity.uncertain,
        )

    def merge(self, source: str, target: str) -> GlobalSpeaker | None:
        """Fold ``source`` into ``target``, keeping the heavier voiceprint.

        Returns the surviving identity, or ``None`` when either is missing.
        """

        keep = self._speakers.get(target)
        drop = self._speakers.get(source)
        if keep is None or drop is None or source == target:
            return keep
        survivor = GlobalSpeaker(
            speaker_id=target,
            voiceprint=blend(
                keep.voiceprint,
                keep.evidence_seconds,
                drop.voiceprint,
                drop.evidence_seconds,
            ),
            evidence_seconds=keep.evidence_seconds + drop.evidence_seconds,
            turns=keep.turns + drop.turns,
            scenes=keep.scenes + drop.scenes,
            uncertain=keep.uncertain or drop.uncertain,
        )
        self._speakers[target] = survivor
        del self._speakers[source]
        return survivor

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe view of the whole registry."""

        return {
            "version": REGISTRY_VERSION,
            "speakers": [
                self._speakers[speaker_id].as_dict()
                for speaker_id in sorted(self._speakers)
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SpeakerRegistry":
        """Rebuild a registry from :meth:`as_dict` output."""

        version = payload.get("version", REGISTRY_VERSION)
        if version != REGISTRY_VERSION:
            raise UnreadableRegistryError(
                f"speaker registry version {version!r} is not supported "
                f"(expected {REGISTRY_VERSION})"
            )
        entries = payload.get("speakers", [])
        if not isinstance(entries, Iterable):
            raise UnreadableRegistryError("speaker registry has no 'speakers' list")
        speakers: dict[str, GlobalSpeaker] = {}
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise UnreadableRegistryError(
                    f"a persisted speaker is not an object: {entry!r}"
                )
            identity = GlobalSpeaker.from_dict(entry)
            speakers[identity.speaker_id] = identity
        return cls(speakers)

    @classmethod
    def load(cls, path: str | Path) -> "SpeakerRegistry":
        """Read a registry from ``path``; a missing file is an empty registry."""

        target = Path(path)
        if not target.exists():
            return cls()
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise UnreadableRegistryError(
                f"could not read the speaker registry {target}: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise UnreadableRegistryError(
                f"the speaker registry {target} is not a JSON object"
            )
        try:
            return cls.from_dict(payload)
        except InvalidVoiceprintError as exc:
            raise UnreadableRegistryError(
                f"the speaker registry {target} holds an unusable voiceprint: {exc}"
            ) from exc

    def save(self, path: str | Path) -> Path:
        """Write the registry to ``path`` atomically, and return the path."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
        return target


__all__ = [
    "AMBIGUOUS",
    "DEFAULT_MATCH_MIN_MARGIN",
    "DEFAULT_MATCH_MIN_SIMILARITY",
    "DEFAULT_MERGE_MIN_SIMILARITY",
    "DEFAULT_REGISTRY_FILENAME",
    "DEFAULT_SUSPECT_SIMILARITY",
    "GlobalSpeaker",
    "IdentityDecision",
    "IdentityError",
    "InvalidVoiceprintError",
    "MATCHED",
    "REGISTRY_VERSION",
    "SpeakerRegistry",
    "UNKNOWN",
    "UnreadableRegistryError",
    "blend",
    "cosine_similarity",
    "normalise",
]
