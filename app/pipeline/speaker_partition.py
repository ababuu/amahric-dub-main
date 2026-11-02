"""Global re-partitioning of speaker evidence.

Diarization's own clustering is a partition of the film made in one pass, and the
mistake it makes is not "one cluster holds two voices" but "one voice is spread over
several clusters": a character who speaks briefly can be split across two neighbours
and appear in neither. No amount of splitting *within* a cluster can undo that, because
the fragments are in different clusters to begin with.

This module answers the question with the evidence instead of the clustering: given the
voiceprint of every reliable performance in the film, what is the partition of those
performances that the audio actually supports? It is deliberately free of pyannote and
torch - it takes vectors and gives back labels - so the decision can be tested without a
GPU, and it never changes a labeling unless the evidence survives three independent
checks:

* **fit** - the silhouette of the partition, in a cohort-centred space, where the
  threshold comes from the film's own same-voice and different-voice distances rather
  than from a constant chosen for a different embedding model;
* **size** - every cluster must stand on a minimum number of performances and seconds,
  because a character made of one 0.6 s fragment is not a character;
* **stability** - the same partition must reappear under bootstrap resampling. A
  partition that exists only for one particular sample of turns is noise.

A partition that fails is returned as a :class:`Candidate` with the numbers that
stopped it, so a run can report *why* it kept pyannote's answer instead of staying
silent.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

import numpy as np

from app.pipeline.speaker_identity import cosine_similarity as _cosine_similarity

#: A performance this long is clean enough to calibrate the film's own similarity
#: scale. Below it, an embedding carries more room than voice.
CLEAN_UNIT_SECONDS = 2.0

#: Fallbacks used when the film cannot calibrate itself (too few performances, or the
#: provisional labels are so contaminated that same-voice and different-voice distances
#: overlap). They are pyannote/WeSpeaker-shaped defaults, not tuned values.
FALLBACK_SAME_SIMILARITY = 0.55
FALLBACK_MIN_SILHOUETTE = 0.10

#: A partition must reproduce under bootstrap resampling at least this well (adjusted
#: Rand index between resamples) before it may replace pyannote's clustering.
MIN_STABILITY = 0.75

#: How much better a larger partition has to score before it is preferred over the
#: number of speakers the model already proposed. Inventing a character is expensive.
INVENTION_MARGIN = 0.03

#: How much better a *smaller* partition (a merge) has to score before it is preferred.
MERGE_MARGIN = 0.05


@dataclass(frozen=True, slots=True)
class Unit:
    """One reliable performance: what diarization proposed, and what it sounds like."""

    provisional_id: str
    seconds: float
    vector: tuple[float, ...]
    tag: int = 0

    def __post_init__(self) -> None:
        if self.seconds <= 0:
            raise ValueError(f"a unit needs positive seconds, got {self.seconds}")
        if not self.vector:
            raise ValueError("a unit needs a voiceprint")


@dataclass(frozen=True, slots=True)
class Calibration:
    """The film's own similarity scale, measured from its own embeddings."""

    #: Median similarity between two performances the model gave the same speaker.
    within: float
    #: Median similarity between two performances the model gave different speakers.
    between: float
    #: Above this two performances are the same voice: the midpoint of the two medians.
    same_similarity: float
    #: How far ahead of a rival an identity must be before it is reused.
    margin: float
    #: Silhouette a re-partition must reach before it is believed.
    min_silhouette: float
    #: ``False`` when the medians did not separate and the fallbacks were used.
    reliable: bool
    #: The measurement the numbers came from.
    units: int
    clean_units: int

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe view of the calibration."""

        return {
            "within_similarity": round(self.within, 4),
            "between_similarity": round(self.between, 4),
            "same_similarity": round(self.same_similarity, 4),
            "margin": round(self.margin, 4),
            "min_silhouette": round(self.min_silhouette, 4),
            "reliable": self.reliable,
            "units": self.units,
            "clean_units": self.clean_units,
        }


@dataclass(frozen=True, slots=True)
class Candidate:
    """One number of speakers tried, and whether the evidence supported it."""

    k: int
    silhouette: float
    stability: float
    units: int
    seconds: float
    smallest_cluster_units: int
    smallest_cluster_seconds: float
    accepted: bool
    reason: str
    #: ``True`` for the partition the run actually adopted.
    chosen: bool = False

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe view of the candidate."""

        return {
            "speakers": self.k,
            "silhouette": round(self.silhouette, 4),
            "stability": round(self.stability, 4),
            "units": self.units,
            "seconds": round(self.seconds, 3),
            "smallest_cluster_units": self.smallest_cluster_units,
            "smallest_cluster_seconds": round(self.smallest_cluster_seconds, 3),
            "accepted": self.accepted,
            "chosen": self.chosen,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class Partition:
    """The partition of the film's performances, chosen or refused."""

    #: One cluster index per unit, in the order the units were given.
    labels: tuple[int, ...]
    k: int
    silhouette: float
    stability: float
    #: ``False`` when nothing beat the provisional grouping; ``labels`` then mirror it.
    accepted: bool
    reason: str
    candidates: tuple[Candidate, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe view of the outcome."""

        return {
            "speakers": self.k,
            "silhouette": round(self.silhouette, 4),
            "stability": round(self.stability, 4),
            "accepted": self.accepted,
            "reason": self.reason,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


# ---------------------------------------------------------------------------
# numeric core
# ---------------------------------------------------------------------------


def _matrix(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    """Return ``vectors`` as a row-normalised float matrix."""

    matrix = np.asarray(
        [[float(value) for value in vector] for vector in vectors], dtype=np.float64
    )
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("a voiceprint matrix needs at least one non-empty vector")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def centre(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    """Remove the film's common direction from every voiceprint.

    Speaker embeddings share a large component that says more about the recording than
    about the speaker. Removing the cohort mean is what makes a comparison of two 0.8 s
    clips mean the same thing as a comparison of two 3 s ones.
    """

    matrix = _matrix(vectors)
    mean = matrix.mean(axis=0)
    centred = matrix - mean
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    # A vector that *is* the cohort mean carries no identity; keep its centered form
    # rather than dividing by zero, it will simply be far from everything.
    norms[norms < 1e-12] = 1.0
    return centred / norms


def kmeans(
    vectors: np.ndarray,
    k: int,
    *,
    iterations: int = 60,
    seed: int = 0,
    restarts: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Spherical k-means with deterministic k-means++ restarts.

    Deterministic because a speaker count that changes between two identical runs is
    not a measurement: the starts come from a seeded generator, and the best objective
    wins. Restarts matter with a handful of noisy short performances, where one start
    can put a whole voice in a neighbour's cluster.
    """

    count = vectors.shape[0]
    if k < 1 or k > count:
        raise ValueError(f"k must be between 1 and {count}, got {k}")
    if k == 1:
        centroid = vectors.mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroid = centroid / norm if norm > 0 else vectors[0]
        return np.zeros(count, dtype=int), centroid.reshape(1, -1)

    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for restart in range(max(1, restarts)):
        labels, centres = _kmeans_once(
            vectors, k, iterations=iterations, seed=seed + restart
        )
        score = float(np.sum(1.0 - np.max(vectors @ centres.T, axis=1)))
        if best is None or score < best[0] - 1e-12:
            best = (score, labels, centres)
    assert best is not None
    return best[1], best[2]


def _kmeans_once(
    vectors: np.ndarray, k: int, *, iterations: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """One spherical k-means pass, started by k-means++."""

    count = vectors.shape[0]
    rng = np.random.default_rng(seed)
    centres = [vectors[int(rng.integers(count))]]
    for _ in range(1, k):
        similarity = np.max(
            np.stack([vectors @ centre for centre in centres]), axis=0
        )
        weights = np.clip(1.0 - similarity, 0.0, None)
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            weights = np.ones(count)
        probabilities = weights / weights.sum()
        centres.append(vectors[int(rng.choice(count, p=probabilities))])
    matrix = np.asarray(centres, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    labels = np.full(count, -1, dtype=int)
    for _ in range(iterations):
        assigned = np.argmax(vectors @ matrix.T, axis=1)
        if np.array_equal(assigned, labels):
            break
        labels = assigned
        for index in range(k):
            members = vectors[labels == index]
            if members.shape[0] == 0:
                continue
            centroid = members.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                matrix[index] = centroid / norm
    return labels, matrix


def silhouette(vectors: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette of ``labels`` under cosine distance (singletons score 0)."""

    k = int(labels.max()) + 1 if labels.size else 0
    if k < 2 or labels.size < 3:
        return 0.0
    similarity = vectors @ vectors.T
    distance = 1.0 - similarity
    total = 0.0
    for position in range(labels.size):
        own = labels == labels[position]
        if int(own.sum()) < 2:
            continue
        inside = distance[position, own].sum() / (int(own.sum()) - 1)
        outside = min(
            distance[position, labels == other].mean()
            for other in range(k)
            if other != labels[position] and bool((labels == other).any())
        )
        worst = max(inside, outside)
        if worst > 0:
            total += (outside - inside) / worst
    return float(total / labels.size)


def adjusted_rand_index(left: Sequence[int], right: Sequence[int]) -> float:
    """Agreement between two labelings, corrected for chance."""

    count = len(left)
    if count < 2 or len(right) != count:
        return 1.0
    left_counts = Counter(left)
    right_counts = Counter(right)
    joint = Counter(zip(left, right))

    def pairs(value: int) -> float:
        return value * (value - 1) / 2.0

    index = sum(pairs(value) for value in joint.values())
    left_pairs = sum(pairs(value) for value in left_counts.values())
    right_pairs = sum(pairs(value) for value in right_counts.values())
    expected = left_pairs * right_pairs / pairs(count)
    maximum = 0.5 * (left_pairs + right_pairs)
    if maximum == expected:
        return 1.0
    return (index - expected) / (maximum - expected)


def stability(
    vectors: np.ndarray, k: int, *, repeats: int = 5, fraction: float = 0.8, seed: int = 0
) -> float:
    """How reproducibly ``k`` clusters appear under bootstrap resampling.

    Each resample leaves out a fifth of the performances, re-clusters, and labels the
    whole film by nearest centroid. The mean pairwise agreement between those labelings
    is the number reported: a partition that depends on two particular turns scores low.
    """

    if k < 2 or vectors.shape[0] < k + 2:
        return 1.0
    rng = np.random.default_rng(seed)
    size = max(k + 1, int(round(fraction * vectors.shape[0])))
    labelings: list[np.ndarray] = []
    for _ in range(repeats):
        sample = rng.choice(vectors.shape[0], size=size, replace=False)
        _, centres = kmeans(vectors[sample], k)
        labelings.append(np.argmax(vectors @ centres.T, axis=1))
    scores = [
        adjusted_rand_index(labelings[first].tolist(), labelings[second].tolist())
        for first in range(len(labelings))
        for second in range(first + 1, len(labelings))
    ]
    return float(np.mean(scores)) if scores else 1.0


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def identity_cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity of two voiceprints, accepting numpy rows as well."""

    return _cosine_similarity(tuple(left), tuple(right))


def calibrate(units: Sequence[Unit]) -> Calibration:
    """Measure the film's own same-voice and different-voice similarity.

    Long performances are used where they exist, because the provisional labels are
    exactly what is in doubt: a short turn may sit in the wrong cluster and would drag
    both medians towards each other.
    """

    clean = [unit for unit in units if unit.seconds >= CLEAN_UNIT_SECONDS]
    sample = clean if len(clean) >= 6 else list(units)
    same: list[float] = []
    different: list[float] = []
    for first in range(len(sample)):
        for second in range(first + 1, len(sample)):
            similarity = identity_cosine(sample[first].vector, sample[second].vector)
            if sample[first].provisional_id == sample[second].provisional_id:
                same.append(similarity)
            else:
                different.append(similarity)
    within = float(np.median(same)) if same else 0.0
    between = float(np.median(different)) if different else 0.0
    reliable = bool(same and different and within > between)
    if reliable:
        same_similarity = between + 0.5 * (within - between)
        spread = within - between
        margin = max(0.05, 0.25 * spread)
        min_silhouette = max(0.05, min(0.30, 0.5 * spread))
    else:
        within = within or FALLBACK_SAME_SIMILARITY
        between = between or max(0.0, within - 0.25)
        same_similarity = FALLBACK_SAME_SIMILARITY
        margin = 0.10
        min_silhouette = FALLBACK_MIN_SILHOUETTE
    return Calibration(
        within=within,
        between=between,
        same_similarity=same_similarity,
        margin=margin,
        min_silhouette=min_silhouette,
        reliable=reliable,
        units=len(units),
        clean_units=len(clean),
    )


# ---------------------------------------------------------------------------
# the decision
# ---------------------------------------------------------------------------


def _cluster_totals(
    units: Sequence[Unit], labels: Sequence[int], k: int
) -> tuple[list[int], list[float]]:
    counts = [0] * k
    seconds = [0.0] * k
    for unit, label in zip(units, labels):
        counts[label] += 1
        seconds[label] += unit.seconds
    return counts, seconds


def _provisional_labels(units: Sequence[Unit]) -> tuple[list[int], int]:
    order: dict[str, int] = {}
    labels: list[int] = []
    for unit in units:
        if unit.provisional_id not in order:
            order[unit.provisional_id] = len(order)
        labels.append(order[unit.provisional_id])
    return labels, len(order)


def repartition(
    units: Sequence[Unit],
    calibration: Calibration,
    *,
    max_extra: int = 3,
    min_cluster_units: int = 3,
    min_cluster_seconds: float = 4.0,
    repeats: int = 5,
) -> Partition:
    """Find the partition of ``units`` the evidence supports, or keep the provisional one.

    The search starts from the number of speakers the model proposed and tries up to
    ``max_extra`` more. It never tries fewer than that number minus one, because merging
    two proposed speakers is the failure this module exists to prevent - a merge is only
    accepted when the evidence for it is far stronger than the evidence for keeping them
    apart.
    """

    if len(units) < max(2, min_cluster_units):
        provisional, provisional_k = _provisional_labels(units)
        return Partition(
            labels=tuple(provisional),
            k=provisional_k,
            silhouette=0.0,
            stability=1.0,
            accepted=False,
            reason=(
                f"only {len(units)} reliable performance(s); "
                f"{min_cluster_units} are needed to judge a partition"
            ),
            candidates=(),
        )

    provisional, provisional_k = _provisional_labels(units)
    vectors = centre([unit.vector for unit in units])
    candidates: list[Candidate] = []
    accepted: list[tuple[Candidate, np.ndarray]] = []

    upper = min(provisional_k + max_extra, len(units))
    for k in range(max(1, provisional_k - 1), upper + 1):
        labels, _ = kmeans(vectors, k)
        counts, seconds = _cluster_totals(units, labels, k)
        # Plain Python numbers: these travel into a JSON manifest, and a numpy scalar
        # there is a run that finishes all its work and then fails to write its report.
        silhouette_score = float(silhouette(vectors, labels))
        stability_score = float(stability(vectors, k, repeats=repeats))
        smallest_units = int(min(counts))
        smallest_seconds = float(min(seconds))

        reasons: list[str] = []
        if smallest_units < min_cluster_units:
            reasons.append(
                f"a cluster has {smallest_units} performance(s), "
                f"{min_cluster_units} are needed"
            )
        if smallest_seconds < min_cluster_seconds:
            reasons.append(
                f"a cluster has {smallest_seconds:.1f}s of speech, "
                f"{min_cluster_seconds:.1f}s are needed"
            )
        if silhouette_score < calibration.min_silhouette:
            reasons.append(
                f"silhouette {silhouette_score:.2f} is below the "
                f"{calibration.min_silhouette:.2f} this film's own distances require"
            )
        if stability_score < MIN_STABILITY:
            reasons.append(
                f"the partition reproduces only {stability_score:.2f} of the time "
                f"({MIN_STABILITY:.2f} needed)"
            )

        candidate = Candidate(
            k=k,
            silhouette=silhouette_score,
            stability=stability_score,
            units=len(units),
            seconds=float(sum(seconds)),
            smallest_cluster_units=smallest_units,
            smallest_cluster_seconds=smallest_seconds,
            accepted=not reasons,
            reason="; ".join(reasons) if reasons else "accepted",
        )
        candidates.append(candidate)
        if candidate.accepted:
            accepted.append((candidate, labels))

    if not accepted:
        return Partition(
            labels=tuple(provisional),
            k=provisional_k,
            silhouette=0.0,
            stability=1.0,
            accepted=False,
            reason=(
                "no partition of the evidence survived the film's own thresholds; "
                "pyannote's clustering is kept"
            ),
            candidates=tuple(candidates),
        )

    best_candidate, best_labels = max(
        accepted, key=lambda item: (item[0].silhouette, -item[0].k)
    )

    # Do not invent a character for a marginal gain over the count the model proposed.
    preferred = [
        (candidate, labels)
        for candidate, labels in accepted
        if candidate.k <= provisional_k
    ]
    if best_candidate.k > provisional_k and preferred:
        near, near_labels = max(preferred, key=lambda item: item[0].silhouette)
        if near.silhouette >= best_candidate.silhouette - INVENTION_MARGIN:
            best_candidate, best_labels = near, near_labels

    # Do not merge two proposed speakers unless the evidence for one voice is clearly
    # stronger than the evidence for two.
    if best_candidate.k < provisional_k:
        same_k = [
            (candidate, labels)
            for candidate, labels in accepted
            if candidate.k == provisional_k
        ]
        if same_k:
            keep, keep_labels = same_k[0]
            if best_candidate.silhouette - keep.silhouette <= MERGE_MARGIN:
                best_candidate, best_labels = keep, keep_labels

    return Partition(
        labels=tuple(int(label) for label in best_labels),
        k=best_candidate.k,
        silhouette=best_candidate.silhouette,
        stability=best_candidate.stability,
        accepted=True,
        reason=(
            f"{best_candidate.k} speaker(s) fit the evidence at silhouette "
            f"{best_candidate.silhouette:.2f} and reproduce {best_candidate.stability:.2f}"
        ),
        candidates=tuple(
            replace(
                candidate,
                chosen=bool(
                    candidate.k == best_candidate.k
                    and candidate.silhouette == best_candidate.silhouette
                ),
            )
            for candidate in candidates
        ),
    )


def cluster_voiceprints(
    units: Sequence[Unit], labels: Sequence[int], k: int
) -> tuple[dict[int, tuple[float, ...]], dict[int, float], dict[int, int]]:
    """Return each cluster's duration-weighted voiceprint, seconds and unit count."""

    grouped: dict[int, list[Unit]] = {}
    for unit, label in zip(units, labels):
        grouped.setdefault(label, []).append(unit)
    voiceprints: dict[int, tuple[float, ...]] = {}
    seconds: dict[int, float] = {}
    counts: dict[int, int] = {}
    for label, members in grouped.items():
        width = len(members[0].vector)
        if any(len(member.vector) != width for member in members):
            continue
        total = sum(member.seconds for member in members)
        blended = [0.0] * width
        for member in members:
            for position, value in enumerate(member.vector):
                blended[position] += value * member.seconds / total
        norm = math.sqrt(sum(value * value for value in blended)) or 1.0
        voiceprints[label] = tuple(value / norm for value in blended)
        seconds[label] = total
        counts[label] = len(members)
    return voiceprints, seconds, counts


def merge_indistinguishable(
    voiceprints: Mapping[int, tuple[float, ...]],
    seconds: Mapping[int, float],
    *,
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Merge clusters that the film's own scale says are one voice.

    Returns ``(keep, drop, similarity)`` triples. The heavier cluster wins, so a
    character with six seconds of speech is not renamed after a two-second fragment.
    """

    order = sorted(voiceprints, key=lambda label: (-seconds.get(label, 0.0), label))
    aliases: dict[int, int] = {label: label for label in order}
    merges: list[tuple[int, int, float]] = []
    for position, first in enumerate(order):
        for second in order[position + 1 :]:
            keep_root = aliases[first]
            drop_root = aliases[second]
            if keep_root == drop_root:
                continue
            similarity = identity_cosine(voiceprints[keep_root], voiceprints[drop_root])
            if similarity >= threshold:
                aliases[drop_root] = keep_root
                merges.append((keep_root, drop_root, similarity))
    return merges


__all__ = [
    "CLEAN_UNIT_SECONDS",
    "Calibration",
    "Candidate",
    "FALLBACK_MIN_SILHOUETTE",
    "FALLBACK_SAME_SIMILARITY",
    "INVENTION_MARGIN",
    "MERGE_MARGIN",
    "MIN_STABILITY",
    "Partition",
    "Unit",
    "adjusted_rand_index",
    "calibrate",
    "centre",
    "cluster_voiceprints",
    "identity_cosine",
    "kmeans",
    "merge_indistinguishable",
    "repartition",
    "silhouette",
    "stability",
]
