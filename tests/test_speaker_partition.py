"""Tests for :mod:`app.pipeline.speaker_partition`.

The module is the part of the identity fix that can be judged without a GPU: given
voiceprints and what pyannote proposed, does the evidence support a different partition?
The fixtures model the two failure modes seen in the film - one voice scattered over two
clusters, and one voice whose short turns are noisy - rather than clean textbook data.
"""

from __future__ import annotations

import json

import pytest

from app.pipeline import speaker_partition as partition
from app.pipeline.speaker_partition import (
    Calibration,
    Unit,
    adjusted_rand_index,
    calibrate,
    cluster_voiceprints,
    kmeans,
    merge_indistinguishable,
    repartition,
    silhouette,
    stability,
)


def _unit(provisional: str, seconds: float, base: int, noise: float, seed: int, tag: int = 0) -> Unit:
    """Build a voiceprint near axis ``base`` of a 16-dim space, plus noise."""

    import numpy as np

    rng = np.random.default_rng(seed)
    vector = np.zeros(16)
    vector[base] = 1.0
    vector += rng.normal(0.0, noise, size=16)
    return Unit(provisional_id=provisional, seconds=seconds, vector=tuple(vector), tag=tag)


def _film() -> list[Unit]:
    """Two clean voices plus a third scattered over both provisional clusters."""

    units: list[Unit] = []
    seed = 0
    for index in range(8):
        units.append(_unit("SPEAKER_00", 2.5, 0, 0.18, seed, tag=len(units)))
        seed += 1
    for index in range(8):
        units.append(_unit("SPEAKER_01", 2.5, 1, 0.18, seed, tag=len(units)))
        seed += 1
    # Character C: five short performances, three of them labelled SPEAKER_00 and two
    # SPEAKER_01 - exactly how a scattered voice arrives at this module.
    for index in range(5):
        provisional = "SPEAKER_00" if index % 2 == 0 else "SPEAKER_01"
        units.append(_unit(provisional, 0.9, 2, 0.28, seed, tag=len(units)))
        seed += 1
    return units


def test_the_scattered_voice_is_recovered_and_the_clean_voices_are_not_merged():
    units = _film()

    calibration = calibrate(units)
    result = repartition(units, calibration, max_extra=3)

    assert result.accepted is True
    assert result.k == 3
    # Every unit of the scattered character lands in one cluster, whichever cluster
    # pyannote had put it in.
    scattered = [label for unit, label in zip(units, result.labels) if unit.seconds < 1.5]
    assert len(set(scattered)) == 1
    # And the two clean voices stay apart.
    first = {label for unit, label in zip(units, result.labels) if unit.provisional_id == "SPEAKER_00" and unit.seconds > 1.5}
    second = {label for unit, label in zip(units, result.labels) if unit.provisional_id == "SPEAKER_01" and unit.seconds > 1.5}
    assert len(first) == 1 and len(second) == 1 and first != second


def test_a_third_voice_is_not_invented_when_the_evidence_is_two_voices():
    units = [
        _unit("SPEAKER_00", 2.5, 0, 0.18, seed)
        for seed in range(10)
    ] + [
        _unit("SPEAKER_01", 2.5, 1, 0.18, seed + 100)
        for seed in range(10)
    ]

    calibration = calibrate(units)
    result = repartition(units, calibration, max_extra=3)

    assert result.k == 2
    assert result.accepted is True


def test_one_noisy_voice_is_not_split_into_several_characters():
    """Short, noisy performances of one voice must not be turned into a cast."""

    units = [_unit("SPEAKER_00", 1.0, 0, 1.1, seed) for seed in range(12)]

    calibration = calibrate(units)
    result = repartition(units, calibration, max_extra=3)

    assert result.k == 1
    assert set(result.labels) == {0}


def test_a_partition_that_cannot_reproduce_is_refused():
    """Random labels must not survive bootstrap resampling."""

    import numpy as np

    rng = np.random.default_rng(7)
    units = [
        Unit(
            provisional_id="SPEAKER_00",
            seconds=2.0,
            vector=tuple(rng.normal(0.0, 1.0, size=16)),
            tag=index,
        )
        for index in range(24)
    ]
    calibration = calibrate(units)

    result = repartition(units, calibration, max_extra=3)

    assert result.accepted is False
    # The provisional grouping is kept, and every attempt says why it failed.
    assert result.k == 1
    assert all("silhouette" in candidate.reason or "reproduces" in candidate.reason
               for candidate in result.candidates)


def test_a_spurious_extra_speaker_is_not_adopted():
    """A third cluster that only splits an existing voice must lose to the simpler fit."""

    units = [
        _unit("SPEAKER_00", 2.5, 0, 0.18, seed) for seed in range(10)
    ] + [
        _unit("SPEAKER_01", 2.5, 1, 0.18, seed + 50) for seed in range(10)
    ] + [
        _unit("SPEAKER_00", 0.6, 2, 0.5, 900)
    ]

    calibration = calibrate(units)
    result = repartition(units, calibration, max_extra=3)

    assert result.k == 2
    chosen = [candidate for candidate in result.candidates if candidate.chosen]
    assert [candidate.k for candidate in chosen] == [2]
    assert all(not candidate.chosen for candidate in result.candidates if candidate.k != 2)


def test_calibration_reads_the_films_own_scale_and_flags_an_unreliable_one():
    units = _film()
    calibration = calibrate(units)

    assert calibration.reliable is True
    assert calibration.same_similarity > calibration.between
    assert calibration.margin > 0

    single = [_unit("SPEAKER_00", 2.0, 0, 0.18, seed) for seed in range(6)]
    unreliable = calibrate(single)
    assert unreliable.reliable is False
    assert unreliable.same_similarity == pytest.approx(partition.FALLBACK_SAME_SIMILARITY)


def test_kmeans_is_deterministic_and_separates_known_clusters():
    units = _film()
    import numpy as np

    vectors = partition.centre([unit.vector for unit in units])
    first, _ = kmeans(vectors, 3)
    second, _ = kmeans(vectors, 3)

    assert list(first) == list(second)
    assert len(set(first.tolist())) == 3


def test_silhouette_is_high_for_separated_clusters_and_zero_for_one():
    units = _film()
    vectors = partition.centre([unit.vector for unit in units])
    labels, _ = kmeans(vectors, 3)

    assert silhouette(vectors, labels) > 0.3
    assert silhouette(vectors, __import__("numpy").zeros(len(units), dtype=int)) == 0.0


def test_adjusted_rand_index_is_one_for_identical_labelings():
    assert adjusted_rand_index([0, 1, 1, 2], [0, 1, 1, 2]) == pytest.approx(1.0)
    assert adjusted_rand_index([0, 1, 1, 2], [2, 0, 0, 1]) == pytest.approx(1.0)


def test_stability_is_high_for_a_clear_partition():
    units = _film()
    vectors = partition.centre([unit.vector for unit in units])

    assert stability(vectors, 3) > 0.8


def test_cluster_voiceprints_are_duration_weighted():
    units = [
        Unit("SPEAKER_00", 6.0, (1.0, 0.0)),
        Unit("SPEAKER_00", 1.0, (0.0, 1.0)),
    ]
    voiceprints, seconds, counts = cluster_voiceprints(units, [0, 0], 1)

    assert seconds[0] == pytest.approx(7.0)
    assert counts[0] == 2
    assert voiceprints[0][0] > voiceprints[0][1]


def test_merge_indistinguishable_keeps_the_heavier_cluster():
    voiceprints = {
        0: (1.0, 0.0),
        1: (0.999, 0.02),
        2: (0.0, 1.0),
    }
    seconds = {0: 12.0, 1: 2.0, 2: 9.0}

    merges = merge_indistinguishable(voiceprints, seconds, threshold=0.9)

    assert merges == [(0, 1, pytest.approx(merges[0][2], abs=0.01))]
    assert merges[0][0] == 0


def test_repartition_refuses_when_there_is_too_little_evidence():
    units = [_unit("SPEAKER_00", 1.0, 0, 0.5, 1)]

    result = repartition(units, calibrate(units))

    assert result.accepted is False
    assert result.labels == (0,)
    assert "reliable performance" in result.reason

def test_the_partition_report_is_json_safe():
    """The QC block travels into a manifest; a numpy scalar there fails the write."""

    units = _film()
    result = repartition(units, calibrate(units), max_extra=3)

    payload = json.dumps(result.as_dict())

    assert "speakers" in payload
    calibration = json.dumps(calibrate(units).as_dict())
    assert "same_similarity" in calibration
