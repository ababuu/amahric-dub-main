"""Orchestration layer for the Amharic dubbing pipeline.

The pipeline is a linear sequence of independent stages. Each stage lives in its
own module and communicates only through files in ``WORK_DIR`` plus a small
manifest, so stages can be re-run, cached, and debugged in isolation.

Stage order
-----------
1. ``video``         - extract the audio from the source video; mux the dub back later.
2. ``separation``    - split the soundtrack into dialogue / music / effects stems.
3. ``diarization``   - identify who spoke when.
4. ``transcription`` - transcribe the dialogue stem with timestamps.
5. ``script_parse``  - overwrite the ``SPEAKER_XX`` labels with the screenplay's
   character names, then assemble the segments into complete sentences.
6. ``translation``   - adapt the sentences into Amharic on the local Qwen model served
   by vLLM, in overlapping concurrent windows.
7. ``voice_profiles``- pre-encode a per-character voice prompt from
   ``voice_samples/<Character>/``, falling back to the film's own cleanest reference.
8. ``tts``           - synthesize Amharic dialogue concurrently, with the line's
   emotion as a style instruction.
9. ``timing``        - place each line against the actor's active vocal zone, filling
   the rest of the window with silence.
10. ``mixing``       - combine dubbed dialogue with the music/effects stems.

Steps 1-10 are implemented and are chained by
:func:`app.pipeline.orchestrator.run_pipeline`, which is the single entry point for
a run:

    python -m app.pipeline.orchestrator data/input/movie.mp4

The deliverable is a dubbed MP4: the original picture, copied rather than
re-encoded, with the Amharic dialogue track in place of the English one.
"""

__all__: list[str] = []
