# Audio Director Notes — Dramatic Real Voice + Original Score

This upgraded kit is set so the finished film uses:

- **Original score only**: every movement is generated as a wordless cinematic score; no copied songs, no lyric beds.
- **Real dramatic emotion**: prompts ask for restrained, believable film-score dynamics instead of generic background music.
- **Character-specific voices**: Kunal, Gunja, Observer, and Chorus are cast to different TTS voices where available.
- **Cinematic voice processing**: lead voices are close-mic compressed and normalized; the chorus is distant, reverberant, and lower in the mix.
- **Film-style final mix**: the music swells in wordless beats, ducks under dialogue, and the final soundtrack is mastered for dramatic loudness.

For highest quality on Kaggle/Lightning, run the normal `filmforge_one.py` pipeline with GPU enabled. The score stage defaults to `facebook/musicgen-large` and automatically falls back to `facebook/musicgen-medium` if the large model is too heavy.

Useful quality controls:

```bash
# richer score if memory allows
export MG_MODEL=facebook/musicgen-large

# safer on small GPUs
export MG_MODEL=facebook/musicgen-medium
export SCORE_CHUNK_SEC=18

# full film render after checking cost projection
export CONFIRM_FULL_RUN=1
python3 src/filmforge_one.py
```
