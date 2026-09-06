# Authoring a NEW STORY — characters, voice & score are all generated

The kit is a general **story to film engine**. A film is defined by exactly TWO
files: a **shots JSON** (cast + scenes) and a **config YAML** (voices, score
mapping, quality). Nothing about characters, voices, or music is hardcoded — the
confrontation film was just one instance used to validate the pipeline.

The engine (everything in `src/`, the models, the hardened post-processing) is
shared across all films. Only the two story files change.

---

## 1. Copy the templates
```bash
cp NEW_STORY_TEMPLATE/shots_mystory.json   shots_mystory.json
cp NEW_STORY_TEMPLATE/config_newstory.yaml  config_mystory.yaml
```
Point the config at your shots file:
```yaml
film_title: MY_STORY
shots_file: shots_mystory.json
```

## 2. Define your CHARACTERS (faces are auto-generated)
In `shots_mystory.json`:
- Add one entry per character under `character_design` (a `summary` + 5
  `forge_prompts`: face_front / face_threequarter / face_profile / body_full / body_coat).
- Add the same names under `characters` (a `refs` path + identity `tokens`).
In `config_mystory.yaml`:
- List the same names under `character_forge.targets`.

`forge_character.py` generates `refs/<name>/face` + `/body` automatically — **no
photos required**. To PIN real faces instead, drop images into
`refs/<name>/face/` before running (forge then reuses them and skips generation),
or point `GODSHAND_ANCHORS=/folder` laid out as `<name>/face/*.jpg`.

## 3. Cast the VOICES
In config `tts.voice_cast`, map each character to Kokoro voice ids
(`am_*` = male, `af_*` = female) and give a `performance` direction line.
`voice_pass.py` reads these at runtime — nothing is fixed.

## 4. Map the SCORE
In config `score.movements`, name each musical movement, give it an output
`file`, and list the shot ids it plays under. `score_gen.py` composes one
original instrumental cue per movement, then it is ducked under dialogue and
crossfaded automatically.

**Important nuance:** the per-movement *compose prompt* lives in
`src/score_gen.py` (the `MOOD` dict), keyed by movement **name**. Either:
- reuse an existing movement name to inherit its mood — available names include
  `prelude, search, confrontation, origin, the_deepening, lullaby, lady_in_red,
  hallucination_bed, the_breaking, the_crossing, machine_years, the_waking,
  the_reveal, the_mirror, the_amber_note, the_verdict, the_unreadable`; or
- add a new entry to `MOOD` for a custom movement name (a one-line dict edit).
An unknown movement name falls back to a generic `DEFAULT` cue.

## 5. Write the SHOTS
Each object in `shots[]` needs: a unique `id`, `dur` (seconds), `chars` (must
match character names), `talking`, `vo`, `camera`, `prompt`, `palette`,
`negative`, `tone`, and `movement` (must match a score movement name).

## 6. Render
```bash
# proof shots (first 4 of your story):
KIT_CONFIG=config_mystory.yaml bash GODSHAND_ONECLICK.sh

# choose specific proof shots:
KIT_CONFIG=config_mystory.yaml GODSHAND_SHOTS="scene1 scene3" bash GODSHAND_ONECLICK.sh

# full film:
KIT_CONFIG=config_mystory.yaml GODSHAND_FULL=1 bash GODSHAND_ONECLICK.sh
```
The one-click reads your config to discover the characters and proof shots
automatically, and it still enforces every no-compromise fix regardless of story.

---

## What is NOT per-story (the shared engine)
`render_motion.py`, `render_stills.py`, `forge_character.py`, `voice_pass.py`,
`score_gen.py`, `stitch.py`, the WAN model set, and the hardened
`postprocess.sh` are shared across all films. You never edit them to make a new
story (except optionally adding a `MOOD` entry for a brand-new movement name).
