# CLIPS MODE -- individual monetizable clips

The kit does two things now:
- **Full film**: `render_film.py` -> `stitch.py` -> one mastered film.
- **Clips**: `clips.py` packages each shot as a standalone, platform-ready, AI-labelled clip.

## Generate clips

```bash
# 1) render the shots you want (GPU) -- resumable, best-of-N
python3 src/render_film.py --only b05_doctor      # one shot
# or render the whole shot list, then package all of them

# 2) package clips (CPU, FREE -- run on Lightning CPU Studio / laptop)
python3 src/clips.py                 # every rendered shot
python3 src/clips.py --only b05_doctor
python3 src/clips.py --no-music      # voice/silent only
```

Each clip lands in `output/clips/<id>/`:

| File | What |
|------|------|
| `<id>_master.mp4`   | 16:9 mastered clip, music ducked under voice |
| `<id>_vertical.mp4` | 9:16 center-crop for Shorts / Reels / TikTok |
| `<id>_thumb.jpg`    | thumbnail frame |
| `<id>_manifest.json`| C2PA-style content-credentials (AI disclosure) |

Both MP4s embed AI-generated + creator metadata. Install **c2patool** to also sign
the clips with cryptographic content credentials (else metadata + sidecar only).

## Two things you MUST do for monetization

1. **Use an owned identity, not a real actor.** Real-person likeness = personality-rights
   liability in India (aggressive 2026 enforcement). Put an original synthetic character in
   `refs/original_identity/` and set `identity_override.enabled: true` in config.yaml.
2. **Disclose AI at upload.** The embedded metadata/C2PA is best practice, but you still must
   toggle each platform's AI switch (e.g. YouTube 'altered or synthetic content'). Disclosed
   original AI monetizes normally; undisclosed mass-produced AI risks the inauthentic-content
   policy.

## Economics (E2E spot A100 ~Rs.70/hr)

- Single-take clip: ~Rs.18-35   |   best-of-4 hero clip: ~Rs.70-140
- Rs.2,000 credit -> ~50-80 single-take clips (or ~18-20 polished hero clips)
- Packaging (clips.py) is CPU/free -- unlimited re-runs cost nothing.
