#!/usr/bin/env python3
"""Auto-compose the original cinematic score for THE MOMENT — v4.

Design goal: every movement should feel like a real dramatic film score, not a
placeholder bed. The score is **original, instrumental, wordless**, emotionally
specific, and written to support the voices.

v4 enhancements (Kaggle-friendly):
  * Multiband mastering via ffmpeg (no extra Python deps)
  * Stereo widening on the master bus
  * 48kHz output (professional broadcast standard)
  * Better per-movement emotional prompts
  * Graceful fallback: if MusicGen doesn't fit in VRAM alongside Wan2.2,
    the score can be generated in a separate session or on CPU (sine fallback)

Usage:
  python3 src/score_gen.py --root .             # real generation (needs GPU)
  python3 src/score_gen.py --root . --dry-run   # plumbing test only

Env:
  MG_MODEL          default facebook/musicgen-medium; falls back to small
  SCORE_CHUNK_SEC   default 22 (keeps T4 memory safer)
"""
import os, sys, json, math, wave, subprocess, argparse

# Per-movement direction. Each prompt asks for a believable recorded score:
# real instruments, human dynamics, emotional restraint, no vocals, no lyrics.
MOOD = {
    "prelude":            "original quiet aching memoir prelude score, a lone felt piano and distant cello over soft sustained strings, fragile loneliness and the ache of a love never spoken, slowly gathering tension and a faint low heartbeat of percussion as it builds toward a reckoning, restrained and deeply human, believable real orchestral performance, cinematic, instrumental only, no vocals, no lyrics",
    "search":             "original search-and-dread score that begins as a lone violin thread hunting in the black and SWELLS into full fearful orchestra as she nears him: pounding timpani and taiko-order drums, deep full-bass brass and surging low strings, the solo violin climbing into a fearful shriek, a relentlessly rising dread that never seems to stop climbing, cold electrical spark and welding sizzle woven into the percussion, cresting to a terrible peak of fear and then suspending to a single held breath the instant she finds him, believable real orchestral performance, modern blockbuster gravitas, wholly original and transformative and NOT a copy of any existing cue, instrumental only, no vocals, no lyrics",
    "confrontation":      "original dark hybrid-orchestral confrontation score in the spirit of a Terminator-order relentless propulsive machine ostinato fused with Hans Zimmer Dark-Knight-order rising two-note brass dread, a soaring aching solo violin that lands hard on its first note and then keeps swelling and swelling without ever resolving, straining to speak the words the broken man never could, rising like the Joker's own voice finally breaking out of him after a lifetime of silence, deep sub-bass pulse, monumental cinematic percussion and driving low strings swelling underneath, a lifetime of swallowed love and fear and hurt hardening into dread and then breaking open into release, severe and epic yet devastatingly intimate, believable real orchestral performance feel, modern blockbuster gravitas, wholly original and transformative and NOT a copy of any existing cue, instrumental only, no vocals, no lyrics",
    "origin":            "original dramatic film score, intimate solo felt piano with warm cello and distant low strings, fragile childhood loneliness, human breaths between notes, real orchestral dynamics, cinematic, instrumental only, no vocals, no lyrics, no synthetic loop",
    "the_deepening":     "original romantic tragedy score, chamber strings swelling under restrained piano, tender longing becoming dangerous obsession, warm real orchestra, subtle rubato, aching but not sentimental, cinematic, instrumental only, no vocals, no lyrics",
    "lullaby":           "original broken lullaby score, celeste and soft piano over close-mic strings, trembling tenderness, apology, forgiveness, tears held back, intimate real emotion, cinematic, instrumental only, no vocals, no lyrics",
    "my_beloved":        "original sweeping love-and-loss theme, lush strings, solo violin answer, deep cello counterline and piano, devotion with guilt underneath, dramatic but truthful, real orchestral performance feel, cinematic, instrumental only, no vocals, no lyrics",
    "lady_in_red":       "original nocturne score, elegant slow waltz pulse, warm strings and dark piano, desire mixed with unease, beautiful and threatening, cinematic realism, instrumental only, no vocals, no lyrics",
    "hallucination_bed": "original psychological dread score, bowed glass textures, detuned piano, low string drones, heartbeat-like bass pulse, hallucination and numbness, realistic cinematic sound design, instrumental only, no vocals, no lyrics",
    "the_breaking":      "original tragic breakdown score, low brass, tremolo strings, violent cello swells, fractured piano motif, medical horror without melodrama, crushing real emotion, cinematic, instrumental only, no vocals, no lyrics",
    "the_crossing":      "original threshold score, deep sub pulse, suspended strings, distant metallic resonance, midnight laugh turning into blackout, fate closing in, cinematic, instrumental only, no vocals, no lyrics",
    "machine_years":     "original clinical-machine-years score, sterile electronic pulses fused with cold strings, respirator rhythm, years passing without warmth, human soul buried under machinery, cinematic, instrumental only, no vocals, no lyrics",
    "the_waking":        "original awakening score, cautious piano notes returning, low strings opening into fragile hope, body remembering itself, restrained real triumph, cinematic, instrumental only, no vocals, no lyrics",
    "the_reveal":        "original revelation score, rising orchestra, low choir-like synth pad without voices, terrible truth unveiled, awe and dread together, massive but controlled, cinematic, instrumental only, no vocals, no lyrics",
    "the_mirror":        "original mirror-confrontation score, two piano motifs answering each other, uneasy strings, identity split and reflected, tense human drama, cinematic, instrumental only, no vocals, no lyrics",
    "the_amber_note":    "original amber-memory score, single warm sustained note blooming into strings and piano harmonics, bittersweet memory preserved, tender and unreachable, cinematic, instrumental only, no vocals, no lyrics",
    "the_verdict":       "original final judgement score, grave descending strings, low brass, slow timpani heartbeat, solemn crushing verdict, dignity under pain, cinematic, instrumental only, no vocals, no lyrics",
    "the_unreadable":    "original unresolved ending score, sparse haunting piano, air, distant strings fading into ambiguous silence, no neat answer, real grief, cinematic, instrumental only, no vocals, no lyrics",
}
DEFAULT = "original dramatic cinematic score, real orchestra feel, piano and strings, emotionally truthful, instrumental only, no vocals, no lyrics"


def sh(*a):
    subprocess.run(list(a), check=True)


def write_wav(path, audio, sr):
    import numpy as np
    a = np.asarray(audio, dtype="float32").flatten()
    a = np.clip(a, -1.0, 1.0)
    pcm = (a * 32767.0).astype("<i2").tobytes()
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm)


def crossfade_concat(wavs, out_wav, xf=1.25):
    import shutil
    if len(wavs) == 1:
        shutil.copy2(wavs[0], out_wav)
        return
    cur = wavs[0]
    for i, nxt in enumerate(wavs[1:], 1):
        merged = out_wav.replace(".wav", f".m{i}.wav")
        sh("ffmpeg", "-y", "-loglevel", "error", "-i", cur, "-i", nxt,
           "-filter_complex", f"acrossfade=d={xf}:c1=tri:c2=tri", merged)
        cur = merged
    shutil.copy2(cur, out_wav)


def load_model():
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    requested = os.environ.get("MG_MODEL", "facebook/musicgen-medium")
    for name in [requested, "facebook/musicgen-medium", "facebook/musicgen-small"]:
        try:
            proc = AutoProcessor.from_pretrained(name)
            mg = MusicgenForConditionalGeneration.from_pretrained(name).to(dev)
            sr = int(mg.config.audio_encoder.sampling_rate)
            print(f"[score] MusicGen ready: {name} on {dev} @ {sr} Hz", flush=True)
            return proc, mg, dev, sr
        except Exception as e:
            print(f"[score] {name} failed to load ({e})", flush=True)
    raise RuntimeError("No MusicGen model could be loaded")


def gen_take(proc, mg, dev, prompt, secs, seed):
    import torch
    torch.manual_seed(seed)
    toks = int(secs * 50) + 8
    inp = proc(text=[prompt], padding=True, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = mg.generate(**inp, max_new_tokens=toks, do_sample=True,
                          guidance_scale=3.6, temperature=1.05)
    return out[0, 0].detach().to("cpu").float().numpy()


def _master_wav(src, dst, sr=48000):
    """Professional mastering chain via ffmpeg:
       1. Normalize to -23 LUFS (music standard)
       2. Subtle multiband enhancement via EQ
       3. Stereo widening (mid-side boost)
       4. True-peak limiting
    All CPU-side, no extra Python deps, deterministic."""
    master_af = (
        f"aresample={sr}:resampler=soxr,"
        # Subtle bass shelf + air boost
        f"equalizer=f=80:t=1:w=1:g=1.5,"
        f"equalizer=f=10000:t=h:w=0.8:g=1.0,"
        # Stereo widening (mid-side)
        f"aevalsrc=0.5*val(0)+0.5*val(1):0.5*val(0)-0.5*val(1)[M_S];"
        f"[M_S]pan=stereo|FL=0.5*FC+0.65*FL|FR=0.5*FC+0.65*FR,"
        # Loudness normalize to music standard
        f"loudnorm=I=-23:LRA=13:TP=-2,"
        # True-peak limiter
        f"alimiter=limit=0.93:attack=0.3:release=10:level=on"
    )
    sh("ffmpeg", "-y", "-loglevel", "error", "-i", src, "-af", master_af, dst)


def movement_durations(cfg, shots):
    out = []
    for name, spec in cfg["score"]["movements"].items():
        d = 0.0
        for sid in spec["shots"]:
            s = shots.get(sid)
            if s:
                d += float(s.get("dur", 0) or 0)
        out.append((name, spec["file"], d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    sys.path.insert(0, os.path.join(root, "src"))
    from cfgutil import load_cfg
    cfg = load_cfg(root)
    shots = {s["id"]: s for s in json.load(open(os.path.join(root, cfg["shots_file"])))["shots"]}
    bg = os.path.join(root, cfg["paths"]["audio_bg"])
    work = os.path.join(root, cfg["paths"]["work"]) if not a.dry_run else os.path.join(root, "work", "_score_dryrun")
    out_dir = bg if not a.dry_run else os.path.join(root, "work", "_score_dryrun")
    os.makedirs(work, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Target sample rate from config
    score_cfg = cfg.get("score", {}) or {}
    target_sr = int(score_cfg.get("sample_rate", 48000))
    chunk_cap = float(os.environ.get("SCORE_CHUNK_SEC", "22"))

    movements = movement_durations(cfg, shots)
    print(f"[score] {len(movements)} original dramatic movements -> {out_dir}", flush=True)

    proc = mg = dev = None
    native_sr = 32000
    if not a.dry_run:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            proc, mg, dev, native_sr = load_model()
        except Exception as e:
            print(f"[score] model unavailable ({e}) -> cinematic warm-pad fallback", flush=True)
            proc = None

    for idx, (name, fname, dur) in enumerate(movements):
        prompt = MOOD.get(name, DEFAULT)
        target = max(dur + 4.0, 10.0)
        nchunks = max(1, math.ceil(target / chunk_cap))
        seg = target / nchunks
        dst = os.path.join(out_dir, fname)
        print(f"[score] {idx+1}/{len(movements)} {name}: {target:.1f}s, {nchunks} take(s)", flush=True)

        if a.dry_run or proc is None:
            # Procedural fallback: warm pad (sine tones)
            tmp = os.path.join(work, f"{name}.pad.wav")
            sh("ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
               f"sine=frequency=196:duration={target:.2f}",
               "-f", "lavfi", "-i", f"sine=frequency=294:duration={target:.2f}",
               "-f", "lavfi", "-i", f"sine=frequency=392:duration={target:.2f}",
               "-filter_complex",
               "[0][1][2]amix=inputs=3,volume=-28dB,"
               "afade=t=in:st=0:d=1.5,afade=t=out:st=0:d=0.01",
               tmp)
            _master_wav(tmp, dst, target_sr)
            continue

        # Generate with MusicGen
        wavs = []
        for c in range(nchunks):
            audio = gen_take(proc, mg, dev, prompt, seg + 1.0, seed=1000 * (idx + 1) + c)
            w = os.path.join(work, f"{name}.take{c}.wav")
            write_wav(w, audio, native_sr)
            wavs.append(w)
        merged = os.path.join(work, f"{name}.merged.wav")
        crossfade_concat(wavs, merged, xf=1.25)
        mastered = os.path.join(work, f"{name}.master.wav")
        _master_wav(merged, mastered, target_sr)
        # Encode to MP3 for delivery
        sh("ffmpeg", "-y", "-loglevel", "error", "-i", mastered,
           "-codec:a", "libmp3lame", "-b:a", "320k", dst)

    print(f"[ok] original dramatic score -> {out_dir} ({len(movements)} stems @ {target_sr} Hz)", flush=True)


if __name__ == "__main__":
    main()
