#!/usr/bin/env python3
"""SOUND DESIGN / AMBIENCE stage -- the missing world.

Until now `ambience_prompt` in audio/prompts_confrontation.csv was written but consumed
by NOTHING: the film had dialogue + score and dead silence where the room, the rain,
the gunfire, the electrical arc, the tinnitus ring and the body-fall should live.
This stage generates a per-shot ambience/foley bed for every shot and one continuous
bed spanning the film, then stitch lays it under the music.

Two modes:
  * GPU/model mode (KIT_SFX_MODE=audiogen): facebook/audiogen-medium text->audio from
    each ambience_prompt. Used automatically on a paid GPU box when torch+model exist.
  * CPU/offline mode (default here, KIT_SFX_MODE=cpu): fully procedural foley synthesized
    with ffmpeg lavfi from keywords in each ambience_prompt -- room tone always, plus
    gunfire cracks, sub-bass fall, tinnitus ring, electrical crackle, rain/storm, fire.
    Deterministic, offline, FREE -- the world is never silent even with zero budget.

Outputs: work/ambience/<shot>.wav (per shot, = shot dur) and work/ambience/bed_amb.wav
(concatenated in story order). Mixed by stitch under the score, ducked beneath dialogue.
"""
import os, sys, json, csv, subprocess, argparse

SR = 48000

def ff(*a):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *a], check=True)

def _has(text, *keys):
    t = (text or "").lower()
    return any(k in t for k in keys)

def _layers(prompt, dur):
    """Return a list of lavfi source graphs (each ending in a single labelled output)
    that together voice this shot's world. Purely procedural / keyword-driven."""
    L = []
    d = f"{dur:.3f}"
    # 0) room tone -- ALWAYS present so no shot is dead-silent (low brown-ish noise)
    L.append(f"anoisesrc=d={d}:c=brown:a=0.05,lowpass=f=600,volume=-3dB")
    # 1) rain / storm
    if _has(prompt, "rain", "storm", "downpour", "drizzle", "wet"):
        L.append(f"anoisesrc=d={d}:c=pink:a=0.14,highpass=f=800,lowpass=f=9000,volume=-6dB")
    # 1b) water / seepage -- water falling through stone seepages, dripping, trickling
    if _has(prompt, "water", "drip", "seep", "trickle", "leak"):
        # a low wet trickle bed under everything
        L.append(f"anoisesrc=d={d}:c=pink:a=0.10,bandpass=f=1200:width_type=o:w=1.5,"
                 f"tremolo=f=0.7:d=0.5,volume=-14dB")
        # discrete drips falling through the seepages, spaced across the shot
        for _k, at in enumerate((0.10, 0.28, 0.44, 0.61, 0.79, 0.93)):
            st = max(0.0, dur * at)
            L.append(f"aevalsrc='0.6*sin(2*PI*(950+400*exp(-32*(t-{st:.3f})))*t)*gt(t,{st:.3f})':"
                     f"s={SR}:d={d},afade=t=out:st={st+0.02:.3f}:d=0.20,volume=-15dB")
    # 2) wind / cavern / vast space / echoing halls
    if _has(prompt, "wind", "cavern", "vast", "hollow", "empty hall", "cathedral", "corridor", "hall", "echo"):
        L.append(f"anoisesrc=d={d}:c=brown:a=0.10,bandpass=f=350:width_type=o:w=2,"
                 f"tremolo=f=0.4:d=0.7,volume=-8dB")
    # 3) fire / embers -> crackle (per direction: 'fires everywhere but her')
    if _has(prompt, "fire", "ember", "flame", "burning", "blaze"):
        L.append(f"anoisesrc=d={d}:c=pink:a=0.5,highpass=f=2500,"
                 f"tremolo=f=17:d=0.9,tremolo=f=7:d=0.8,volume=-12dB")
    # 4) electrical arc / lightning / ECT (the lightning that took the man)
    if _has(prompt, "electric", "lightning", "arc", "spark", "current", "shock", "ect", "buzz", "weld", "welding", "solder", "sizzle"):
        L.append(f"aevalsrc='0.2*sin(2*PI*60*t)+0.1*sin(2*PI*120*t)':s={SR}:d={d},volume=-14dB")
        L.append(f"anoisesrc=d={d}:c=white:a=0.6,highpass=f=4000,"
                 f"tremolo=f=23:d=1.0,volume=-16dB")
    # 4b) welding / solder arc -> bright electric sizzle and intermittent spark spatter
    if _has(prompt, "weld", "welding", "solder", "sizzle"):
        L.append(f"anoisesrc=d={d}:c=white:a=0.7,highpass=f=3500,"
                 f"tremolo=f=42:d=1.0,tremolo=f=9:d=0.8,volume=-13dB")
        for _k, at in enumerate((0.15, 0.40, 0.66, 0.88)):
            st = max(0.0, dur * at)
            L.append(f"anoisesrc=d={d}:c=white:a=0.9,highpass=f=5000,"
                     f"afade=t=in:st={st:.3f}:d=0.004,afade=t=out:st={st:.3f}:d=0.10,volume=-8dB")
    # 5) gunfire / gunshot -> sharp transient cracks (fired around the room, never at her)
    if _has(prompt, "gun", "gunfire", "gunshot", "shot", "fire point-blank", "pistol", "revolver"):
        # three cracks spaced across the shot
        for k, at in enumerate((0.2, 0.5, 0.78)):
            st = max(0.0, dur * at)
            L.append(f"anoisesrc=d={d}:c=white:a=0.9,highpass=f=1200,"
                     f"afade=t=in:st={st:.3f}:d=0.005,afade=t=out:st={st:.3f}:d=0.14,"
                     f"volume=-4dB")
            # each crack kicks a low thump
            L.append(f"aevalsrc='0.7*sin(2*PI*(90*exp(-9*(t-{st:.3f})))*t)*gt(t,{st:.3f})':"
                     f"s={SR}:d={d},afade=t=out:st={st+0.05:.3f}:d=0.25,volume=-6dB")
    # 6) tinnitus / ring / ears (after the gun -- the world goes to a high ring)
    if _has(prompt, "tinnitus", "ring", "ringing", "ears", "deaf", "high whine"):
        L.append(f"aevalsrc='0.12*sin(2*PI*6300*t)':s={SR}:d={d},"
                 f"afade=t=in:st=0:d=0.4,afade=t=out:st={max(dur-0.6,0.0):.3f}:d=0.6,volume=-10dB")
    # 7) body fall / collapse / drop / thud -> deep sub impact
    if _has(prompt, "fall", "collapse", "drop", "thud", "body", "slump", "hits the floor", "crumple"):
        st = max(0.0, dur * 0.5)
        L.append(f"aevalsrc='0.9*sin(2*PI*(55*exp(-2.4*(t-{st:.3f})))*t)*gt(t,{st:.3f})':"
                 f"s={SR}:d={d},afade=t=out:st={st+0.6:.3f}:d=0.8,volume=-3dB")
    # 8) heartbeat / pulse / tension
    if _has(prompt, "heartbeat", "pulse", "heart", "throb"):
        L.append(f"aevalsrc='0.5*sin(2*PI*50*t)*(gt(mod(t,0.9),0.0)*lt(mod(t,0.9),0.09)+"
                 f"0.7*gt(mod(t,0.9),0.16)*lt(mod(t,0.9),0.23))':s={SR}:d={d},volume=-10dB")
    return L

def _synth_cpu(prompt, dur, dst, gain_db=-8):
    layers = _layers(prompt, dur)
    parts, labels = [], []
    for i, g in enumerate(layers):
        parts.append(f"{g}[l{i}]")
        labels.append(f"[l{i}]")
    graph = (";".join(parts) + ";" + "".join(labels) +
             f"amix=inputs={len(labels)}:normalize=0[mx];"
             f"[mx]aformat=sample_rates={SR}:channel_layouts=stereo,"
             f"volume={gain_db}dB,alimiter=limit=0.9[out]")
    ff("-filter_complex", graph, "-map", "[out]", "-t", f"{dur:.3f}", dst)

def _synth_audiogen(prompt, dur, dst):
    import torch  # noqa
    from audiocraft.models import AudioGen
    from audiocraft.data.audio import audio_write
    model = _synth_audiogen._m if hasattr(_synth_audiogen, "_m") else None
    if model is None:
        model = AudioGen.get_pretrained("facebook/audiogen-medium")
        _synth_audiogen._m = model
    model.set_generation_params(duration=max(1.0, dur))
    wav = model.generate([prompt])[0].cpu()
    stem = dst[:-4] if dst.endswith(".wav") else dst
    audio_write(stem, wav, model.sample_rate, strategy="loudness")

def build(root):
    sys.path.insert(0, os.path.join(root, "src"))
    from cfgutil import load_cfg
    cfg = load_cfg(root)
    sfx = (cfg.get("sfx") or {})
    if not sfx.get("enabled", True):
        print("[sfx] disabled in config; skipping.")
        return None
    mode = os.environ.get("KIT_SFX_MODE", sfx.get("mode", "cpu")).lower()
    gain_db = float(sfx.get("gain_db", -8))
    work = cfg["paths"]["work"]
    adir = os.path.join(work, "ambience"); os.makedirs(adir, exist_ok=True)
    shots = json.load(open(os.path.join(root, cfg["shots_file"])))["shots"]
    durs = {s["id"]: float(s.get("dur", 4)) for s in shots}
    # map ambience prompts
    amb = {}
    pth = os.path.join(root, cfg["paths"]["audio_prompts"])
    for r in csv.DictReader(open(pth)):
        amb[r["shot_id"]] = r.get("ambience_prompt", "") or ""
    made = []
    for s in shots:
        sid = s["id"]; dur = durs[sid]; prompt = amb.get(sid, "")
        dst = os.path.join(adir, f"{sid}.wav")
        try:
            if mode == "audiogen":
                _synth_audiogen(prompt, dur, dst)
            else:
                _synth_cpu(prompt, dur, dst, gain_db)
        except Exception as e:
            print(f"[sfx][warn] {sid}: {e} -> silent room tone fallback")
            _synth_cpu("", dur, dst, gain_db)
        made.append((sid, dst))
    # concatenate one continuous ambience bed in story order
    lst = os.path.join(adir, "amb_list.txt")
    open(lst, "w").write("".join(f"file '{p}'\n" for _, p in made))
    bed = os.path.join(adir, "bed_amb.wav")
    ff("-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", bed)
    print(f"[sfx] mode={mode} shots={len(made)} -> {bed}")
    return bed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    build(os.path.abspath(a.root))

if __name__ == "__main__":
    main()
