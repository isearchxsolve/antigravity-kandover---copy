#!/usr/bin/env python3
"""Build THE ONE - the single merged original film.

Fuses the two arcs into ONE continuous river:
  ACT I  - THE BEAUTIFUL : the real Bangalore memoir (real Kunal/Gunja likeness)
  CROSS  - THE CROSSING  : the midnight laugh into blackout (still the real man)
  ACT II - THE METABOLIZED: the invented machine-augmented creature (forged, no photos)

Deterministic + idempotent. Reads shots_bangalore.json + shots_observer.json and
their line scripts, and writes:
  shots_one.json          (merged, per-shot tonal register baked in)
  script/lines_one.csv    (lead lines + a recurring CHORUS layer)
  config_one.yaml         (hybrid: real characters + forged observer, merged score)
"""
import os, csv, json, copy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load_json(p):
    return json.load(open(os.path.join(ROOT, p), encoding="utf-8"))

# ---- the ordered spine -------------------------------------------------------
# (shot_id, source, movement)   source: 'm'=shots_bangalore.json  'o'=shots_observer.json
ACT_I = [
    ("p01_thin_boy","m","origin"), ("p02_rejection","m","origin"),
    ("b03_interview","m","the_deepening"), ("b03_tribute","m","the_deepening"),
    ("b03a_minute","m","the_deepening"), ("b03b_tap","m","the_deepening"),
    ("b03b2_embrace","m","lullaby"), ("b03b2_confession","m","lullaby"),
    ("b03b2_reply","m","lullaby"), ("b03b2_plea","m","lullaby"), ("b03b2_vo","m","lullaby"),
    ("b03c_nooneshearing","m","my_beloved"), ("b03c2_possessive","m","my_beloved"),
    ("b03e_appraisal","m","my_beloved"), ("b03e2_angel","m","my_beloved"),
    ("b03f_apology","m","my_beloved"), ("b03d_father","m","my_beloved"),
    ("b04_trembling","m","my_beloved"), ("b04a_leaving_project","m","my_beloved"),
    ("b04b_lift","m","my_beloved"), ("b04c_observer","m","my_beloved"),
    ("b05_doctor","m","lady_in_red"), ("b05i_dress","m","lady_in_red"), ("b05i_vo","m","lady_in_red"),
    ("b05e_numbing","m","hallucination_bed"), ("b05f_hallucination","m","hallucination_bed"),
    ("b05f2_ect","m","the_breaking"), ("b05f3_aftermath","m","the_breaking"),
    ("b05f4_vegetable","m","the_breaking"), ("b05f5_norecognition","m","the_breaking"),
    ("b05wl_wohlamhe","m","the_breaking"), ("b05om_omkara","m","the_breaking"),
    ("b05g_panic","m","the_breaking"), ("b05f6_doyouremember","m","the_breaking"),
    ("b05f7_lastpoint","m","the_breaking"),
]
CROSSING = [
    ("obs05_lightsdie","o","the_crossing"), ("obs06_stillness","o","the_crossing"),
    ("obs07_laugh","o","the_crossing"), ("obs08_blackout","o","the_crossing"),
]
ACT_II = [
    ("obs09_coma","o","machine_years"), ("obs10_ectoverload","o","machine_years"),
    ("obs11_rebuild","o","machine_years"), ("obs12_firstcurrent","o","machine_years"),
    ("obs13_rise","o","the_waking"), ("obs14_mirror_first","o","the_waking"),
    ("obs02_facereveal","o","the_waking"), ("obs15_stand","o","the_waking"),
    ("obs01_myth","o","the_reveal"), ("obs03_walk","o","the_reveal"), ("obs16_city","o","the_reveal"),
    ("obs17_doctors","o","the_mirror"), ("obs18_satire","o","the_mirror"), ("obs19_world","o","the_mirror"),
    ("obs20_sealed","o","the_amber_note"), ("obs21_hershadow","o","the_amber_note"), ("obs22_unreachable","o","the_amber_note"),
    ("obs23_monologue","o","the_verdict"), ("obs24_riseabove","o","the_verdict"),
    ("obs25_dawn","o","the_unreadable"), ("obs26_lastlook","o","the_unreadable"),
    ("obs27_gone","o","the_unreadable"), ("obs28_endcard","o","the_unreadable"),
]
SPINE = ACT_I + CROSSING + ACT_II
CROSS_IDS = {c[0] for c in CROSSING}
CROSS_OVERRIDE = {"chars": ["kunal"], "era": "40s"}  # the crossing is still the real man, aged

# ---- the recurring CHORUS refrain (the theme, told again and again) ----------
CHORUS_VOICE = ("distant reverent whispered chorus, low doubled murmur, almost subliminal, "
                "philosophical, cold cavernous reverb, unhurried")
CHORUS = [
    ("p01_thin_boy", "He was born beautiful. He was born ugly. It was the same wound - the face they were given, the identity the world would not hold."),
    ("b03_interview", "The tender one was loved for his tenderness. It was real. It could never build a home."),
    ("b05f2_ect", "What the world cannot hold, it tries to erase. Mere acceptance was never enough."),
    ("obs07_laugh", "This is the crossing - the beautiful face and the ruined one, meeting in the dark."),
    ("obs10_ectoverload", "The ugly one and the beautiful one were always one man. Two roads to the same face."),
    ("obs16_city", "They did not accept it. They metabolised it."),
    ("obs24_riseabove", "Not treatment. Not a bandage. A metamorphosis. That is how a man rises above his grief."),
    ("obs28_endcard", "Born to spread a little joy, and move along. Never a normal existence. This was the only solution."),
]

# ---- the protagonist's forged face (original, gravitas register; NOT a real person) --
# Two ages of ONE invented man so the memoir can span youth -> the 40s crossing.
_BASE_FACE = ("an original invented character (not a real or famous person): a gentle-faced "
              "man with warm deep-set dark eyes holding a quiet melancholy, soft expressive "
              "features, calm tender mouth, understated cinematic gravitas, fair-skinned Western")
KUNAL_DESIGN = {
    "kunal_young": {
        "summary": ("The protagonist as a young man (early-mid 20s): " + _BASE_FACE +
                    ", lean, smooth unlined skin, thick dark hair, a boyish softness and "
                    "luminous hopeful eyes, vulnerable and tender. Naturalistic, real."),
        "forge_prompts": {
            "face_front": "cinematic portrait, " + _BASE_FACE + ", early-mid 20s, lean smooth unlined face, thick dark hair, boyish softness, luminous hopeful eyes, front view, soft naturalistic key light, shallow depth of field, photoreal, film grain, highly detailed",
            "face_threequarter": "cinematic portrait, same young man, early-mid 20s, tender expressive eyes, thick dark hair, three-quarter angle, soft motivated light, photoreal, highly detailed",
            "face_profile": "cinematic profile portrait, same young man early 20s, gentle features, dark hair, profile against soft grey, natural light, photoreal, highly detailed",
            "body_full": "full body cinematic shot, a slender gentle young man early 20s in plain modest 1990s Western clothes, quiet posture, natural daylight, photoreal, highly detailed",
            "body_coat": "full body, the same young man walking, plain shirt, ordinary Western city street behind, natural light, photoreal, highly detailed"
        },
        "forge_count_per_prompt": 2,
        "seed_base": 101
    },
    "kunal_40s": {
        "summary": ("The same man aged into his 40s: " + _BASE_FACE +
                    ", now lined and worn, tired shadowed eyes, flecks of grey at the temples, "
                    "heavier and slower, dignified but broken-in. The same bone structure and "
                    "the same gentle eyes as the young man."),
        "forge_prompts": {
            "face_front": "cinematic portrait, " + _BASE_FACE + ", aged into the 40s, lined forehead, tired shadowed deep-set eyes, flecks of grey at the temples, worn dignified face, front view, hard motivated key light, deep shadow, photoreal, film grain, highly detailed",
            "face_threequarter": "cinematic portrait, same man in his 40s, lined weary gentle face, greying temples, three-quarter angle, dramatic single key light, deep chiaroscuro, photoreal, highly detailed",
            "face_profile": "cinematic profile portrait, same man in his 40s, worn gentle features, greying hair, profile against black, rim light, photoreal, highly detailed",
            "body_full": "full body cinematic shot, a tired dignified man in his 40s in plain dark clothes, heavier slower posture, volumetric low light, photoreal, highly detailed",
            "body_coat": "full body, the same 40s man standing alone in a dim room, plain dark clothes, single window light, photoreal, highly detailed"
        },
        "forge_count_per_prompt": 2,
        "seed_base": 131
    }
}

# ---- her forged face (Cate Blanchett register; an original invented woman, NOT a real person) --
_HER_FACE = ("an original invented character (not a real or famous person): a luminous, regal "
             "woman with a cool penetrating gaze, sharp architectural cheekbones, clear "
             "intelligent eyes, a composed and formidable presence, understated cinematic "
             "gravitas, an elegant, fair, luminous Western woman")
GUNJA_DESIGN = {
    "gunja": {
        "summary": ("Her, the woman he loved (late 20s-30s): " + _HER_FACE +
                    ", poised and brilliant, quietly commanding, warm beneath a formidable "
                    "stillness, dressed in elegant evening gowns and tailored dresses. Naturalistic, real."),
        "forge_prompts": {
            "face_front": "cinematic portrait, " + _HER_FACE + ", late 20s-30s, luminous skin, sharp cheekbones, clear penetrating intelligent eyes, composed formidable expression, front view, soft naturalistic key light, shallow depth of field, photoreal, film grain, highly detailed",
            "face_threequarter": "cinematic portrait, the same woman, late 20s-30s, regal luminous face, intelligent eyes, three-quarter angle, soft motivated light, photoreal, highly detailed",
            "face_profile": "cinematic profile portrait, the same woman, elegant architectural features, hair swept back, profile against soft grey, natural light, photoreal, highly detailed",
            "body_full": "full body cinematic shot, a poised elegant woman late 20s-30s in a graceful evening gown, quiet commanding posture, natural daylight, photoreal, highly detailed",
            "body_coat": "full body, the same woman in an elegant tailored dress, standing composed in a bright architect studio, natural light, photoreal, highly detailed"
        },
        "forge_count_per_prompt": 2,
        "seed_base": 161
    }
}


def _read_lines(p):
    with open(os.path.join(ROOT, p), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))

def build():
    M = _load_json("shots_bangalore.json"); S1 = {s["id"]: s for s in M["shots"]}; G1 = M["globals"]
    O = _load_json("shots_observer.json");  S2 = {s["id"]: s for s in O["shots"]}; G2 = O["globals"]

    out_shots = []
    for sid, src, mov in SPINE:
        base = S1 if src == "m" else S2
        if sid not in base:
            raise SystemExit(f"[FAIL] shot {sid} not found in source '{src}'")
        sh = copy.deepcopy(base[sid])
        g = G1 if src == "m" else G2
        sh.setdefault("negative", g["negative_prompt"])  # bake tonal register per shot
        sh["tone"] = g["tone"]
        sh["movement"] = mov
        if sid in CROSS_IDS:
            sh.update(copy.deepcopy(CROSS_OVERRIDE))
            sh["tone"] = G2["tone"]  # real man, shot in the myth's register
        out_shots.append(sh)

    merged = {
        "film": "THE ONE - Same Wound (the single film)",
        "note": ("One original film, end to end. The real memoir (Act I, real likeness) crosses "
                 "at the midnight laugh (real face) into the metabolized creature (Act II, forged, "
                 "no photos). A recurring chorus voice carries the theme: the ugly one and the "
                 "beautiful one are one man - the same identity-wound, metabolized, not merely accepted."),
        "globals": {
            "fps": G2["fps"], "width": G2["width"], "height": G2["height"],
            "base_model": G2["base_model"], "quant": G2["quant"], "steps": G2["steps"],
            "cfg": G2["cfg"], "sampler": G2["sampler"], "scheduler": G2["scheduler"],
            "interpolate_to_fps": G2["interpolate_to_fps"], "upscale": G2["upscale"],
            "rejection_samples": G2["rejection_samples"],
            "tone": G2["tone"], "negative_prompt": G2["negative_prompt"],
        },
        "character_design": {**(O.get("character_design") or {}), **KUNAL_DESIGN, **GUNJA_DESIGN},
        "characters": {**M.get("characters", {}), **O.get("characters", {})},
        "shots": out_shots,
    }
    json.dump(merged, open(os.path.join(ROOT, "shots_one.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)

    # ---- lines_one.csv : lead lines (merged) + chorus layer -------------------
    L1 = {r["shot_id"]: r for r in _read_lines("script/lines.csv")}
    L2 = {r["shot_id"]: r for r in _read_lines("script/lines_observer.csv")}
    rows = []
    for sid, src, mov in SPINE:
        srcmap = L1 if src == "m" else L2
        r = srcmap.get(sid)
        if r and (r.get("text") or "").strip():
            rows.append({"shot_id": sid, "character": r.get("character", ""),
                         "language": r.get("language", "en"), "text": r["text"],
                         "voice_description": r.get("voice_description", ""), "layer": "lead"})
    for sid, text in CHORUS:
        rows.append({"shot_id": sid, "character": "chorus", "language": "en",
                     "text": text, "voice_description": CHORUS_VOICE, "layer": "chorus"})
    cols = ["shot_id", "character", "language", "text", "voice_description", "layer"]
    with open(os.path.join(ROOT, "script", "lines_one.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- config_one.yaml : hybrid config with merged, contiguous movements ----
    import yaml
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config_observer.yaml"), encoding="utf-8"))
    cfg["film_title"] = "THE ONE"
    cfg["shots_file"] = "shots_one.json"
    cfg["paths"]["script"] = "script/lines_one.csv"
    cfg["paths"]["audio_bg"] = "audio/bg_one"
    cfg["paths"]["audio_prompts"] = "audio/prompts_one.csv"
    cfg["paths"]["work"] = "/tmp/one_work"
    cfg["paths"]["out"] = "output_one"
    cfg["real_characters"] = []              # no real photos: the entire cast is AI-generated
    _fc = dict(cfg.get("character_forge") or {})
    _fc["enabled"] = True
    _fc["targets"] = ["kunal_young", "kunal_40s", "observer", "gunja"]  # forge both ages, the creature, and her
    cfg["character_forge"] = _fc
    cfg["score"]["chorus_gain_db"] = -14             # the background voice sits under everything
    movements, order = {}, []
    for sid, src, mov in SPINE:
        if mov not in movements:
            movements[mov] = {"file": f"{mov}.mp3", "shots": []}; order.append(mov)
        movements[mov]["shots"].append(sid)
    cfg["score"]["movements"] = {m: movements[m] for m in order}
    yaml.safe_dump(cfg, open(os.path.join(ROOT, "config_one.yaml"), "w", encoding="utf-8"),
                   sort_keys=False, allow_unicode=True)

    os.makedirs(os.path.join(ROOT, "audio", "bg_one"), exist_ok=True)
    open(os.path.join(ROOT, "audio", "bg_one", ".gitkeep"), "a").close()

    lead = len([r for r in rows if r["layer"] == "lead"])
    chorus = len([r for r in rows if r["layer"] == "chorus"])
    print(f"[ok] shots_one.json: {len(out_shots)} shots")
    print(f"[ok] script/lines_one.csv: {lead} lead + {chorus} chorus")
    print(f"[ok] config_one.yaml: {len(order)} movements -> {order}")

if __name__ == "__main__":
    build()
