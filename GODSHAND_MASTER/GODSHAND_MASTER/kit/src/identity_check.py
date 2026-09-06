#!/usr/bin/env python3
"""identity_check.py -- flag IDENTITY DRIFT across shots (the monetization-critical check).

For every rendered shot it detects the face(s), embeds them with ArcFace, and compares
against each listed character's reference images (refs/<char>/). A cosine similarity below
`identity.drift_cosine` means the on-screen face no longer matches that character -- the
single biggest risk for a monetized AI film (viewers and platforms notice face drift).

Requires insightface on the render box (onnxruntime is already present):
    pip install insightface onnxruntime-gpu   # or onnxruntime for CPU
First run downloads the buffalo_l model (~300MB). On a CPU-only box set KIT_FACE_CTX=-1.

    python3 src/identity_check.py            # check work/<id>/final.mp4
    python3 src/identity_check.py --clips    # check output/clips/<id>/<id>_master.mp4

Exit 0 = no drift. 1 = drift found. 2 = cannot run (missing insightface / no references).
"""
import os, sys, json, glob, argparse


def _load_app():
    """Return an insightface FaceAnalysis app, or None with an explanation."""
    try:
        from insightface.app import FaceAnalysis
    except Exception:
        print("[identity] insightface is not installed -- cannot run the identity check here.\n"
              "          Install on the render box (onnxruntime is already present):\n"
              "              pip install insightface onnxruntime-gpu   # or onnxruntime (CPU)\n"
              "          First run downloads the buffalo_l model (~300MB).")
        return None
    ctx = int(os.environ.get("KIT_FACE_CTX", "0"))  # 0 = GPU, -1 = CPU
    try:
        app = FaceAnalysis(name="buffalo_l",
                           providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=ctx, det_size=(640, 640))
        return app
    except Exception as e:
        print(f"[identity] failed to initialise insightface: {e}")
        return None


def _largest_faces(app, img):
    faces = app.get(img)
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return faces


def _ref_embedding(app, ref_dir, min_det):
    import cv2, numpy as np
    embs = []
    for fn in sorted(glob.glob(os.path.join(ref_dir, "*"))):
        if os.path.splitext(fn)[1].lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        img = cv2.imread(fn)
        if img is None:
            continue
        faces = _largest_faces(app, img)
        if faces and faces[0].det_score >= min_det:
            embs.append(faces[0].normed_embedding)
    if not embs:
        return None
    v = np.mean(embs, axis=0)
    n = np.linalg.norm(v)
    return v / n if n else None


def _shot_embeddings(app, path, n_frames, min_det):
    import cv2
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release(); return []
    step = max(1, total // max(1, n_frames))
    embs, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            for f in _largest_faces(app, fr):
                if f.det_score >= min_det:
                    embs.append(f.normed_embedding)
        i += 1
    cap.release()
    return embs


def main():
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", action="store_true")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "src"))
    from cfgutil import load_cfg
    cfg = load_cfg(root)
    idc = cfg.get("identity", {}) or {}
    drift = float(idc.get("drift_cosine", 0.35))
    n_frames = int(idc.get("sample_frames", 12))
    min_det = float(idc.get("min_det_score", 0.5))

    refs_root = os.path.join(root, cfg["paths"].get("refs", "refs"))
    override = cfg.get("identity_override", {}) or {}
    forced_dir = os.path.join(root, override.get("refs_dir", "")) if override.get("enabled") else None

    work = cfg["paths"]["work"]
    out = os.path.join(root, cfg["paths"]["out"])
    shots = json.load(open(os.path.join(root, cfg["shots_file"])))["shots"]

    app = _load_app()
    if app is None:
        raise SystemExit(2)

    ref_cache, drift_hits, checked = {}, 0, 0
    for s in shots:
        sid = s["id"]
        chars = s.get("chars", []) or []
        path = (os.path.join(out, "clips", sid, f"{sid}_master.mp4") if a.clips
                else os.path.join(work, sid, "final.mp4"))
        if not os.path.exists(path):
            print(f"[SKIP] {sid}: no rendered file"); continue
        if not chars:
            print(f"[----] {sid}: no characters listed (establishing/insert shot)"); continue
        shot_embs = _shot_embeddings(app, path, n_frames, min_det)
        if not shot_embs:
            print(f"[WARN] {sid}: no face detected in shot ({', '.join(chars)} expected)"); continue
        for c in chars:
            ref_dir = forced_dir or os.path.join(refs_root, c)
            if ref_dir not in ref_cache:
                ref_cache[ref_dir] = _ref_embedding(app, ref_dir, min_det)
            ref = ref_cache[ref_dir]
            if ref is None:
                print(f"[WARN] {sid}/{c}: no usable reference face in {os.path.relpath(ref_dir, root)}"); continue
            best = max(float(np.dot(ref, e)) for e in shot_embs)
            checked += 1
            if best < drift:
                drift_hits += 1
                print(f"[DRIFT] {sid}/{c}: best cosine {best:.2f} < {drift:.2f} -- face does not match reference")
            else:
                print(f"[OK]   {sid}/{c}: cosine {best:.2f}")

    print(f"\n[identity] {checked} (shot,character) pair(s) checked: {drift_hits} DRIFT")
    sys.exit(1 if drift_hits else 0)


if __name__ == "__main__":
    main()
