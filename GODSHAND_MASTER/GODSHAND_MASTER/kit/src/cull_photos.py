#!/usr/bin/env python3
"""Auto-culler: turn a big pile of photos in refs/<name>/_inbox/ into a filed,
best-of set under face/ body/ costume/, and write selection_report.csv.

Pipeline per image:
  1. face-detect (OpenCV Haar / DNN)  -> drop no-face frames
  2. dedup (perceptual average hash)  -> drop near-duplicates
  3. quality score (Laplacian sharpness + resolution + exposure + face size)
  4. angle bin (front / three-quarter / profile) and body-vs-close-up (face area ratio)
  5. expression flag (neutral vs smiling)  -> honor serious-tone; prefer neutral
  6. auto-pick top few per bin, copy into face/ body/ costume/

Usage: python3 cull_photos.py <person_dir> [--per-bin 2] [--dry-run]
Deterministic, resumable, override-friendly. No network required.
"""
import os, sys, csv, argparse, shutil

try:
    import cv2, numpy as np
except Exception:
    cv2 = None

EXT = (".jpg", ".jpeg", ".png", ".webp")

def list_images(d):
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(EXT))

def ahash(gray):
    small = cv2.resize(gray, (8, 8))
    return (small > small.mean()).flatten()

def hamming(a, b):
    return int(np.count_nonzero(a != b))

def analyze(path, face_cascade, smile_cascade):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return {"path": path, "face": False, "hash": ahash(gray)}
    fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
    face_area = (fw * fh) / float(w * h)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    exposure = gray.mean()  # 0-255; penalize very dark/bright
    exp_pen = 1.0 - abs(exposure - 120) / 120.0
    # crude angle: face box offset from center + aspect
    cx = (fx + fw / 2) / w
    angle = "front" if 0.4 < cx < 0.6 else ("three_quarter" if 0.28 < cx < 0.72 else "profile")
    roi = gray[fy:fy + fh, fx:fx + fw]
    smiling = len(smile_cascade.detectMultiScale(roi, 1.7, 20)) > 0 if roi.size else False
    kind = "body" if face_area < 0.06 else "face"
    quality = (min(sharp, 1500) / 1500) * 0.5 + min(face_area * 6, 1) * 0.2 + max(exp_pen, 0) * 0.2 \
        + min(min(h, w) / 1024.0, 1) * 0.1
    return {"path": path, "face": True, "hash": ahash(gray), "kind": kind, "angle": angle,
            "smiling": smiling, "quality": round(quality, 4), "sharp": round(sharp, 1),
            "face_area": round(face_area, 4), "res": f"{w}x{h}"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("person_dir")
    ap.add_argument("--per-bin", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if cv2 is None:
        print("[FAIL] opencv-python not available; install opencv-python-headless.")
        sys.exit(1)
    inbox = os.path.join(a.person_dir, "_inbox")
    imgs = list_images(inbox)
    if not imgs:
        print(f"[warn] no images in {inbox}; nothing to cull (hand-placed photos are used as-is).")
        return
    base = cv2.data.haarcascades
    face_cascade = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier(base + "haarcascade_smile.xml")
    recs, seen = [], []
    for p in imgs:
        r = analyze(p, face_cascade, smile_cascade)
        if not r or not r.get("face"):
            continue
        if any(hamming(r["hash"], h) <= 5 for h in seen):  # near-duplicate
            continue
        seen.append(r["hash"])
        recs.append(r)
    # bins: face/front, face/three_quarter, face/profile, body/*
    bins = {}
    for r in recs:
        key = (r["kind"], r["angle"]) if r["kind"] == "face" else ("body", "any")
        bins.setdefault(key, []).append(r)
    report = os.path.join(a.person_dir, "selection_report.csv")
    picked = []
    with open(report, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["path", "decision", "kind", "angle", "smiling", "quality", "sharp", "res"])
        for key, group in bins.items():
            # serious-tone rule: prefer neutral; smiling only as tie-breaker fill
            group.sort(key=lambda r: (r["smiling"], -r["quality"]))
            keep = group[: a.per_bin]
            for r in group:
                dec = "KEEP" if r in keep else "skip"
                wr.writerow([os.path.basename(r["path"]), dec, r["kind"], r["angle"],
                             r["smiling"], r["quality"], r["sharp"], r["res"]])
            for r in keep:
                dest_sub = "body" if r["kind"] == "body" else "face"
                picked.append((r["path"], dest_sub))
    if not a.dry_run:
        for src, sub in picked:
            dd = os.path.join(a.person_dir, sub)
            os.makedirs(dd, exist_ok=True)
            shutil.copy2(src, os.path.join(dd, os.path.basename(src)))
    print(f"[ok] {os.path.basename(a.person_dir)}: scanned {len(imgs)}, kept {len(picked)} "
          f"across {len(bins)} bins. Report -> {report}")
    print("      Review selection_report.csv and move/replace files to override.")

if __name__ == "__main__":
    main()
