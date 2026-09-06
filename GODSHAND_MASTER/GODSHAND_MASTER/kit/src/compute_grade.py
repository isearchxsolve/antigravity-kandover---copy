#!/usr/bin/env python3
"""Compute a color-grade TARGET from a reference clip: global LAB mean/std over
sampled frames, saved to refs/grade_target.json and consumed by grade_match.py.

Decodes via ffmpeg -> PNG -> cv2.imread(COLOR) because OpenCV's VideoCapture is
unreliable on some H.264 profiles in headless builds (it can silently return
degenerate/grayscale frames). This path always yields real 3-channel BGR.

Usage: compute_grade.py REFERENCE.mp4 OUT.json [SAMPLE_FPS]
"""
import sys, os, json, glob, tempfile, subprocess


def main():
    if len(sys.argv) < 3:
        print("usage: compute_grade.py REFERENCE.mp4 OUT.json [SAMPLE_FPS]")
        return 1
    src, out = sys.argv[1], sys.argv[2]
    sample_fps = sys.argv[3] if len(sys.argv) > 3 else "8"
    import cv2, numpy as np
    tmp = tempfile.mkdtemp()
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                        "-vf", f"fps={sample_fps}", os.path.join(tmp, "f_%05d.png")])
    files = sorted(glob.glob(os.path.join(tmp, "f_*.png")))
    if r.returncode != 0 or not files:
        print("failed to decode frames from", src)
        return 1
    ssum = np.zeros(3, np.float64)
    ssq = np.zeros(3, np.float64)
    cnt = 0
    for f in files:
        im = cv2.imread(f, cv2.IMREAD_COLOR)  # force 3-channel BGR
        if im is None:
            continue
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float64).reshape(-1, 3)
        ssum += lab.sum(0)
        ssq += (lab * lab).sum(0)
        cnt += lab.shape[0]
    if cnt == 0:
        print("no readable frames")
        return 1
    mean = ssum / cnt
    std = np.sqrt(np.maximum(ssq / cnt - mean ** 2, 1e-6))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump({
        "source": os.path.basename(src),
        "note": "OpenCV 8-bit LAB global mean/std; target look for grade_match.py",
        "colorspace": "opencv-LAB-8bit",
        "frames_sampled": len(files),
        "lab_mean": [float(x) for x in mean],
        "lab_std": [float(x) for x in std],
    }, open(out, "w"), indent=2)
    print("wrote", out, "| frames", len(files),
          "| mean", mean.round(1).tolist(), "| std", std.round(1).tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
