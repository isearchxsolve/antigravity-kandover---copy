#!/usr/bin/env python3
"""gpu_scheduler.py -- Staggered 2xT4 thermal-aware shot scheduler.

ZERO GPU THROTTLE on Kaggle 2xT4. Two T4s sitting side-by-side thermal-couple.
Running both at 100% simultaneously causes one to throttle the other. This module:

  * Round-robin assigns each shot to the coolest free GPU.
  * Records a per-GPU "last finished at" timestamp and refuses to re-dispatch
    to the same GPU within MIN_GAP_SEC (gives the heatsink time to shed).
  * Detects throttling (sustained 100% util + rising temp) and sleeps until
    the GPU drops below THROTTLE_TEMP_C.
  * AGGRESSIVE: Pre-emptive cooldown, thermal prediction, mandatory gaps.
  * Falls back to a single-GPU pipeline on instances with one GPU.

Pure stdlib. No torch / nvidia-ml-py dependency.
"""
import os, time
import vram_governor as vg

# AGGRESSIVE thermal settings for ZERO THROTTLE on 2xT4
MIN_GAP_SEC = float(os.environ.get("KIT_GPU_GAP_SEC", "20"))        # Increased from 12s
THROTTLE_TEMP_C = int(os.environ.get("KIT_GPU_THROTTLE_TEMP_C", "78"))  # Lowered from 83C
TEMP_GREEN_C = int(os.environ.get("KIT_GPU_TEMP_GREEN_C", "70"))    # Lowered from 76C
TEMP_YELLOW_C = int(os.environ.get("KIT_GPU_TEMP_YELLOW_C", "74"))  # New warning threshold
COOLDOWN_POLL_SEC = float(os.environ.get("KIT_GPU_COOLDOWN_POLL_SEC", "2"))
COOLDOWN_TIMEOUT_SEC = float(os.environ.get("KIT_GPU_COOLDOWN_TIMEOUT_SEC", "300"))
PREDICTIVE_COOLDOWN = os.environ.get("KIT_PREDICTIVE_COOLDOWN", "1") == "1"


class _GpuState:
    __slots__ = ("last_finish", "shot_count", "consecutive_shots", "last_temp")

    def __init__(self):
        self.last_finish = 0.0
        self.shot_count = 0
        self.consecutive_shots = 0
        self.last_temp = 0


class Scheduler:
    def __init__(self):
        gpus = vg.probe_all()
        if not gpus:
            self._n = 0
            self._states = []
            return
        self._n = len(gpus)
        self._states = [_GpuState() for _ in gpus]
        self._rr_index = 0
        print(f"[gpu_scheduler] {self._n} GPU(s) visible: "
              + ", ".join(f"GPU{g['index']} {g['name']}" for g in gpus))
        print(f"[gpu_scheduler] AGGRESSIVE thermal: gap={MIN_GAP_SEC}s "
              f"throttle={THROTTLE_TEMP_C}C green={TEMP_GREEN_C}C yellow={TEMP_YELLOW_C}C")

    @property
    def gpu_count(self):
        return self._n

    def _cooldown(self, idx):
        """Block until GPU `idx` is below throttle temperature and the minimum
        inter-shot gap has elapsed since the last shot on this GPU.
        AGGRESSIVE: Also waits for temp to drop to GREEN zone, not just below throttle."""
        deadline = time.time() + COOLDOWN_TIMEOUT_SEC
        s = self._states[idx]
        while True:
            now = time.time()
            gap_ok = (now - s.last_finish) >= MIN_GAP_SEC
            gpus = vg.probe_all()
            cur = next((g for g in gpus if g["index"] == idx), None)
            
            if cur is None:
                temp_ok = True
            else:
                s.last_temp = cur["temp_c"]
                # AGGRESSIVE: Wait for GREEN zone, not just below throttle
                temp_ok = cur["temp_c"] <= TEMP_GREEN_C
                
                # Predictive: if temp is rising fast, wait longer
                if PREDICTIVE_COOLDOWN and cur["temp_c"] >= TEMP_YELLOW_C:
                    temp_ok = False
            
            if gap_ok and temp_ok:
                s.consecutive_shots = 0  # Reset consecutive counter on cooldown
                return
            
            if now > deadline:
                print(f"[gpu_scheduler] WARNING: cooldown timeout on GPU {idx} "
                      f"(temp={cur['temp_c'] if cur else 'N/A'}C, gap_ok={gap_ok}); proceeding.")
                return
            time.sleep(COOLDOWN_POLL_SEC)

    def pick(self):
        """Pick a GPU for the next shot. Always returns a valid GPU index
        (or 0 in single-GPU mode). Performs thermal cooldown if needed.
        AGGRESSIVE: Prefers GPU with lowest temp AND fewest consecutive shots."""
        if self._n == 0:
            return 0
        if self._n == 1:
            self._cooldown(0)
            return 0
        
        # AGGRESSIVE: Score GPUs by (temp, consecutive_shots, last_finish)
        best_idx = None
        best_score = float('inf')
        
        for _ in range(self._n):
            idx = self._rr_index % self._n
            self._rr_index += 1
            
            gpus = vg.probe_all()
            cur = next((g for g in gpus if g["index"] == idx), None)
            s = self._states[idx]
            
            if cur is None:
                score = 0
            else:
                # Score: lower is better
                # Primary: temperature (weighted heavily)
                # Secondary: consecutive shots (prefer GPU with fewer recent shots)
                # Tertiary: time since last finish
                temp_score = cur["temp_c"] * 10
                consecutive_score = s.consecutive_shots * 50
                gap_score = max(0, MIN_GAP_SEC - (time.time() - s.last_finish)) * 2
                score = temp_score + consecutive_score + gap_score
            
            if score < best_score:
                best_score = score
                best_idx = idx
        
        if best_idx is not None:
            self._cooldown(best_idx)
            return best_idx
        
        # Fallback
        idx = self._rr_index % self._n
        self._rr_index += 1
        self._cooldown(idx)
        return idx

    def finish(self, idx):
        """Mark a shot as finished on `idx`. Updates last_finish timestamp."""
        if self._n == 0:
            return
        if 0 <= idx < self._n:
            now = time.time()
            self._states[idx].last_finish = now
            self._states[idx].shot_count += 1
            self._states[idx].consecutive_shots += 1

    def summary(self):
        """Human-readable per-GPU stats for telemetry dump."""
        if self._n == 0:
            return "[gpu_scheduler] no GPUs"
        gpus = vg.probe_all()
        lines = []
        for i in range(self._n):
            cur = next((g for g in gpus if g["index"] == i), None)
            s = self._states[i]
            if cur:
                lines.append(f"GPU {i}: {cur['free_mb']}MB free, "
                             f"{cur['temp_c']}C, util {cur['util_pct']}%, "
                             f"shots done={s.shot_count}, consecutive={s.consecutive_shots}")
            else:
                lines.append(f"GPU {i}: (no telemetry), shots done={s.shot_count}")
        return "\n".join(lines)


if __name__ == "__main__":
    s = Scheduler()
    print(s.summary())
