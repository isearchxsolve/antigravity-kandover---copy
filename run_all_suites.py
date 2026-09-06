#!/usr/bin/env python3
"""run_all_suites.py -- single entrypoint that runs the full local test stack.

Also makes the pervasive-abort suites accumulate failures instead of stopping
at the first one, by running each suite as a separate subprocess and always
continuing through to the next one. Prints a unified matrix at the end.

Run:
    python run_all_suites.py                 # all suites
    python run_all_suites.py --skip gpu_smoke # skip GPU-required suites
    python run_all_suites.py --only cinema4k,edge_case
    python run_all_suites.py --list

Exit 0 iff every suite exits 0. Per-suite stdout/stderr are saved to
$opencode/logs/<suite>.log so you can drill in afterwards.
"""
import os, sys, subprocess, argparse, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_KIT = r"C:\dev\antigravity-kandover\GODSHAND_MASTER\GODSHAND_MASTER\kit"
KIT = Path(os.environ.get("GODSHAND_KIT_ROOT", DEFAULT_KIT))
if not KIT.is_dir():
    alt = Path(r"C:\Users\Admin\Downloads\antigravity-kandover\GODSHAND_MASTER\GODSHAND_MASTER\kit")
    if alt.is_dir():
        KIT = alt

LOG_DIR = HERE / "logs"
LOG_DIR.mkdir(exist_ok=True)

# (suite_name, script_relpath, requires_gpu)
SUITES = [
    ("cinema4k",      "cinema4k_sanity_test.py",     False),
    ("edge_case",     "edge_case_integration_test.py", False),
    ("comprehensive", "comprehensive_smoke_test.py", False),
    ("selftest_quick", "selftest_quick_wrapper.py",  False),  # owns wrapper below
    ("gpu_smoke",     "gpu_smoke.py",                True),
]


def write_selftest_quick_wrapper():
    """Self-test --quick is invoked from the kit, so we need a tiny shim that
    shells out to ``python src/selftest.py --quick``."""
    p = HERE / "selftest_quick_wrapper.py"
    src = (
        "#!/usr/bin/env python3\n"
        "import os, sys, subprocess\n"
        f"KIT = r'{KIT}'\n"
        "p = subprocess.run([sys.executable, 'src/selftest.py', '--quick'],\n"
        "                   cwd=KIT, capture_output=False)\n"
        "sys.exit(p.returncode)\n"
    )
    p.write_text(src, encoding="utf-8")
    return p


def color(s, c):
    cols = {"red": "31", "green": "32", "yellow": "33", "cyan": "36"}
    if not sys.stdout.isatty():
        return s
    return f"\x1b[{cols.get(c,'0')}m{s}\x1b[0m"


def run_suite(name, script, requires_gpu):
    script_path = HERE / script
    if not script_path.is_file():
        return name, "SKIP", f"script missing: {script}", 0
    if requires_gpu and "KIT_RUN_GPU" not in os.environ:
        return name, "SKIP", "GPU-required (set KIT_RUN_GPU=1 to force)", 0

    env = dict(os.environ)
    env["GODSHAND_KIT_ROOT"] = str(KIT)
    if sys.platform == "win32":
        env.setdefault("PYTHONIOENCODING", "utf-8")

    log = LOG_DIR / f"{name}.log"
    t0 = time.time()
    status, detail = "PASS", ""
    try:
        with open(log, "wb") as f:
            print(f"  running {name} ({script}) ...", flush=True)
            r = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(HERE), env=env, stdout=f, stderr=subprocess.STDOUT,
                timeout=900,  # 15 min hard cap per suite
            )
        if r.returncode != 0:
            status = "FAIL"
            detail = f"exit {r.returncode}"
    except subprocess.TimeoutExpired:
        status = "TIMEOUT"
        detail = "killed after 900s"
    except Exception as e:
        status = "ERROR"
        detail = f"uncaught {e!r}"
    return name, status, detail, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-list of suite names to run")
    ap.add_argument("--skip", help="comma-list of suite names to skip")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        print("Available suites:")
        for n, s, gpu in SUITES:
            print(f"  {n:14} (gpu={gpu})  {s}")
        return

    write_selftest_quick_wrapper()
    print(f"\n# run_all_suites -- kit: {KIT}")
    print(f"# logs: {LOG_DIR}\n")

    only = {x.strip() for x in a.only.split(",")} if a.only else None
    skip = {x.strip() for x in a.skip.split(",")} if a.skip else set()

    rows = []
    for name, script, gpu in SUITES:
        if only and name not in only:
            continue
        if name in skip:
            rows.append((name, "SKIP", "explicit --skip", 0))
            continue
        n, st, det, secs = run_suite(name, script, gpu)
        rows.append((n, st, det, secs))
        sym = {"PASS": ".", "FAIL": "X", "SKIP": "-", "TIMEOUT": "T", "ERROR": "?"}[st]
        line = f"  [{sym}] {n:14} {st:7} ({secs:5.1f}s)"
        if det:
            line += f"  {det}"
        print(line)

    # ---- matrix ----
    print("\n" + "=" * 56)
    print("SUITE MATRIX")
    print("=" * 56)
    for n, st, det, secs in rows:
        suffix = f"  {det}" if det else ""
        print(f"  {n:14} {st:7} ({secs:5.1f}s){suffix}")

    n_pass = sum(1 for _, st, _, _ in rows if st == "PASS")
    n_fail = sum(1 for _, st, _, _ in rows if st in ("FAIL", "TIMEOUT", "ERROR"))
    n_skip = sum(1 for _, st, _, _ in rows if st == "SKIP")

    print("\n  suites:  {} pass, {} fail (incl timeout/err), {} skip".format(
        n_pass, n_fail, n_skip))
    print(f"  logs:    {LOG_DIR}")

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main() or 0)
