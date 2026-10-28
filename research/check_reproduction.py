"""Re-run committed result artifacts and compare them with the stored files.

Discrete fields (strings, ints, booleans, keys, list lengths) must match
exactly; floats must agree to a relative tolerance, since Python 3.12
changed float ``sum`` rounding. Timing and host metadata are skipped.

    python -m research.check_reproduction
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "research" / "results"

# Fields that legitimately vary between machines or runs.
VOLATILE = {
    "cpu_seconds", "wall_seconds", "adaptive_cpu_seconds",
    "comparator_cpu_seconds", "total_cpu_seconds", "total_wall_seconds",
    "total_cpu_matched_or_lower", "platform", "python",
}


def compare(expected, actual, path, tol, out):
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key in VOLATILE:
                continue
            if key not in expected or key not in actual:
                out.append(f"{path}/{key}: present on one side only")
            else:
                compare(expected[key], actual[key], f"{path}/{key}", tol, out)
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            out.append(f"{path}: length {len(expected)} != {len(actual)}")
            return
        for i, (a, b) in enumerate(zip(expected, actual)):
            compare(a, b, f"{path}[{i}]", tol, out)
    elif (isinstance(expected, float) and isinstance(actual, float)
          and not isinstance(expected, bool)):
        if not math.isclose(expected, actual, rel_tol=tol, abs_tol=1e-15):
            out.append(f"{path}: {expected!r} != {actual!r}")
    elif expected != actual or type(expected) is not type(actual):
        out.append(f"{path}: {expected!r} != {actual!r}")


def run(*args):
    subprocess.run([sys.executable, "-m", *args], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8")
            .splitlines() if line.strip()]


def check_g0_ac(tmp, tol):
    calibration = RESULTS / "G0_AC_CALIBRATION.json"
    run("research.run_g0_ac_heldout", "calibrate",
        "--output", str(tmp / "calibration.json"))
    # Evaluate against the committed calibration so calibration_sha256 must
    # match exactly; a regenerated calibration differs in its float bits.
    run("research.run_g0_ac_heldout", "evaluate",
        "--calibration", str(calibration), "--output", str(tmp / "heldout.json"))
    for name, produced in (("G0_AC_CALIBRATION.json", "calibration.json"),
                           ("G0_AC_HELDOUT.json", "heldout.json")):
        out = []
        compare(json.loads((RESULTS / name).read_text(encoding="utf-8")),
                json.loads((tmp / produced).read_text(encoding="utf-8")),
                "", tol, out)
        yield name, out


def check_g0_y_embodied(tmp, tol):
    name = "G0_Y_EMBODIED_2026-10-27.jsonl"
    run("esmf.embodied_y", "--seeds", "32", "--start", "0",
        "--output", str(tmp / "embodied.jsonl"))
    out = []
    compare(load_jsonl(RESULTS / name), load_jsonl(tmp / "embodied.jsonl"),
            "", tol, out)
    yield name, out


CHECKS = {"g0_ac": check_g0_ac, "g0_y_embodied": check_g0_y_embodied}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=sorted(CHECKS), action="append")
    parser.add_argument("--rel-tol", type=float, default=1e-9)
    args = parser.parse_args()
    failed = False
    with tempfile.TemporaryDirectory() as tmp:
        for key in args.only or sorted(CHECKS):
            for name, diffs in CHECKS[key](Path(tmp), args.rel_tol):
                failed |= bool(diffs)
                print(f"{'FAIL' if diffs else 'ok  '} {name}"
                      + (f" ({len(diffs)} differences)" if diffs else ""))
                for line in diffs[:20]:
                    print(f"     {line}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
