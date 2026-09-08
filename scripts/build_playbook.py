#!/usr/bin/env python
"""Build the coach playbook (cross-run evidence) from finished run directories.

    uv run python scripts/build_playbook.py --out data/results/playbook.json \
        data/results/evolve data/results/coach_v4/runs data/results/coach_v3/runs
    uv run python scripts/build_playbook.py --show rough-medium-naive --success 0.0 ...

The evolution driver (scripts/evolve_coach.py) rebuilds <root>/playbook.json
before every batch by itself; this script is for ad-hoc builds and for
reading what a coach would be told (``--show``). See llm_feedback/playbook.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from quadruped_rl.llm_feedback.playbook import COACHED_CONDITIONS, Playbook


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("roots", nargs="+", help="result roots or run directories")
    p.add_argument("--out", default="data/results/playbook.json")
    p.add_argument("--all-conditions", action="store_true", help="count random/hillclimb runs too")
    p.add_argument("--show", default=None, help="print the evidence for this terrain-level-reward")
    p.add_argument("--success", type=float, default=0.0, help="current success rate for --show")
    args = p.parse_args()

    dirs = Playbook.run_dirs_under(*args.roots)
    pb = Playbook.build(dirs, conditions=None if args.all_conditions else COACHED_CONDITIONS)
    out = Path(args.out)
    pb.save(out)
    n_coached = sum(pb.coached(c) for c in pb.cases)
    print(f"{out}: {len(pb.cases)} runs ({n_coached} coached) from {len(dirs)} run dirs")
    if args.show:
        print()
        print(pb.evidence(args.show, args.success))
        stalled = pb.stalled_text()
        if stalled:
            print()
            print(stalled)


if __name__ == "__main__":
    main()
