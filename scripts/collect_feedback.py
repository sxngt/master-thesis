#!/usr/bin/env python
"""Interactive CLI for recording human feedback (structured or free-form).

Used during Phase 3 feedback sessions. Entries are stored anonymized
(source group only) in data/feedback/feedback.jsonl per the IRB protocol.
"""

import argparse

from quadruped_rl.llm_feedback.collector import FeedbackStore


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", required=True, choices=["expert", "non_expert"])
    p.add_argument("--mode", required=True, choices=["structured", "free_form", "realtime"])
    p.add_argument("--video-ref", default=None)
    args = p.parse_args()

    store = FeedbackStore()
    print("Enter feedback (empty line to finish session):")
    while True:
        if args.mode == "structured":
            situation = input("상황 (situation): ").strip()
            if not situation:
                break
            behavior = input("행동 (behavior): ").strip()
            assessment = input("평가 (assessment): ").strip()
            e = store.add(
                args.group,
                args.mode,
                situation=situation,
                behavior=behavior,
                assessment=assessment,
                video_ref=args.video_ref,
            )
        else:
            text = input("> ").strip()
            if not text:
                break
            e = store.add(args.group, args.mode, free_text=text, video_ref=args.video_ref)
        print(f"  saved {e.feedback_id}")


if __name__ == "__main__":
    main()
