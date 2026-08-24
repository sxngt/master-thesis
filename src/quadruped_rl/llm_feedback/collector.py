"""Human feedback collection & storage (IRB-compliant).

Protocol (docs/feedback_protocol.md): expert group (n=10 robotics
researchers) + non-expert group (n=20). Three modes: structured template,
free-form post-video, realtime observation comments.
All entries are anonymized before storage (source_group only, no identity).
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from quadruped_rl.llm_feedback.schemas import FeedbackEntry

FEEDBACK_DIR = Path(__file__).resolve().parents[3] / "data" / "feedback"


class FeedbackStore:
    def __init__(self, root: str | Path = FEEDBACK_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "feedback.jsonl"

    def add(
        self,
        source_group: str,
        mode: str,
        *,
        situation: str | None = None,
        behavior: str | None = None,
        assessment: str | None = None,
        free_text: str | None = None,
        video_ref: str | None = None,
    ) -> FeedbackEntry:
        entry = FeedbackEntry(
            feedback_id=uuid.uuid4().hex,
            source_group=source_group,
            mode=mode,
            situation=situation,
            behavior=behavior,
            assessment=assessment,
            free_text=free_text,
            video_ref=video_ref,
            timestamp=time.time(),
        )
        with open(self.path, "a") as f:
            f.write(entry.model_dump_json() + "\n")
        return entry

    def load_all(self) -> list[FeedbackEntry]:
        if not self.path.exists():
            return []
        return [
            FeedbackEntry.model_validate(json.loads(line))
            for line in self.path.read_text().splitlines()
            if line.strip()
        ]

    def as_snippets(self) -> list[str]:
        """Render entries as plain-text snippets for LLM context."""
        out = []
        for e in self.load_all():
            if e.mode == "structured":
                out.append(f"[{e.source_group}] {e.situation} / {e.behavior} / {e.assessment}")
            elif e.free_text:
                out.append(f"[{e.source_group}] {e.free_text}")
        return out
