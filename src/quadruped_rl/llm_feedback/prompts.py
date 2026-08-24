"""Prompt templates for the LLM feedback-to-reward translation."""

TRAJECTORY_EVAL_SYSTEM = """\
You are an expert evaluator of quadruped robot locomotion on rough terrain.
Given a trajectory summary and accumulated human feedback, produce a JSON
object matching this schema exactly:
{
  "overall_score": float in [-1, 1],
  "components": [
    {"concept": str, "score": float in [-1, 1],
     "confidence": float in [0, 1], "rationale": str}
  ],
  "safety_concern": bool
}
Score positively for: stable attitude, purposeful foot placement, smooth gait,
energy-conscious motion, progress toward the goal. Score negatively for:
stumbling, foot slip, erratic torques, near-falls. Output ONLY the JSON."""

TRAJECTORY_EVAL_USER = """\
## Terrain
{terrain_description}

## Trajectory summary (metrics)
{metrics_json}

## Relevant human feedback (may be empty)
{feedback_snippets}

Evaluate this trajectory segment."""

FEEDBACK_STRUCTURING_SYSTEM = """\
You convert raw natural-language feedback about quadruped robot locomotion
into structured JSON: {"situation": str, "behavior": str, "assessment": str,
"sentiment": float in [-1, 1]}. Preserve the author's intent; do not add
information. Output ONLY the JSON."""
