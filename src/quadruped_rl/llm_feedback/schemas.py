"""Structured schemas for feedback and LLM outputs.

Every LLM response is validated against LLMRewardOutput before use.
Validation failure => discard + log (never let malformed output pollute
the reward signal). See CLAUDE.md.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FeedbackEntry(BaseModel):
    """One anonymized human feedback item (IRB: no personal identifiers)."""

    feedback_id: str
    source_group: Literal["expert", "non_expert"]
    mode: Literal["structured", "free_form", "realtime"]
    # structured template: "로봇이 [situation]에서 [behavior]할 때 [assessment]"
    situation: str | None = None
    behavior: str | None = None
    assessment: str | None = None
    free_text: str | None = None
    video_ref: str | None = None  # trajectory/video id being assessed
    timestamp: float


class RewardComponentScore(BaseModel):
    """One reward concept extracted by the LLM, mapped to a numeric score."""

    concept: str = Field(description="e.g. 'foot placement caution', 'gait smoothness'")
    score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class LLMRewardOutput(BaseModel):
    """Validated LLM response for one trajectory segment evaluation."""

    overall_score: float = Field(ge=-1.0, le=1.0)
    components: list[RewardComponentScore]
    safety_concern: bool = False


class PreferencePair(BaseModel):
    """Trajectory pair with human preference label (for reward model training)."""

    trajectory_a: str
    trajectory_b: str
    preferred: Literal["a", "b", "equal"]
    source: Literal["human", "llm"]


class CoachAction(BaseModel):
    """One reward-parameter change proposed by the LLM coach."""

    param: str = Field(description="flat key, e.g. 'energy.weight'")
    value: float
    rationale: str = ""


class CoachOutput(BaseModel):
    """Validated LLM coach response (llm_feedback/coach.py).

    Bounds, sign locks and step-size limits are enforced afterwards by the
    ParamSpace guardrail — schema validation only guarantees structure.
    """

    diagnosis: str
    actions: list[CoachAction] = Field(default_factory=list)
    expected_effect: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
