"""LLM feedback pipeline: schema validation, poisoning guard, feedback store."""

import json

from quadruped_rl.llm_feedback.collector import FeedbackStore
from quadruped_rl.llm_feedback.schemas import LLMRewardOutput
from quadruped_rl.llm_feedback.translator import LLMRewardScorer, _extract_json


class FakeClient:
    def __init__(self, response: str):
        self.response = response

    def complete(self, system, user, max_tokens=1024):
        return self.response


LLM_CFG = {"provider": "anthropic", "model": "test", "cache": True}

VALID = json.dumps(
    {
        "overall_score": 0.6,
        "components": [
            {
                "concept": "gait smoothness",
                "score": 0.7,
                "confidence": 0.9,
                "rationale": "steady contacts",
            }
        ],
        "safety_concern": False,
    }
)


def test_valid_response_scored_and_cached():
    scorer = LLMRewardScorer(LLM_CFG, client=FakeClient(f"Here you go: {VALID}"))
    summary = {"terrain": "stairs", "metrics": {"v": 1.0}}
    assert scorer.score(summary) == 0.6
    scorer.client = FakeClient("GARBAGE")  # cache must serve the repeat call
    assert scorer.score(summary) == 0.6
    assert scorer.discarded == 0


def test_malformed_response_discarded():
    scorer = LLMRewardScorer(LLM_CFG, client=FakeClient("not json at all"))
    assert scorer.score({"terrain": "mud", "metrics": {}}) == 0.0
    assert scorer.discarded == 1


def test_out_of_range_score_discarded():
    bad = VALID.replace("0.6", "5.0")  # overall_score out of [-1, 1]
    scorer = LLMRewardScorer(LLM_CFG, client=FakeClient(bad))
    assert scorer.score({"terrain": "gap", "metrics": {}}) == 0.0
    assert scorer.discarded == 1


def test_extract_json():
    obj = _extract_json('prefix {"a": 1} suffix')
    assert obj == {"a": 1}


def test_schema_bounds():
    parsed = LLMRewardOutput.model_validate(json.loads(VALID))
    assert -1.0 <= parsed.overall_score <= 1.0


def test_feedback_store_roundtrip(tmp_path):
    store = FeedbackStore(root=tmp_path)
    store.add(
        "expert",
        "structured",
        situation="계단 상행",
        behavior="앞발을 낮게 끎",
        assessment="발끝을 더 들어야 함",
    )
    store.add("non_expert", "free_form", free_text="로봇이 자주 미끄러져요")
    entries = store.load_all()
    assert len(entries) == 2
    snippets = store.as_snippets()
    assert any("계단 상행" in s for s in snippets)
