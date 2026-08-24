"""LLM-integrated feedback pipeline (thesis contribution #2).

Flow: collector (human NL feedback) -> translator (LLM -> structured JSON ->
numeric reward components) -> reward_model (learned preference reward) ->
rewards.hybrid.HybridReward (weighted combination).
"""
