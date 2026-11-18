"""Tests for baseline policies."""

from Baseline.environment.synthetic_users import generate_synthetic_scenario
from Baseline.policies.static_schedule import StaticSchedulePolicy
from Baseline.policies.rule_based import RuleBasedPolicy
from Baseline.policies.rl_qlearning import TabularQLearningPolicy
from Baseline.policies.llm_ollama import OllamaLLMPolicy

def test_static_schedule_unadapting():
    pol = StaticSchedulePolicy()
    s = generate_synthetic_scenario(seed=42)
    # Day 0 action
    act0 = pol.recommend(s)
    assert act0["workout_type"] == "strength"
    # Same recommendation irrespective of changing fatigue
    s_tired = generate_synthetic_scenario(seed=42)
    s_tired.fatigue = 0.95
    act_tired = pol.recommend(s_tired)
    assert act_tired["target_intensity"] == act0["target_intensity"]
    assert act_tired["estimated_duration_min"] == act0["estimated_duration_min"]

def test_rule_based_recovery_adaptation():
    pol = RuleBasedPolicy()
    # High fatigue user
    s = generate_synthetic_scenario(seed=50)
    s.fatigue = 0.85
    act = pol.recommend(s)
    assert act["workout_type"] == "recovery"
    assert act["target_intensity"] <= 4.0
    assert "recovery" in act["rationale"].lower() or "fatigue" in act["rationale"].lower()

def test_rule_based_time_adaptation():
    pol = RuleBasedPolicy()
    s = generate_synthetic_scenario(seed=51)
    s.fatigue = 0.2
    s.stress = 0.2
    s.available_time_min = 20
    act = pol.recommend(s)
    assert act["estimated_duration_min"] <= 20

def test_tabular_rl_action_mapping():
    pol = TabularQLearningPolicy()
    s = generate_synthetic_scenario(seed=52)
    act = pol.recommend(s)
    assert "workout_type" in act
    assert "exercises" in act
    assert act["estimated_duration_min"] <= s.available_time_min

def test_ollama_defensive_parsing():
    raw_valid_json = '{"workout_type": "strength", "target_intensity": 7.0, "estimated_duration_min": 30, "volume": "moderate", "exercises": [], "rationale": "ok"}'
    parsed, err = OllamaLLMPolicy.extract_json(raw_valid_json)
    assert parsed is not None
    assert err is None
    assert parsed["workout_type"] == "strength"

    # Markdown wrapped
    raw_md = '```json\n{"workout_type": "endurance", "target_intensity": 6.0, "estimated_duration_min": 20, "volume": "low", "exercises": [], "rationale": "run"}\n```'
    parsed_md, err_md = OllamaLLMPolicy.extract_json(raw_md)
    assert parsed_md is not None
    assert parsed_md["workout_type"] == "endurance"

    # Embedded in conversational text
    raw_mixed = 'Here is your recommendation:\n{"workout_type": "hiit", "target_intensity": 8.0, "estimated_duration_min": 25, "volume": "high", "exercises": [], "rationale": "hiit"}\nHope this helps!'
    parsed_mixed, err_mixed = OllamaLLMPolicy.extract_json(raw_mixed)
    assert parsed_mixed is not None
    assert parsed_mixed["workout_type"] == "hiit"
