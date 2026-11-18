"""Tests for synthetic user generation and reproducibility."""

from Baseline.environment.synthetic_users import generate_synthetic_scenario, generate_scenarios

def test_reproducibility():
    s1 = generate_synthetic_scenario(seed=777)
    s2 = generate_synthetic_scenario(seed=777)
    assert s1.user_id == s2.user_id
    assert s1.fitness_level == s2.fitness_level
    assert s1.goal == s2.goal
    assert s1.available_time_min == s2.available_time_min
    assert s1.available_equipment == s2.available_equipment
    assert s1.fatigue == s2.fatigue
    assert s1.target_rpe == s2.target_rpe

def test_distinct_seeds():
    s1 = generate_synthetic_scenario(seed=101)
    s2 = generate_synthetic_scenario(seed=102)
    assert s1.user_id != s2.user_id or s1.fatigue != s2.fatigue

def test_bounds():
    scenarios = generate_scenarios(start_seed=500, count=50)
    for s in scenarios:
        assert 0.0 <= s.fatigue <= 1.0
        assert 0.0 <= s.stress <= 1.0
        assert 1.0 <= s.target_rpe <= 10.0
        assert 10 <= s.available_time_min <= 120
        assert "bodyweight" in s.available_equipment
