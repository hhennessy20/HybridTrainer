"""Self-contained automated smoke test suite for HybridTrainer."""

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Baseline.environment.schemas import UserState, validate_workout_action
from Baseline.environment.synthetic_users import generate_synthetic_scenario, generate_scenarios
from Baseline.environment.reward import compute_simulated_feedback, step_state
from Baseline.policies.static_schedule import StaticSchedulePolicy
from Baseline.policies.rule_based import RuleBasedPolicy
from Baseline.policies.rl_qlearning import TabularQLearningPolicy
from Baseline.policies.llm_ollama import OllamaLLMPolicy
from Baseline.evaluation.metrics import ScenarioResult, calculate_aggregate_metrics

def run_all_smoke_tests() -> bool:
    print("Running HybridTrainer smoke tests:")
    tests_passed = 0
    total_tests = 0

    def check(name: str, fn):
        nonlocal tests_passed, total_tests
        total_tests += 1
        t0 = time.perf_counter()
        try:
            fn()
            el = (time.perf_counter() - t0) * 1000.0
            print(f" [PASS] {name:<48} ({el:.1f}ms)")
            tests_passed += 1
        except Exception as e:
            el = (time.perf_counter() - t0) * 1000.0
            print(f" [FAIL] {name:<48} ({el:.1f}ms) -> ERROR: {e}")

    # 1. Schemas & Validation
    def test_schema():
        s = generate_synthetic_scenario(seed=1)
        valid_act = {
            "workout_type": "endurance",
            "target_intensity": 5.0,
            "estimated_duration_min": s.available_time_min,
            "volume": "moderate",
            "exercises": [{"name": "Outdoor Running", "equipment": "bodyweight", "duration_min": s.available_time_min}],
        }
        res = validate_workout_action(valid_act, s)
        assert res.is_valid, f"Validation failed: {res.violations}"
    check("1. Shared Schema & Action Validation", test_schema)

    # 2. Synthetic Scenario Generator
    def test_synth():
        s1 = generate_synthetic_scenario(seed=42)
        s2 = generate_synthetic_scenario(seed=42)
        assert s1.user_id == s2.user_id
        assert s1.available_time_min == s2.available_time_min
        assert 0.0 <= s1.fatigue <= 1.0
        scens = generate_scenarios(start_seed=100, count=10)
        assert len(scens) == 10
    check("2. Synthetic User Generator & Reproducibility", test_synth)

    # 3. Reward & Feedback
    def test_reward():
        s = generate_synthetic_scenario(seed=42)
        act = {
            "workout_type": "endurance",
            "target_intensity": s.target_rpe,
            "estimated_duration_min": s.available_time_min,
            "volume": "moderate",
            "exercises": [{"name": "Outdoor Running", "equipment": "bodyweight", "duration_min": s.available_time_min}],
        }
        val = validate_workout_action(act, s)
        fb = compute_simulated_feedback(s, act, val)
        assert fb.completion > 0.0
        assert fb.satisfaction > 0.0
        assert fb.reward > 0.0
        next_s = step_state(s, act, fb)
        assert next_s.step == 1
    check("3. Simulated Feedback, Reward & Transitions", test_reward)

    # 4. Static Schedule Baseline
    def test_static():
        pol = StaticSchedulePolicy()
        s = generate_synthetic_scenario(seed=42)
        act, val, fb, _ = pol.evaluate_on_state(s)
        assert "workout_type" in act
    check("4. Static Schedule Policy", test_static)

    # 5. Rule-Based Baseline
    def test_rule():
        pol = RuleBasedPolicy()
        s = generate_synthetic_scenario(seed=42)
        act, val, fb, _ = pol.evaluate_on_state(s)
        assert val.is_valid, f"Rule-based action invalid: {val.violations}"
    check("5. Rule-Based Adaptive Policy", test_rule)

    # 6. Tabular RL Baseline
    def test_rl():
        pol = TabularQLearningPolicy()
        s = generate_synthetic_scenario(seed=42)
        act, val, fb, _ = pol.evaluate_on_state(s)
        assert val.is_valid, f"RL action invalid: {val.violations}"
    check("6. Tabular Q-Learning Policy", test_rl)

    # 7. Local Ollama LLM Baseline
    def test_llm():
        llm = OllamaLLMPolicy()
        avail, msg = llm.is_available()
        if not avail:
            print(f"      (Ollama offline/model missing: {msg})")
            return
        s = generate_synthetic_scenario(seed=42)
        act, val, fb, lat = llm.evaluate_on_state(s)
        assert "workout_type" in act
        assert lat > 0.0
    check("7. Local Ollama LLM Policy (qwen3.5:0.8b)", test_llm)

    # 8. Evaluation Metrics Calculation
    def test_metrics():
        r1 = ScenarioResult(
            scenario_idx=1,
            seed=1,
            user_id="u1",
            policy_name="P1",
            is_valid=True,
            violations_count=0,
            violations=[],
            reward=1.0,
            completion=1.0,
            satisfaction=0.8,
            target_rpe=6.0,
            actual_rpe=6.0,
            rpe_error=0.0,
            fatigue_penalty=0.1,
            prescribed_duration_min=30,
            available_time_min=30,
            duration_exceeded=False,
            latency_ms=10.0,
            workout_type="endurance",
            rationale="ok",
        )
        agg = calculate_aggregate_metrics([r1])
        assert agg is not None
        assert agg.mean_reward == 1.0
        assert agg.validity_rate_pct == 100.0
    check("8. Evaluation Metrics Calculation", test_metrics)

    print(f"Results: {tests_passed}/{total_tests} smoke tests passed.")
    return tests_passed == total_tests

if __name__ == "__main__":
    success = run_all_smoke_tests()
    sys.exit(0 if success else 1)
