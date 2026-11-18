"""Tests for evaluation metrics and aggregation."""

from Baseline.evaluation.metrics import ScenarioResult, calculate_aggregate_metrics

def test_aggregate_metrics_calculation():
    r1 = ScenarioResult(
        scenario_idx=1,
        seed=1001,
        user_id="u1",
        policy_name="TestPolicy",
        is_valid=True,
        violations_count=0,
        violations=[],
        reward=1.2,
        completion=1.0,
        satisfaction=0.8,
        target_rpe=6.0,
        actual_rpe=6.2,
        rpe_error=0.2,
        fatigue_penalty=0.1,
        prescribed_duration_min=30,
        available_time_min=30,
        duration_exceeded=False,
        latency_ms=5.0,
        workout_type="endurance",
        rationale="ok",
    )
    r2 = ScenarioResult(
        scenario_idx=2,
        seed=1002,
        user_id="u2",
        policy_name="TestPolicy",
        is_valid=False,
        violations_count=1,
        violations=["Duration exceeded"],
        reward=0.2,
        completion=0.6,
        satisfaction=0.4,
        target_rpe=7.0,
        actual_rpe=8.0,
        rpe_error=1.0,
        fatigue_penalty=0.3,
        prescribed_duration_min=45,
        available_time_min=30,
        duration_exceeded=True,
        latency_ms=7.0,
        workout_type="strength",
        rationale="violation",
    )

    agg = calculate_aggregate_metrics([r1, r2])
    assert agg is not None
    assert agg.num_scenarios == 2
    assert agg.mean_reward == 0.7  # (1.2 + 0.2)/2
    assert agg.validity_rate_pct == 50.0
    assert agg.constraint_violation_rate_pct == 50.0
    assert agg.pct_exceeding_available_time == 50.0
    assert agg.mean_violations_per_rec == 0.5
