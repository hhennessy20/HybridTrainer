"""Tests for simulated feedback, reward formula, and state transition."""

from Baseline.environment.schemas import UserState, validate_workout_action
from Baseline.environment.reward import compute_simulated_feedback, step_state

def test_reward_formula_components():
    state = UserState(
        user_id="u_reward_01",
        seed=1,
        fitness_level="intermediate",
        goal="endurance",
        available_time_min=30,
        available_equipment=["bodyweight"],
        preferred_categories=["endurance"],
        dispreferred_categories=[],
        fatigue=0.2,
        stress=0.2,
        recent_completion_rate=1.0,
        target_rpe=6.0,
    )
    action = {
        "workout_type": "endurance",
        "target_intensity": 6.0,
        "estimated_duration_min": 30,
        "volume": "moderate",
        "exercises": [
            {"name": "Outdoor Running", "category": "endurance", "equipment": "bodyweight", "duration_min": 30}
        ],
    }
    val = validate_workout_action(action, state)
    fb = compute_simulated_feedback(state, action, val)

    assert 0.0 <= fb.completion <= 1.0
    assert 0.0 <= fb.satisfaction <= 1.0
    assert 0.0 <= fb.fatigue_penalty <= 1.0
    assert fb.actual_rpe >= 1.0
    # Reward should be positive for well-aligned feasible workout
    assert fb.reward > 0.5

def test_overload_penalty():
    state = UserState(
        user_id="u_reward_02",
        seed=2,
        fitness_level="beginner",
        goal="strength",
        available_time_min=45,
        available_equipment=["bodyweight", "barbell"],
        preferred_categories=["strength"],
        dispreferred_categories=[],
        fatigue=0.85,  # Very high fatigue
        stress=0.80,   # Very high stress
        recent_completion_rate=0.7,
        target_rpe=4.0,
    )
    # Action prescribes very heavy workout despite exhaustion
    heavy_action = {
        "workout_type": "strength",
        "target_intensity": 9.0,
        "estimated_duration_min": 45,
        "volume": "high",
        "exercises": [
            {"name": "Barbell Bench Press", "category": "strength_upper", "equipment": "barbell", "sets": 5, "reps": 10}
        ],
    }
    val = validate_workout_action(heavy_action, state)
    fb_heavy = compute_simulated_feedback(state, heavy_action, val)

    # Action prescribes recovery
    recovery_action = {
        "workout_type": "recovery",
        "target_intensity": 3.0,
        "estimated_duration_min": 25,
        "volume": "low",
        "exercises": [
            {"name": "Yoga Mobility Sequence", "category": "flexibility", "equipment": "bodyweight", "duration_min": 25}
        ],
    }
    val_rec = validate_workout_action(recovery_action, state)
    fb_rec = compute_simulated_feedback(state, recovery_action, val_rec)

    # Recovery should yield significantly higher reward and lower fatigue penalty
    assert fb_rec.fatigue_penalty < fb_heavy.fatigue_penalty
    assert fb_rec.reward > fb_heavy.reward

def test_state_transition():
    state = UserState(
        user_id="u_step_01",
        seed=1,
        fitness_level="intermediate",
        goal="strength",
        available_time_min=40,
        available_equipment=["bodyweight"],
        preferred_categories=["strength"],
        dispreferred_categories=[],
        fatigue=0.7,
        stress=0.6,
        recent_completion_rate=0.8,
        target_rpe=5.0,
        step=0,
    )
    rec_action = {
        "workout_type": "recovery",
        "target_intensity": 3.0,
        "estimated_duration_min": 20,
        "volume": "low",
        "exercises": [{"name": "Yoga Mobility Sequence", "category": "flexibility", "equipment": "bodyweight", "duration_min": 20}],
    }
    val = validate_workout_action(rec_action, state)
    fb = compute_simulated_feedback(state, rec_action, val)
    next_s = step_state(state, rec_action, fb)

    assert next_s.step == 1
    # Fatigue should reduce following recovery session
    assert next_s.fatigue < state.fatigue
