"""Tests for environment schemas and common action validator."""

import pytest
from Baseline.environment.schemas import (
    UserState,
    validate_workout_action,
)

def test_user_state_serialization():
    state = UserState(
        user_id="test_01",
        seed=123,
        fitness_level="intermediate",
        goal="strength",
        available_time_min=45,
        available_equipment=["bodyweight", "dumbbells"],
        preferred_categories=["strength"],
        dispreferred_categories=["hiit"],
        fatigue=0.3,
        stress=0.2,
        recent_completion_rate=0.85,
        target_rpe=7.0,
    )
    d = state.to_dict()
    assert d["user_id"] == "test_01"
    assert d["available_time_min"] == 45
    state2 = UserState.from_dict(d)
    assert state2.user_id == state.user_id
    assert state2.target_rpe == 7.0

def test_action_validation_valid():
    state = UserState(
        user_id="test_02",
        seed=124,
        fitness_level="beginner",
        goal="endurance",
        available_time_min=30,
        available_equipment=["bodyweight"],
        preferred_categories=["endurance"],
        dispreferred_categories=[],
        fatigue=0.2,
        stress=0.1,
        recent_completion_rate=1.0,
        target_rpe=5.0,
    )
    action = {
        "workout_type": "endurance",
        "target_intensity": 5.0,
        "estimated_duration_min": 30,
        "volume": "moderate",
        "exercises": [
            {"name": "Outdoor Running", "category": "endurance", "equipment": "bodyweight", "duration_min": 30}
        ],
        "rationale": "Valid outdoor run.",
    }
    res = validate_workout_action(action, state)
    assert res.is_valid is True
    assert len(res.violations) == 0

def test_action_validation_missing_equipment():
    state = UserState(
        user_id="test_03",
        seed=125,
        fitness_level="beginner",
        goal="strength",
        available_time_min=45,
        available_equipment=["bodyweight"],  # No barbell
        preferred_categories=["strength"],
        dispreferred_categories=[],
        fatigue=0.2,
        stress=0.1,
        recent_completion_rate=1.0,
        target_rpe=6.0,
    )
    action = {
        "workout_type": "strength",
        "target_intensity": 6.0,
        "estimated_duration_min": 40,
        "volume": "moderate",
        "exercises": [
            {"name": "Barbell Bench Press", "category": "strength_upper", "equipment": "barbell", "sets": 3, "reps": 10}
        ],
    }
    res = validate_workout_action(action, state)
    assert res.is_valid is False
    assert any("barbell" in v.lower() for v in res.violations)

def test_action_validation_exceeding_duration():
    state = UserState(
        user_id="test_04",
        seed=126,
        fitness_level="intermediate",
        goal="general_fitness",
        available_time_min=20,  # 20 min available, max with 10% tol is 22 min
        available_equipment=["bodyweight"],
        preferred_categories=[],
        dispreferred_categories=[],
        fatigue=0.2,
        stress=0.2,
        recent_completion_rate=0.9,
        target_rpe=6.0,
    )
    action = {
        "workout_type": "general",
        "target_intensity": 6.0,
        "estimated_duration_min": 45,  # Exceeds 22 min
        "volume": "moderate",
        "exercises": [
            {"name": "Push-ups", "category": "strength_upper", "equipment": "bodyweight", "sets": 3, "reps": 10}
        ],
    }
    res = validate_workout_action(action, state)
    assert res.is_valid is False
    assert any("exceeds available time" in v.lower() for v in res.violations)
