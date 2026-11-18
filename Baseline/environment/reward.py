"""Simulated feedback, reward function, and state transition logic for HybridTrainer.

Implements the specification formula:
    R_t = w_c * C_t + w_s * S_t - w_e * abs(RPE_t - RPE_target_t) - w_f * F_t
"""

from dataclasses import dataclass
from typing import Dict, Any, List
from Baseline.config import (
    WEIGHT_COMPLETION,
    WEIGHT_SATISFACTION,
    WEIGHT_RPE_ERROR,
    WEIGHT_FATIGUE,
)
from Baseline.environment.schemas import UserState, ValidationResult

@dataclass
class FeedbackResult:
    completion: float            # C_t in [0.0, 1.0]
    satisfaction: float          # S_t in [0.0, 1.0]
    actual_rpe: float            # RPE_t in [1.0, 10.0]
    target_rpe: float            # RPE_target_t in [1.0, 10.0]
    rpe_error: float             # abs(RPE_t - RPE_target_t)
    fatigue_penalty: float       # F_t in [0.0, 1.0]
    reward: float                # R_t
    violations_count: int
    is_valid: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "completion": round(self.completion, 4),
            "satisfaction": round(self.satisfaction, 4),
            "actual_rpe": round(self.actual_rpe, 2),
            "target_rpe": round(self.target_rpe, 2),
            "rpe_error": round(self.rpe_error, 2),
            "fatigue_penalty": round(self.fatigue_penalty, 4),
            "reward": round(self.reward, 4),
            "violations_count": self.violations_count,
            "is_valid": self.is_valid,
        }

def compute_simulated_feedback(
    state: UserState,
    action_dict: Dict[str, Any],
    validation: ValidationResult,
    wc: float = WEIGHT_COMPLETION,
    ws: float = WEIGHT_SATISFACTION,
    we: float = WEIGHT_RPE_ERROR,
    wf: float = WEIGHT_FATIGUE,
) -> FeedbackResult:
    """Compute transparent, reproducible feedback and reward for a recommendation."""
    violations_count = len(validation.violations)
    is_valid = validation.is_valid

    # Extract action parameters
    norm = validation.normalized_action or action_dict
    workout_type = str(norm.get("workout_type", "general")).lower()
    target_intensity = float(norm.get("target_intensity", 5.0))
    est_duration = float(norm.get("estimated_duration_min", state.available_time_min))

    # 1. Completion / Adherence (C_t)
    completion = 1.0

    # Duration feasibility penalty
    if est_duration > state.available_time_min:
        ratio = state.available_time_min / max(1.0, est_duration)
        completion *= max(0.2, ratio)

    # Equipment mismatch penalty
    for v in validation.violations:
        if "requires" in v.lower() and "equipment" in v.lower():
            completion -= 0.20
    completion = max(0.1, completion)

    # High fatigue vs high intensity strain on completion
    if target_intensity > 7.0 and state.fatigue > 0.65:
        fatigue_strain = (state.fatigue - 0.65) / 0.35
        completion -= 0.25 * fatigue_strain

    # Preference impact on adherence
    if any(p in workout_type for p in state.preferred_categories):
        completion += 0.05
    if any(d in workout_type for d in state.dispreferred_categories):
        completion -= 0.15

    completion = max(0.0, min(1.0, completion))

    # 2. Actual reported RPE (RPE_t)
    fitness_shift = {"beginner": 0.5, "intermediate": 0.0, "advanced": -0.5}.get(state.fitness_level, 0.0)
    actual_rpe = target_intensity + 1.5 * state.fatigue + 0.5 * state.stress + fitness_shift
    actual_rpe = max(1.0, min(10.0, actual_rpe))
    rpe_error = abs(actual_rpe - state.target_rpe)

    # 3. Satisfaction (S_t)
    satisfaction = 0.4 * completion
    if any(p in workout_type for p in state.preferred_categories):
        satisfaction += 0.30
    elif any(d in workout_type for d in state.dispreferred_categories):
        satisfaction -= 0.20

    # RPE alignment contribution
    rpe_alignment = max(0.0, 1.0 - (rpe_error / 4.0))
    satisfaction += 0.30 * rpe_alignment

    # Recovery satisfaction bonus when exhausted
    if state.fatigue > 0.70 and workout_type in ["recovery", "flexibility"]:
        satisfaction += 0.15

    satisfaction = max(0.0, min(1.0, satisfaction))

    # 4. Fatigue / Stress Penalty (F_t)
    if workout_type in ["recovery", "flexibility"] or target_intensity <= 3.5:
        fatigue_penalty = 0.0
    else:
        strain = ((state.fatigue + state.stress) / 2.0) * (target_intensity / 10.0)
        fatigue_penalty = max(0.0, min(1.0, (strain - 0.25) * 1.5))

    # 5. Composite Reward (R_t)
    reward = (wc * completion) + (ws * satisfaction) - (we * rpe_error) - (wf * fatigue_penalty)

    # Additional penalty for hard constraint violations
    if not is_valid:
        reward -= 0.25 * violations_count

    return FeedbackResult(
        completion=completion,
        satisfaction=satisfaction,
        actual_rpe=actual_rpe,
        target_rpe=state.target_rpe,
        rpe_error=rpe_error,
        fatigue_penalty=fatigue_penalty,
        reward=reward,
        violations_count=violations_count,
        is_valid=is_valid,
    )

def step_state(state: UserState, action_dict: Dict[str, Any], feedback: FeedbackResult) -> UserState:
    """Transition state to next sequential workout day based on action and feedback."""
    norm = action_dict
    workout_type = str(norm.get("workout_type", "general")).lower()
    target_intensity = float(norm.get("target_intensity", 5.0))

    # Fatigue update
    if workout_type in ["recovery", "flexibility"] or target_intensity < 4.0:
        new_fatigue = max(0.1, state.fatigue - 0.25)
    else:
        new_fatigue = min(0.95, state.fatigue + 0.05 * (target_intensity / 5.0))

    # Stress update
    if feedback.satisfaction >= 0.7:
        new_stress = max(0.1, state.stress - 0.10)
    elif feedback.fatigue_penalty > 0.5:
        new_stress = min(0.95, state.stress + 0.10)
    else:
        new_stress = state.stress

    # Completion rate update (EMA)
    new_completion = 0.7 * state.recent_completion_rate + 0.3 * feedback.completion

    return UserState(
        user_id=state.user_id,
        seed=state.seed + 1,
        fitness_level=state.fitness_level,
        goal=state.goal,
        available_time_min=state.available_time_min,
        available_equipment=list(state.available_equipment),
        preferred_categories=list(state.preferred_categories),
        dispreferred_categories=list(state.dispreferred_categories),
        fatigue=round(new_fatigue, 3),
        stress=round(new_stress, 3),
        recent_completion_rate=round(new_completion, 3),
        target_rpe=state.target_rpe,
        restrictions=list(state.restrictions),
        step=state.step + 1,
    )
