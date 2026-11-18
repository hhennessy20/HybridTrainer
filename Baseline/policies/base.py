"""Base policy interface and evaluation wrapper."""

import time
from typing import Dict, Any, Tuple
from Baseline.environment.schemas import UserState, ValidationResult, validate_workout_action
from Baseline.environment.reward import FeedbackResult, compute_simulated_feedback

class BasePolicy:
    def __init__(self, name: str):
        self.name = name

    def recommend(self, state: UserState) -> Dict[str, Any]:
        """Generate a recommendation dictionary for the given user state."""
        raise NotImplementedError

    def evaluate_on_state(
        self, state: UserState
    ) -> Tuple[Dict[str, Any], ValidationResult, FeedbackResult, float]:
        """Execute recommendation, time latency, validate action, and compute simulated feedback."""
        t0 = time.perf_counter()
        raw_action = self.recommend(state)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        val_result = validate_workout_action(raw_action, state)
        feedback = compute_simulated_feedback(state, raw_action, val_result)

        return raw_action, val_result, feedback, latency_ms
