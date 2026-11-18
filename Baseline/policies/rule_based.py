"""Rule-based baseline policy.

Applies deterministic, inspectable heuristic rules considering fatigue, stress,
equipment availability, time limits, and target RPE.
"""

from typing import Dict, Any, List
from Baseline.environment.schemas import UserState
from Baseline.policies.base import BasePolicy
from Baseline.environment.exercises import get_exercises_for_equipment, ExerciseDefinition

class RuleBasedPolicy(BasePolicy):
    def __init__(self, name: str = "RuleBased"):
        super().__init__(name=name)

    def recommend(self, state: UserState) -> Dict[str, Any]:
        """Generate recommendation using explicit adaptive heuristic rules."""
        avail_exercises = get_exercises_for_equipment(state.available_equipment)
        avail_time = state.available_time_min

        # Rule 1: High Fatigue or High Stress -> Active Recovery / Mobility
        if state.fatigue >= 0.70 or state.stress >= 0.75:
            duration = min(30, avail_time)
            recovery_exercises = [
                ex for ex in avail_exercises if ex.category in ["flexibility", "recovery"]
            ]
            if not recovery_exercises:
                # Default bodyweight stretch
                chosen = [{"name": "Yoga Mobility Sequence", "category": "flexibility", "equipment": "bodyweight", "duration_min": duration}]
            else:
                chosen = [
                    {
                        "name": recovery_exercises[0].name,
                        "category": recovery_exercises[0].category,
                        "equipment": recovery_exercises[0].required_equipment,
                        "duration_min": duration,
                    }
                ]
            return {
                "workout_type": "recovery",
                "target_intensity": 3.0,
                "estimated_duration_min": duration,
                "volume": "low",
                "exercises": chosen,
                "rationale": (
                    f"Rule: Fatigue ({state.fatigue:.2f}) or Stress ({state.stress:.2f}) is elevated. "
                    f"Prescribed gentle recovery to reduce cumulative fatigue and prevent strain."
                ),
            }

        # Rule 2: Short Available Time (< 25 min) -> Compact Circuit
        if avail_time < 25:
            duration = avail_time
            short_pool = [ex for ex in avail_exercises if ex.category in ["hiit", "strength_upper", "strength_lower"]]
            if not short_pool:
                short_pool = avail_exercises
            selected = short_pool[:2]
            chosen = []
            for ex in selected:
                chosen.append({
                    "name": ex.name,
                    "category": ex.category,
                    "equipment": ex.required_equipment,
                    "sets": 3,
                    "reps": 10,
                })
            target_intensity = min(7.0, max(4.0, state.target_rpe))
            return {
                "workout_type": "hiit" if "hiit" in [e.category for e in selected] else "strength",
                "target_intensity": target_intensity,
                "estimated_duration_min": duration,
                "volume": "low",
                "exercises": chosen,
                "rationale": (
                    f"Rule: Limited time window ({avail_time} min). "
                    f"Prescribed high-efficiency compact circuit matching available equipment."
                ),
            }

        # Rule 3: Goal & Preference Alignment with Available Equipment
        preferred = [p.lower() for p in state.preferred_categories]
        dispreferred = [d.lower() for d in state.dispreferred_categories]

        if state.goal == "endurance" and "endurance" not in dispreferred:
            workout_type = "endurance"
            candidates = [ex for ex in avail_exercises if ex.category == "endurance"]
        elif state.goal == "strength" and "strength" not in dispreferred:
            workout_type = "strength"
            candidates = [ex for ex in avail_exercises if ex.category.startswith("strength")]
        elif "hiit" in preferred and "hiit" not in dispreferred:
            workout_type = "hiit"
            candidates = [ex for ex in avail_exercises if ex.category == "hiit"]
        else:
            workout_type = "strength" if "strength" in preferred else "endurance"
            candidates = [ex for ex in avail_exercises if workout_type in ex.category]

        if not candidates:
            candidates = avail_exercises

        duration = min(45, avail_time)
        target_intensity = max(1.0, min(10.0, state.target_rpe))
        volume = "moderate" if duration <= 45 else "high"

        # Pick 2-3 suitable exercises from candidates
        picked_defs = candidates[:3]
        chosen = []
        for ex in picked_defs:
            item: Dict[str, Any] = {
                "name": ex.name,
                "category": ex.category,
                "equipment": ex.required_equipment,
            }
            if ex.default_duration_min is not None:
                item["duration_min"] = min(ex.default_duration_min, duration // max(1, len(picked_defs)))
            else:
                item["sets"] = ex.default_sets or 3
                item["reps"] = ex.default_reps or 10
            chosen.append(item)

        return {
            "workout_type": workout_type,
            "target_intensity": target_intensity,
            "estimated_duration_min": duration,
            "volume": volume,
            "exercises": chosen,
            "rationale": (
                f"Rule: Aligned with goal '{state.goal}', target RPE {state.target_rpe:.1f}, "
                f"and available equipment {state.available_equipment} within {duration} min."
            ),
        }
