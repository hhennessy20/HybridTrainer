"""Static schedule baseline policy.

Recommends a fixed, unadapting 4-day cycle regardless of user state, fatigue,
time constraints, or available equipment.
"""

from typing import Dict, Any
from Baseline.environment.schemas import UserState
from Baseline.policies.base import BasePolicy

class StaticSchedulePolicy(BasePolicy):
    def __init__(self, name: str = "StaticSchedule"):
        super().__init__(name=name)

        # Predeclared 4-day recurring workout cycle
        self.schedule_cycle = [
            # Day 0: Heavy Barbell Upper Body
            {
                "workout_type": "strength",
                "target_intensity": 8.0,
                "estimated_duration_min": 50,
                "volume": "high",
                "exercises": [
                    {"name": "Barbell Bench Press", "category": "strength_upper", "equipment": "barbell", "sets": 4, "reps": 8},
                    {"name": "Pull-ups", "category": "strength_upper", "equipment": "pullup_bar", "sets": 3, "reps": 8},
                    {"name": "Dumbbell Shoulder Press", "category": "strength_upper", "equipment": "dumbbells", "sets": 3, "reps": 10},
                ],
                "rationale": "Static Day 1: Prescribed heavy upper body barbell and dumbbell session.",
            },
            # Day 1: Moderate Endurance Run
            {
                "workout_type": "endurance",
                "target_intensity": 7.0,
                "estimated_duration_min": 45,
                "volume": "moderate",
                "exercises": [
                    {"name": "Outdoor Running", "category": "endurance", "equipment": "bodyweight", "duration_min": 45},
                ],
                "rationale": "Static Day 2: Prescribed steady-state outdoor endurance run.",
            },
            # Day 2: Heavy Barbell Lower Body
            {
                "workout_type": "strength",
                "target_intensity": 8.0,
                "estimated_duration_min": 50,
                "volume": "high",
                "exercises": [
                    {"name": "Barbell Back Squat", "category": "strength_lower", "equipment": "barbell", "sets": 4, "reps": 8},
                    {"name": "Barbell Romanian Deadlift", "category": "strength_lower", "equipment": "barbell", "sets": 3, "reps": 8},
                    {"name": "Dumbbell Walking Lunges", "category": "strength_lower", "equipment": "dumbbells", "sets": 3, "reps": 12},
                ],
                "rationale": "Static Day 3: Prescribed heavy lower body strength session.",
            },
            # Day 3: Active Recovery & Mobility
            {
                "workout_type": "recovery",
                "target_intensity": 3.0,
                "estimated_duration_min": 30,
                "volume": "low",
                "exercises": [
                    {"name": "Yoga Mobility Sequence", "category": "flexibility", "equipment": "bodyweight", "duration_min": 20},
                    {"name": "Hamstring & Hip Stretches", "category": "flexibility", "equipment": "bodyweight", "duration_min": 10},
                ],
                "rationale": "Static Day 4: Prescribed scheduled recovery and mobility sequence.",
            },
        ]

    def recommend(self, state: UserState) -> Dict[str, Any]:
        """Return the next scheduled workout independent of current state/feedback."""
        day_idx = state.step % len(self.schedule_cycle)
        # Deep copy template dictionary
        template = self.schedule_cycle[day_idx]
        return {
            "workout_type": template["workout_type"],
            "target_intensity": template["target_intensity"],
            "estimated_duration_min": template["estimated_duration_min"],
            "volume": template["volume"],
            "exercises": [dict(ex) for ex in template["exercises"]],
            "rationale": template["rationale"],
        }
