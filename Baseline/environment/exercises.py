"""Controlled exercise catalogue with equipment requirements and categories."""

from dataclasses import dataclass
from typing import List, Optional

@dataclass(frozen=True)
class ExerciseDefinition:
    name: str
    category: str  # endurance, strength_upper, strength_lower, strength_core, flexibility, hiit
    required_equipment: str  # "bodyweight", "dumbbells", "barbell", "kettlebell", "resistance_bands", "pullup_bar", "cardio_machine"
    default_sets: Optional[int] = None
    default_reps: Optional[int] = None
    default_duration_min: Optional[int] = None

# Controlled catalogue for baseline evaluation
EXERCISE_CATALOGUE = [
    # Endurance / Cardio
    ExerciseDefinition("Brisk Walking", "endurance", "bodyweight", default_duration_min=30),
    ExerciseDefinition("Outdoor Running", "endurance", "bodyweight", default_duration_min=30),
    ExerciseDefinition("Stationary Cycling", "endurance", "cardio_machine", default_duration_min=30),
    ExerciseDefinition("Rowing Ergometer", "endurance", "cardio_machine", default_duration_min=25),
    ExerciseDefinition("Jump Rope", "endurance", "resistance_bands", default_duration_min=20),

    # Strength Upper
    ExerciseDefinition("Push-ups", "strength_upper", "bodyweight", default_sets=3, default_reps=12),
    ExerciseDefinition("Dumbbell Bench Press", "strength_upper", "dumbbells", default_sets=3, default_reps=10),
    ExerciseDefinition("Barbell Bench Press", "strength_upper", "barbell", default_sets=3, default_reps=8),
    ExerciseDefinition("Pull-ups", "strength_upper", "pullup_bar", default_sets=3, default_reps=6),
    ExerciseDefinition("Dumbbell Shoulder Press", "strength_upper", "dumbbells", default_sets=3, default_reps=10),
    ExerciseDefinition("Resistance Band Rows", "strength_upper", "resistance_bands", default_sets=3, default_reps=15),
    ExerciseDefinition("Dumbbell Bicep Curls", "strength_upper", "dumbbells", default_sets=3, default_reps=12),

    # Strength Lower
    ExerciseDefinition("Bodyweight Air Squats", "strength_lower", "bodyweight", default_sets=3, default_reps=15),
    ExerciseDefinition("Goblet Squats", "strength_lower", "kettlebell", default_sets=3, default_reps=12),
    ExerciseDefinition("Barbell Back Squat", "strength_lower", "barbell", default_sets=3, default_reps=8),
    ExerciseDefinition("Dumbbell Walking Lunges", "strength_lower", "dumbbells", default_sets=3, default_reps=12),
    ExerciseDefinition("Barbell Romanian Deadlift", "strength_lower", "barbell", default_sets=3, default_reps=8),
    ExerciseDefinition("Bodyweight Glute Bridges", "strength_lower", "bodyweight", default_sets=3, default_reps=15),

    # Strength Core
    ExerciseDefinition("Plank Hold", "strength_core", "bodyweight", default_duration_min=5),
    ExerciseDefinition("Russian Twists", "strength_core", "bodyweight", default_sets=3, default_reps=20),
    ExerciseDefinition("Hanging Knee Raises", "strength_core", "pullup_bar", default_sets=3, default_reps=12),

    # Flexibility & Mobility (Recovery)
    ExerciseDefinition("Dynamic Stretching Flow", "flexibility", "bodyweight", default_duration_min=15),
    ExerciseDefinition("Yoga Mobility Sequence", "flexibility", "bodyweight", default_duration_min=20),
    ExerciseDefinition("Hamstring & Hip Stretches", "flexibility", "bodyweight", default_duration_min=15),

    # HIIT / Circuits
    ExerciseDefinition("Burpees", "hiit", "bodyweight", default_sets=4, default_reps=10),
    ExerciseDefinition("Mountain Climbers", "hiit", "bodyweight", default_duration_min=10),
    ExerciseDefinition("Kettlebell Swings", "hiit", "kettlebell", default_sets=4, default_reps=15),
]

CATALOGUE_BY_NAME = {ex.name.lower(): ex for ex in EXERCISE_CATALOGUE}
ALL_EQUIPMENT_TYPES = [
    "bodyweight",
    "dumbbells",
    "barbell",
    "kettlebell",
    "resistance_bands",
    "pullup_bar",
    "cardio_machine"
]
WORKOUT_TYPES = ["endurance", "strength", "hiit", "flexibility", "recovery"]

def get_exercises_by_category(category: str) -> List[ExerciseDefinition]:
    return [ex for ex in EXERCISE_CATALOGUE if ex.category.startswith(category) or category.startswith(ex.category)]

def get_exercises_for_equipment(available_equipment: List[str]) -> List[ExerciseDefinition]:
    avail = set(e.lower() for e in available_equipment)
    avail.add("bodyweight")
    avail.add("none")
    return [ex for ex in EXERCISE_CATALOGUE if ex.required_equipment.lower() in avail]
