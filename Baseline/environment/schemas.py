"""Shared data schemas and common action validator for HybridTrainer."""

from dataclasses import dataclass, field, asdict
import json
from typing import List, Dict, Any, Optional
from Baseline.config import (
    DURATION_TOLERANCE_PCT,
    INTENSITY_MIN,
    INTENSITY_MAX,
    MIN_WORKOUT_DURATION_MIN,
    MAX_WORKOUT_DURATION_MIN,
)
from Baseline.environment.exercises import CATALOGUE_BY_NAME

@dataclass
class UserState:
    user_id: str
    seed: int
    fitness_level: str               # "beginner", "intermediate", "advanced"
    goal: str                        # "endurance", "strength", "general_fitness", "weight_loss"
    available_time_min: int          # 15 to 90 min
    available_equipment: List[str]   # list of equipment strings
    preferred_categories: List[str]  # e.g. ["strength", "flexibility"]
    dispreferred_categories: List[str]  # e.g. ["hiit"]
    fatigue: float                   # 0.0 to 1.0 (0=fresh, 1=exhausted)
    stress: float                    # 0.0 to 1.0 (0=calm, 1=high stress)
    recent_completion_rate: float    # 0.0 to 1.0 historical adherence
    target_rpe: float                # 1.0 to 10.0 target Borg CR10 RPE
    restrictions: List[str] = field(default_factory=list)
    step: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserState":
        return cls(**data)

@dataclass
class ExerciseItem:
    name: str
    category: str
    equipment: str
    sets: Optional[int] = None
    reps: Optional[int] = None
    duration_min: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class WorkoutAction:
    workout_type: str                # "endurance", "strength", "hiit", "flexibility", "recovery"
    target_intensity: float          # 1.0 to 10.0
    estimated_duration_min: int      # estimated duration in minutes
    volume: str                      # "low", "moderate", "high"
    exercises: List[Dict[str, Any]]  # exercise items with details
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkoutAction":
        return cls(**data)

@dataclass
class ValidationResult:
    is_valid: bool
    violations: List[str]
    normalized_action: Optional[Dict[str, Any]] = None
    raw_action: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "violations": self.violations,
            "normalized_action": self.normalized_action,
            "raw_action": self.raw_action,
        }

def validate_workout_action(
    action_dict: Dict[str, Any],
    state: UserState,
    tolerance_pct: float = DURATION_TOLERANCE_PCT
) -> ValidationResult:
    """Validate a recommendation against user constraints and shared action schema."""
    violations: List[str] = []
    
    if not isinstance(action_dict, dict):
        return ValidationResult(
            is_valid=False,
            violations=["Action payload is not a valid JSON dictionary."],
            raw_action=action_dict
        )

    # Required top-level keys
    required_keys = ["workout_type", "target_intensity", "estimated_duration_min", "exercises"]
    for k in required_keys:
        if k not in action_dict:
            violations.append(f"Missing required field: '{k}'.")

    # Workout type
    workout_type = str(action_dict.get("workout_type", "")).strip().lower()
    valid_types = {"endurance", "strength", "hiit", "flexibility", "recovery", "general"}
    if not any(vt in workout_type for vt in valid_types):
        violations.append(f"Unrecognized workout_type '{workout_type}'. Expected one of {sorted(valid_types)}.")

    # Target intensity bounds
    target_intensity_raw = action_dict.get("target_intensity")
    target_intensity = None
    try:
        target_intensity = float(target_intensity_raw)
    except (TypeError, ValueError):
        import re
        match = re.search(r"(\d+(?:\.\d+)?)", str(target_intensity_raw))
        if match:
            target_intensity = float(match.group(1))
        else:
            violations.append(f"target_intensity '{target_intensity_raw}' is not a numeric value.")
            target_intensity = 5.0

    if target_intensity is not None and not (INTENSITY_MIN <= target_intensity <= INTENSITY_MAX):
        violations.append(
            f"target_intensity {target_intensity:.1f} out of bounds [{INTENSITY_MIN}, {INTENSITY_MAX}]."
        )

    # Duration bounds and user time constraint
    est_duration_raw = action_dict.get("estimated_duration_min")
    est_duration = None
    try:
        est_duration = int(round(float(est_duration_raw)))
    except (TypeError, ValueError):
        import re
        match = re.search(r"(\d+)", str(est_duration_raw))
        if match:
            est_duration = int(match.group(1))
        else:
            violations.append(f"estimated_duration_min '{est_duration_raw}' is not a valid integer.")
            est_duration = state.available_time_min

    max_allowed = int(round(state.available_time_min * (1.0 + tolerance_pct)))
    if est_duration < MIN_WORKOUT_DURATION_MIN:
        violations.append(
            f"estimated_duration_min {est_duration} min is below minimum ({MIN_WORKOUT_DURATION_MIN} min)."
        )
    if est_duration > max_allowed:
        violations.append(
            f"estimated_duration_min {est_duration} min exceeds available time {state.available_time_min} min "
            f"(max allowed with {int(tolerance_pct*100)}% tolerance: {max_allowed} min)."
        )

    # Exercises validation
    exercises = action_dict.get("exercises")
    if not isinstance(exercises, list) or len(exercises) == 0:
        violations.append("exercises must be a non-empty list.")
        exercises = []
    
    # Equipment check
    user_equipment = {e.strip().lower() for e in state.available_equipment}
    user_equipment.add("bodyweight")
    user_equipment.add("none")

    normalized_exercises = []
    for idx, ex in enumerate(exercises):
        if not isinstance(ex, dict):
            violations.append(f"Exercise at index {idx} is not an object.")
            continue
        ex_name = str(ex.get("name", f"Exercise_{idx+1}")).strip()
        ex_equip = str(ex.get("equipment", "bodyweight")).strip().lower()

        # Check against catalogue if possible to verify official equipment requirement
        cat_match = CATALOGUE_BY_NAME.get(ex_name.lower())
        required_equip = cat_match.required_equipment.lower() if cat_match else ex_equip

        if required_equip not in user_equipment:
            violations.append(
                f"Exercise '{ex_name}' requires '{required_equip}' not in user available equipment: "
                f"{sorted(list(user_equipment))}."
            )

        normalized_exercises.append({
            "name": ex_name,
            "category": ex.get("category", workout_type),
            "equipment": required_equip,
            "sets": ex.get("sets"),
            "reps": ex.get("reps"),
            "duration_min": ex.get("duration_min")
        })

    # Normalized action structure
    normalized_action = {
        "workout_type": workout_type,
        "target_intensity": target_intensity,
        "estimated_duration_min": est_duration,
        "volume": str(action_dict.get("volume", "moderate")).lower(),
        "exercises": normalized_exercises,
        "rationale": str(action_dict.get("rationale", "")).strip()
    }

    return ValidationResult(
        is_valid=(len(violations) == 0),
        violations=violations,
        normalized_action=normalized_action,
        raw_action=action_dict
    )
