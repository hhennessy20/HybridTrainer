"""Evaluation metrics calculation and aggregation for HybridTrainer baselines."""

import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

@dataclass
class ScenarioResult:
    scenario_idx: int
    seed: int
    user_id: str
    policy_name: str
    is_valid: bool
    violations_count: int
    violations: List[str]
    reward: float
    completion: float
    satisfaction: float
    target_rpe: float
    actual_rpe: float
    rpe_error: float
    fatigue_penalty: float
    prescribed_duration_min: int
    available_time_min: int
    duration_exceeded: bool
    latency_ms: float
    workout_type: str
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["violations_str"] = "; ".join(self.violations)
        return d

@dataclass
class AggregateMetrics:
    policy_name: str
    num_scenarios: int
    mean_reward: float
    std_reward: float
    mean_completion: float
    mean_satisfaction: float
    mean_rpe_error: float
    mean_fatigue_penalty: float
    validity_rate_pct: float            # Percentage of recommendations fully valid (0 violations)
    constraint_violation_rate_pct: float # Percentage of recommendations with >= 1 violation
    mean_violations_per_rec: float
    mean_prescribed_duration_min: float
    pct_exceeding_available_time: float # Percentage of recommendations exceeding available time
    mean_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "num_scenarios": self.num_scenarios,
            "mean_reward": round(self.mean_reward, 4),
            "std_reward": round(self.std_reward, 4),
            "mean_completion": round(self.mean_completion, 4),
            "mean_satisfaction": round(self.mean_satisfaction, 4),
            "mean_rpe_error": round(self.mean_rpe_error, 4),
            "mean_fatigue_penalty": round(self.mean_fatigue_penalty, 4),
            "validity_rate_pct": round(self.validity_rate_pct, 1),
            "constraint_violation_rate_pct": round(self.constraint_violation_rate_pct, 1),
            "mean_violations_per_rec": round(self.mean_violations_per_rec, 2),
            "mean_prescribed_duration_min": round(self.mean_prescribed_duration_min, 1),
            "pct_exceeding_available_time": round(self.pct_exceeding_available_time, 1),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
        }

def calculate_aggregate_metrics(results: List[ScenarioResult]) -> Optional[AggregateMetrics]:
    """Compute aggregate statistical metrics across scenario evaluations."""
    if not results:
        return None

    policy_name = results[0].policy_name
    n = len(results)

    rewards = [r.reward for r in results]
    mean_r = sum(rewards) / n
    std_r = math.sqrt(sum((x - mean_r) ** 2 for x in rewards) / max(1, n - 1)) if n > 1 else 0.0

    mean_c = sum(r.completion for r in results) / n
    mean_s = sum(r.satisfaction for r in results) / n
    mean_rpe_err = sum(r.rpe_error for r in results) / n
    mean_fatigue = sum(r.fatigue_penalty for r in results) / n

    valid_count = sum(1 for r in results if r.is_valid)
    validity_pct = (valid_count / n) * 100.0

    viol_count = sum(1 for r in results if r.violations_count > 0)
    viol_rate_pct = (viol_count / n) * 100.0
    mean_viols = sum(r.violations_count for r in results) / n

    mean_dur = sum(r.prescribed_duration_min for r in results) / n
    exceeded_count = sum(1 for r in results if r.duration_exceeded)
    pct_exceeded = (exceeded_count / n) * 100.0

    mean_lat = sum(r.latency_ms for r in results) / n

    return AggregateMetrics(
        policy_name=policy_name,
        num_scenarios=n,
        mean_reward=mean_r,
        std_reward=std_r,
        mean_completion=mean_c,
        mean_satisfaction=mean_s,
        mean_rpe_error=mean_rpe_err,
        mean_fatigue_penalty=mean_fatigue,
        validity_rate_pct=validity_pct,
        constraint_violation_rate_pct=viol_rate_pct,
        mean_violations_per_rec=mean_viols,
        mean_prescribed_duration_min=mean_dur,
        pct_exceeding_available_time=pct_exceeded,
        mean_latency_ms=mean_lat,
    )
