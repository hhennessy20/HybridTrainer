"""Held-out scenario evaluation pipeline for all HybridTrainer baselines."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Baseline.config import (
    SEED_TEST,
    OUTPUTS_DIR,
    RL_QTABLE_PATH,
)
from Baseline.environment.synthetic_users import generate_scenarios
from Baseline.policies.static_schedule import StaticSchedulePolicy
from Baseline.policies.rule_based import RuleBasedPolicy
from Baseline.policies.rl_qlearning import TabularQLearningPolicy
from Baseline.policies.llm_ollama import OllamaLLMPolicy
from Baseline.evaluation.metrics import (
    ScenarioResult,
    AggregateMetrics,
    calculate_aggregate_metrics,
)

def run_evaluation(
    num_scenarios: int = 25,
    llm_scenarios: Optional[int] = None,
    seed_test: int = SEED_TEST,
    include_llm: bool = True,
    output_dir: Path = OUTPUTS_DIR,
) -> Dict[str, Any]:
    """Run fair evaluation across identical held-out synthetic scenarios for all baselines."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating {num_scenarios} held-out scenarios (seed={seed_test})")

    # Generate held-out test scenarios
    scenarios = generate_scenarios(seed_test, num_scenarios)

    # Initialize baselines
    policies = [
        StaticSchedulePolicy(),
        RuleBasedPolicy(),
        TabularQLearningPolicy(qtable_path=RL_QTABLE_PATH),
    ]

    llm_policy = None
    if include_llm:
        llm = OllamaLLMPolicy()
        avail, msg = llm.is_available()
        if avail:
            llm_policy = llm
            print(f"Ollama LLM policy active: {msg}")
        else:
            print(f"Ollama LLM skipped: {msg}")

    all_scenario_results: List[ScenarioResult] = []
    summary_by_policy: Dict[str, Dict[str, Any]] = {}

    # Evaluate non-LLM baselines
    for policy in policies:
        print(f"Evaluating {policy.name} on {len(scenarios)} scenarios...")
        pol_results: List[ScenarioResult] = []
        for idx, state in enumerate(scenarios):
            raw_act, val_res, fb, lat = policy.evaluate_on_state(state)
            norm = val_res.normalized_action or raw_act
            est_dur = norm.get("estimated_duration_min", state.available_time_min)
            res = ScenarioResult(
                scenario_idx=idx + 1,
                seed=state.seed,
                user_id=state.user_id,
                policy_name=policy.name,
                is_valid=val_res.is_valid,
                violations_count=len(val_res.violations),
                violations=val_res.violations,
                reward=fb.reward,
                completion=fb.completion,
                satisfaction=fb.satisfaction,
                target_rpe=state.target_rpe,
                actual_rpe=fb.actual_rpe,
                rpe_error=fb.rpe_error,
                fatigue_penalty=fb.fatigue_penalty,
                prescribed_duration_min=est_dur,
                available_time_min=state.available_time_min,
                duration_exceeded=(est_dur > state.available_time_min),
                latency_ms=lat,
                workout_type=str(norm.get("workout_type", "")),
                rationale=str(norm.get("rationale", "")),
            )
            pol_results.append(res)
            all_scenario_results.append(res)

        agg = calculate_aggregate_metrics(pol_results)
        if agg:
            summary_by_policy[policy.name] = agg.to_dict()

    # Evaluate LLM baseline if active
    if llm_policy:
        target_count = llm_scenarios if llm_scenarios is not None else num_scenarios
        target_count = min(target_count, len(scenarios))
        llm_test_set = scenarios[:target_count]
        print(f"Evaluating {llm_policy.name} on {len(llm_test_set)} scenarios (local Ollama)...")
        llm_results: List[ScenarioResult] = []
        for idx, state in enumerate(llm_test_set):
            raw_act, val_res, fb, lat = llm_policy.evaluate_on_state(state)
            norm = val_res.normalized_action or raw_act
            est_dur = norm.get("estimated_duration_min", state.available_time_min)
            res = ScenarioResult(
                scenario_idx=idx + 1,
                seed=state.seed,
                user_id=state.user_id,
                policy_name=llm_policy.name,
                is_valid=val_res.is_valid,
                violations_count=len(val_res.violations),
                violations=val_res.violations,
                reward=fb.reward,
                completion=fb.completion,
                satisfaction=fb.satisfaction,
                target_rpe=state.target_rpe,
                actual_rpe=fb.actual_rpe,
                rpe_error=fb.rpe_error,
                fatigue_penalty=fb.fatigue_penalty,
                prescribed_duration_min=est_dur,
                available_time_min=state.available_time_min,
                duration_exceeded=(est_dur > state.available_time_min),
                latency_ms=lat,
                workout_type=str(norm.get("workout_type", "")),
                rationale=str(norm.get("rationale", "")),
            )
            llm_results.append(res)
            all_scenario_results.append(res)

        agg_llm = calculate_aggregate_metrics(llm_results)
        if agg_llm:
            summary_by_policy[llm_policy.name] = agg_llm.to_dict()

    # Print comparison table
    print(f"\n{'Policy':<16} | {'N':<3} | {'Mean Reward':<11} | {'Compl.':<7} | {'Satis.':<7} | {'RPE Err':<7} | {'Valid %':<7} | {'Viol/Rec':<8} | {'Lat(ms)':<8}")
    for pol_name, m in summary_by_policy.items():
        print(
            f"{pol_name:<16} | {m['num_scenarios']:<3} | "
            f"{m['mean_reward']:>6.3f} +/- {m['std_reward']:<4.2f} | "
            f"{m['mean_completion']:>6.3f}  | "
            f"{m['mean_satisfaction']:>6.3f}  | "
            f"{m['mean_rpe_error']:>6.3f}  | "
            f"{m['validity_rate_pct']:>6.1f}% | "
            f"{m['mean_violations_per_rec']:>6.2f}   | "
            f"{m['mean_latency_ms']:>8.1f}"
        )

    # Save summary JSON
    summary_json_path = output_dir / "evaluation_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "num_scenarios": num_scenarios,
                    "seed_test": seed_test,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                "results": summary_by_policy,
            },
            f,
            indent=2,
        )

    # Save summary CSV
    summary_csv_path = output_dir / "evaluation_summary.csv"
    if summary_by_policy:
        fieldnames = list(next(iter(summary_by_policy.values())).keys())
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_by_policy.values():
                writer.writerow(row)

    # Save per-scenario CSV
    scenarios_csv_path = output_dir / "evaluation_scenarios.csv"
    if all_scenario_results:
        scen_fields = list(all_scenario_results[0].to_dict().keys())
        # Remove violations list object for clean CSV
        if "violations" in scen_fields:
            scen_fields.remove("violations")
        with open(scenarios_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=scen_fields)
            writer.writeheader()
            for sr in all_scenario_results:
                d = sr.to_dict()
                d.pop("violations", None)
                writer.writerow(d)

    print(f"Saved evaluation summary JSON: {summary_json_path}")
    print(f"Saved evaluation summary CSV:  {summary_csv_path}")
    print(f"Saved scenario details CSV:    {scenarios_csv_path}")

    return summary_by_policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HybridTrainer baselines on held-out scenarios.")
    parser.add_argument("--scenarios", type=int, default=25, help="Number of held-out test scenarios")
    parser.add_argument("--llm-scenarios", type=int, default=10, help="Number of scenarios for Ollama LLM")
    parser.add_argument("--seed-test", type=int, default=SEED_TEST, help="Seed for test scenarios")
    parser.add_argument("--no-llm", action="store_true", help="Skip Ollama LLM evaluation")
    args = parser.parse_args()

    run_evaluation(
        num_scenarios=args.scenarios,
        llm_scenarios=args.llm_scenarios,
        seed_test=args.seed_test,
        include_llm=(not args.no_llm),
    )
