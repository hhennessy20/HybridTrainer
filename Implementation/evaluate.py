"""Comprehensive evaluation benchmark comparing Novel HybridTrainer against all baselines.

Evaluates across identical 25 held-out scenarios (seeds 3000-3024) established in Ckpt 2:
1. StaticSchedule (Non-adaptive baseline)
2. RuleBased (Expert heuristics baseline)
3. TabularRL (Discretized tabular Q-learning baseline)
4. LLM_Ollama (Pure local LLM Qwen3.5:0.8B baseline)
5. TabularRL_LLM (Hybrid: Tabular RL + Local LLM synthesis)
6. DuelingDQN (Novel Continuous-State Deep Q-Network without LLM)
7. HybridTrainer (Full Novel System: Dueling DQN + Local LLM synthesis)
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Baseline.config import (
    SEED_TEST,
    RL_QTABLE_PATH,
)
from Baseline.environment.synthetic_users import generate_scenarios
from Baseline.policies.static_schedule import StaticSchedulePolicy
from Baseline.policies.rule_based import RuleBasedPolicy
from Baseline.policies.rl_qlearning import TabularQLearningPolicy
from Baseline.policies.llm_ollama import OllamaLLMPolicy

from Implementation.dqn_agent import DuelingDDQNAgent
from Implementation.hybrid_policy import HybridPolicy

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"


def run_benchmark(
    num_scenarios: int = 25,
    llm_scenarios: int = 25,
    seed_test: int = SEED_TEST,
    output_dir: Path = OUTPUTS_DIR,
    model_checkpoint: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute complete fair evaluation across identical held-out scenarios."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_checkpoint or (output_dir / "dqn_model.pt")

    print(f"Evaluating {len(policies)} policies on {num_scenarios} held-out scenarios (seed={seed_test})")

    # Generate held-out test scenarios
    scenarios = generate_scenarios(seed_test, num_scenarios)

    # 1. Initialize Baseline Policies
    static_pol = StaticSchedulePolicy()
    rule_pol = RuleBasedPolicy()
    tabular_pol = TabularQLearningPolicy(qtable_path=RL_QTABLE_PATH)
    pure_llm = OllamaLLMPolicy()

    # 2. Initialize Novel Policies
    dqn_pol = DuelingDDQNAgent(name="DuelingDQN", checkpoint_path=ckpt_path)

    # 3. Initialize Hybrid Synthesis Policies
    tabular_hybrid = HybridPolicy(base_rl_policy=tabular_pol, name="TabularRL_LLM", use_llm=True)
    dqn_hybrid = HybridPolicy(base_rl_policy=dqn_pol, name="HybridTrainer_DQN_LLM", use_llm=True)

    policies = [
        static_pol,
        rule_pol,
        tabular_pol,
        pure_llm,
        tabular_hybrid,
        dqn_pol,
        dqn_hybrid,
    ]

    all_scenario_rows: List[Dict[str, Any]] = []
    summary_by_policy: Dict[str, Dict[str, Any]] = {}

    for policy in policies:
        pol_name = policy.name
        # For heavy pure LLM or hybrids, respect scenario cap if specified
        target_scenarios = scenarios[:llm_scenarios] if "LLM" in pol_name else scenarios
        print(f"\nEvaluating '{pol_name}' on {len(target_scenarios)} held-out scenarios...")

        rewards = []
        completions = []
        satisfactions = []
        rpe_errors = []
        fatigue_penalties = []
        violations = []
        over_times = []
        latencies = []
        valid_count = 0

        for idx, state in enumerate(target_scenarios):
            raw_act, val_res, fb, lat = policy.evaluate_on_state(state)
            norm = val_res.normalized_action or raw_act
            est_dur = norm.get("estimated_duration_min", state.available_time_min)
            is_over = est_dur > state.available_time_min

            rewards.append(fb.reward)
            completions.append(fb.completion)
            satisfactions.append(fb.satisfaction)
            rpe_errors.append(fb.rpe_error)
            fatigue_penalties.append(fb.fatigue_penalty)
            violations.append(len(val_res.violations))
            over_times.append(1 if is_over else 0)
            latencies.append(lat)
            if val_res.is_valid:
                valid_count += 1

            all_scenario_rows.append({
                "scenario_idx": idx + 1,
                "seed": state.seed,
                "user_id": state.user_id,
                "policy": pol_name,
                "valid": val_res.is_valid,
                "violations_count": len(val_res.violations),
                "reward": round(fb.reward, 4),
                "adherence": round(fb.completion, 4),
                "satisfaction": round(fb.satisfaction, 4),
                "rpe_error": round(fb.rpe_error, 2),
                "fatigue_penalty": round(fb.fatigue_penalty, 4),
                "prescribed_dur": est_dur,
                "avail_time": state.available_time_min,
                "over_time": is_over,
                "latency_ms": round(lat, 2),
                "workout_type": norm.get("workout_type", ""),
            })

        n = len(target_scenarios)
        summary = {
            "policy_name": pol_name,
            "num_scenarios": n,
            "mean_reward": round(float(np.mean(rewards)), 4),
            "std_reward": round(float(np.std(rewards)), 4),
            "mean_completion": round(float(np.mean(completions)), 4),
            "mean_satisfaction": round(float(np.mean(satisfactions)), 4),
            "mean_rpe_error": round(float(np.mean(rpe_errors)), 4),
            "mean_fatigue_penalty": round(float(np.mean(fatigue_penalties)), 4),
            "validity_rate_pct": round((valid_count / n) * 100.0, 1),
            "constraint_violation_rate_pct": round((1.0 - (valid_count / n)) * 100.0, 1),
            "mean_violations_per_rec": round(float(np.mean(violations)), 2),
            "pct_exceeding_available_time": round(float(np.mean(over_times)) * 100.0, 1),
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
        }
        summary_by_policy[pol_name] = summary
        print(
            f"   -> Reward: {summary['mean_reward']:+.3f} ± {summary['std_reward']:.3f} | "
            f"Adherence: {summary['mean_completion']:.2f} | "
            f"Satisfaction: {summary['mean_satisfaction']:.2f} | "
            f"Latency: {summary['mean_latency_ms']:.1f}ms | "
            f"Valid: {summary['validity_rate_pct']:.0f}%"
        )

    # Save summary JSON
    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_by_policy, f, indent=2)

    # Save summary CSV table
    summary_csv_path = output_dir / "evaluation_summary.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(summary_by_policy.values())).keys()))
        writer.writeheader()
        for s in summary_by_policy.values():
            writer.writerow(s)

    # Save per-scenario CSV
    scenarios_csv_path = output_dir / "evaluation_scenarios.csv"
    with open(scenarios_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_scenario_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_scenario_rows)

    print(f"\n{'Policy':<24} {'Reward':<16} {'Adh':<8} {'Sat':<8} {'RPE_Err':<8} {'Viol%':<7} {'Latency':<9}")
    for p_name, s in summary_by_policy.items():
        print(
            f"{p_name:<24} "
            f"{s['mean_reward']:+.3f} ± {s['std_reward']:.2f}    "
            f"{s['mean_completion']:<8.2f} "
            f"{s['mean_satisfaction']:<8.2f} "
            f"{s['mean_rpe_error']:<8.2f} "
            f"{s['constraint_violation_rate_pct']:<7.1f} "
            f"{s['mean_latency_ms']:<8.1f}ms"
        )

    return summary_by_policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=int, default=25)
    parser.add_argument("--llm-scenarios", type=int, default=25)
    args = parser.parse_args()
    run_benchmark(num_scenarios=args.scenarios, llm_scenarios=args.llm_scenarios)

