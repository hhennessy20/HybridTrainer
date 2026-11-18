"""One-command demo for HybridTrainer baseline experiment.

Generates a reproducible synthetic user, executes all four recommendation baselines
on the identical scenario, displays structured recommendations and a concise comparison
table, and saves full artifacts for checkpoint reporting.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Baseline.config import (
    SEED_DEMO,
    OUTPUTS_DIR,
    RL_QTABLE_PATH,
)
from Baseline.environment.synthetic_users import generate_synthetic_scenario
from Baseline.policies.static_schedule import StaticSchedulePolicy
from Baseline.policies.rule_based import RuleBasedPolicy
from Baseline.policies.rl_qlearning import TabularQLearningPolicy
from Baseline.policies.llm_ollama import OllamaLLMPolicy

def run_demo(
    seed: int = SEED_DEMO,
    skip_llm: bool = False,
    output_dir: Path = OUTPUTS_DIR,
) -> Dict[str, Any]:
    """Execute the interactive baseline comparison demo."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running Baseline experiment demo (seed={seed})")

    # 1. Ollama Readiness Check
    llm_policy = None
    llm_status_str = "DISABLED BY USER (--no-llm)" if skip_llm else "CHECKING..."

    if not skip_llm:
        candidate_llm = OllamaLLMPolicy()
        is_avail, avail_msg = candidate_llm.is_available()
        if is_avail:
            llm_policy = candidate_llm
            llm_status_str = f"ONLINE ({candidate_llm.active_model})"
        else:
            llm_status_str = f"UNAVAILABLE ({avail_msg})"

    print(f"\n[Environment Status] Local Ollama Service: {llm_status_str}")

    # 2. Generate Synthetic User Scenario
    state = generate_synthetic_scenario(seed=seed)
    print(f"\nUser Profile ({state.user_id}, Seed {state.seed}):")
    print(f"  Fitness: {state.fitness_level.upper()} | Goal: {state.goal.upper()} | Time: {state.available_time_min}m | Equip: {state.available_equipment}")
    print(f"  Fatigue: {state.fatigue:.2f} | Stress: {state.stress:.2f} | Target RPE: {state.target_rpe:.1f} | Adherence: {state.recent_completion_rate * 100:.0f}%")
    print(f"  Preferred: {state.preferred_categories} | Dispreferred: {state.dispreferred_categories}")
    if state.restrictions:
        print(f"  Restrictions: {state.restrictions}")

    # 3. Initialize Baselines
    # Ensure trained Q-table exists; if not, train a quick agent
    if not RL_QTABLE_PATH.exists():
        print("\nNotice: Trained RL policy not found. Training quick tabular agent (500 episodes)...")
        from Baseline.train_rl import train_rl
        train_rl(episodes=500, output_dir=output_dir)

    policies = [
        StaticSchedulePolicy(),
        RuleBasedPolicy(),
        TabularQLearningPolicy(qtable_path=RL_QTABLE_PATH),
    ]
    if llm_policy:
        policies.append(llm_policy)

    eval_results = []
    comparison_rows = []

    print("\n[Baseline Recommendations]")
    for p in policies:
        raw_act, val_res, fb, lat = p.evaluate_on_state(state)
        norm = val_res.normalized_action or raw_act

        ex_summary = []
        for ex in norm.get("exercises", []):
            detail = f"{ex.get('name')} [{ex.get('equipment')}]"
            if ex.get("duration_min"):
                detail += f" {ex.get('duration_min')}m"
            elif ex.get("sets") and ex.get("reps"):
                detail += f" {ex.get('sets')}x{ex.get('reps')}"
            ex_summary.append(detail)

        print(f"\n>> {p.name.upper()}:")
        print(f"   Workout Type:      {norm.get('workout_type')} (Prescribed Duration: {norm.get('estimated_duration_min')} min, Intensity: {norm.get('target_intensity')})")
        print(f"   Exercises:         {'; '.join(ex_summary) if ex_summary else 'None'}")
        print(f"   Rationale:         {norm.get('rationale')}")
        valid_tag = "VALID" if val_res.is_valid else f"INVALID ({len(val_res.violations)} violations)"
        print(f"   Constraints Check: {valid_tag}")
        if not val_res.is_valid:
            for viol in val_res.violations:
                print(f"      - {viol}")
        print(f"   Simulated Reward:  {fb.reward:+.3f} (Adherence: {fb.completion:.2f}, Satis: {fb.satisfaction:.2f}, Latency: {lat:.1f}ms)")

        eval_results.append({
            "policy": p.name,
            "raw_action": raw_act,
            "normalized_action": norm,
            "validation": val_res.to_dict(),
            "feedback": fb.to_dict(),
            "latency_ms": round(lat, 2),
        })

        comparison_rows.append({
            "Policy": p.name,
            "Valid": "YES" if val_res.is_valid else "NO",
            "Viols": len(val_res.violations),
            "Duration": f"{norm.get('estimated_duration_min')}m",
            "Adh(C)": f"{fb.completion:.2f}",
            "Sat(S)": f"{fb.satisfaction:.2f}",
            "RPE_Err": f"{fb.rpe_error:.1f}",
            "Fat_Pen": f"{fb.fatigue_penalty:.2f}",
            "Reward": f"{fb.reward:+.3f}",
            "Lat(ms)": f"{lat:.0f}",
        })

    # Add skipped row if LLM wasn't available
    if not llm_policy:
        comparison_rows.append({
            "Policy": "LLM_Ollama",
            "Valid": "SKIPPED",
            "Viols": "-",
            "Duration": "-",
            "Adh(C)": "-",
            "Sat(S)": "-",
            "RPE_Err": "-",
            "Fat_Pen": "-",
            "Reward": "-",
            "Lat(ms)": "-",
        })

    # 4. Comparison Table
    print(f"\n{'Policy':<16} {'Valid':<6} {'Viols':<5} {'Dur':<5} {'Adh':<6} {'Sat':<6} {'RPE_Err':<7} {'Fat_Pen':<7} {'Reward':<8} {'Latency'}")
    for row in comparison_rows:
        print(
            f"{row['Policy']:<16} "
            f"{row['Valid']:<6} "
            f"{row['Viols']:<5} "
            f"{row['Duration']:<5} "
            f"{row['Adh(C)']:<6} "
            f"{row['Sat(S)']:<6} "
            f"{row['RPE_Err']:<7} "
            f"{row['Fat_Pen']:<7} "
            f"{row['Reward']:<8} "
            f"{row['Lat(ms)']} ms"
        )

    # 5. Save Artifact
    demo_artifact = {
        "metadata": {
            "seed": seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ollama_status": llm_status_str,
        },
        "user_state": state.to_dict(),
        "baseline_results": eval_results,
    }
    demo_path = output_dir / "demo_result.json"
    with open(demo_path, "w", encoding="utf-8") as f:
        json.dump(demo_artifact, f, indent=2)

    print(f"\n[Saved Artifact] Demo results saved to: {demo_path}\n")

    return demo_artifact

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HybridTrainer Baseline Quick Demo")
    parser.add_argument("--seed", type=int, default=SEED_DEMO, help="Random seed for synthetic user")
    parser.add_argument("--no-llm", action="store_true", help="Skip Ollama local LLM baseline")
    args = parser.parse_args()

    run_demo(seed=args.seed, skip_llm=args.no_llm)
