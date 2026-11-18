"""Interactive demonstration and batch testing for HybridTrainer."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Baseline.config import SEED_DEMO, RL_QTABLE_PATH
from Baseline.environment.synthetic_users import generate_scenarios, generate_synthetic_scenario
from Baseline.policies.static_schedule import StaticSchedulePolicy
from Baseline.policies.rule_based import RuleBasedPolicy
from Baseline.policies.rl_qlearning import TabularQLearningPolicy
from Baseline.policies.llm_ollama import OllamaLLMPolicy

from Implementation.state_encoder import encode_user_state, STATE_DIM
from Implementation.model import DuelingDQN
from Implementation.dqn_agent import DuelingDDQNAgent
from Implementation.hybrid_policy import HybridPolicy

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"


def run_code_demo(
    seed: int = SEED_DEMO,
    batch_size: int = 3,
    skip_llm: bool = False,
    output_dir: Path = OUTPUTS_DIR,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "dqn_model.pt"

    print(f"Running HybridTrainer demonstration (seed={seed}, batch_size={batch_size})")

    # Initialize baseline and neural RL policies
    static_pol = StaticSchedulePolicy()
    rule_pol = RuleBasedPolicy()
    tabular_pol = TabularQLearningPolicy(qtable_path=RL_QTABLE_PATH)
    dqn_pol = DuelingDDQNAgent(name="DuelingDQN", checkpoint_path=ckpt_path)

    use_ollama = not skip_llm
    pure_llm = OllamaLLMPolicy()
    is_avail, avail_msg = pure_llm.is_available() if use_ollama else (False, "Skipped by flag")
    print(f"Ollama Status: {'online (' + pure_llm.active_model + ')' if is_avail else 'offline (' + avail_msg + ')'}")

    tabular_hybrid = HybridPolicy(base_rl_policy=tabular_pol, name="TabularRL_LLM", use_llm=is_avail)
    dqn_hybrid = HybridPolicy(base_rl_policy=dqn_pol, name="HybridTrainer", use_llm=is_avail)

    policies = [static_pol, rule_pol, tabular_pol, dqn_pol]
    if is_avail:
        policies.extend([pure_llm, tabular_hybrid, dqn_hybrid])
    else:
        policies.extend([tabular_hybrid, dqn_hybrid])

    # Batch evaluation
    print(f"\nEvaluating batch of {batch_size} held-out user profiles:")
    batch_scenarios = generate_scenarios(start_seed=seed, count=batch_size)
    batch_results: List[Dict[str, Any]] = []

    for b_idx, u_state in enumerate(batch_scenarios):
        print(f"\nProfile {b_idx + 1} | User: {u_state.user_id} | Goal: {u_state.goal} | Time: {u_state.available_time_min}m | Fatigue: {u_state.fatigue:.2f} | Equipment: {u_state.available_equipment}")

        raw_act, val_res, fb, lat = dqn_pol.evaluate_on_state(u_state)
        norm = val_res.normalized_action or raw_act
        print(f"  Dueling DQN:   Type={norm.get('workout_type')} | Dur={norm.get('estimated_duration_min')}m | Intensity={norm.get('target_intensity')} | Reward={fb.reward:+.3f} | Latency={lat:.2f}ms")

        raw_h, val_h, fb_h, lat_h = dqn_hybrid.evaluate_on_state(u_state)
        norm_h = val_h.normalized_action or raw_h
        print(f"  HybridTrainer: Type={norm_h.get('workout_type')} | Dur={norm_h.get('estimated_duration_min')}m | Intensity={norm_h.get('target_intensity')} | Reward={fb_h.reward:+.3f} | Latency={lat_h:.1f}ms")
        print(f"    Rationale:   {norm_h.get('rationale')}")

        batch_results.append({
            "user_id": u_state.user_id,
            "dqn_reward": fb.reward,
            "hybrid_reward": fb_h.reward,
            "hybrid_satis": fb_h.satisfaction,
            "hybrid_adh": fb_h.completion,
            "valid": val_h.is_valid,
        })

    # Canonical scenario comparison
    demo_state = generate_synthetic_scenario(seed=seed)
    print(f"\nCanonical Scenario Comparison (User: {demo_state.user_id}, Goal: {demo_state.goal}, Time: {demo_state.available_time_min}m):")

    comparison_table = []
    detailed_outputs = {}

    for pol in policies:
        raw_act, val_res, fb, lat = pol.evaluate_on_state(demo_state)
        norm = val_res.normalized_action or raw_act
        est_dur = norm.get("estimated_duration_min", demo_state.available_time_min)

        row = {
            "Policy": pol.name,
            "Valid": "YES" if val_res.is_valid else f"NO ({len(val_res.violations)})",
            "Duration": f"{est_dur}m",
            "Reward": f"{fb.reward:+.3f}",
            "Adherence": f"{fb.completion:.2f}",
            "Satisfaction": f"{fb.satisfaction:.2f}",
            "RPE_Error": f"{fb.rpe_error:.1f}",
            "Latency": f"{lat:.1f}ms" if lat >= 1.0 else "<1ms",
        }
        comparison_table.append(row)
        detailed_outputs[pol.name] = {
            "action": norm,
            "feedback": fb.to_dict(),
            "validation": val_res.to_dict(),
            "latency_ms": lat,
        }

    print(f"\n{'Policy':<24} {'Valid':<8} {'Dur':<6} {'Reward':<9} {'Adh':<6} {'Sat':<6} {'RPE_Err':<8} {'Latency':<9}")
    for r in comparison_table:
        print(f"{r['Policy']:<24} {r['Valid']:<8} {r['Duration']:<6} {r['Reward']:<9} {r['Adherence']:<6} {r['Satisfaction']:<6} {r['RPE_Error']:<8} {r['Latency']:<9}")

    # Save artifacts
    demo_artifact = {
        "scenario": demo_state.to_dict(),
        "batch_summary": batch_results,
        "comparison_table": comparison_table,
        "policy_details": detailed_outputs,
    }
    demo_file = output_dir / "demo_result.json"
    with open(demo_file, "w", encoding="utf-8") as f:
        json.dump(demo_artifact, f, indent=2)
    print(f"\nDemo results saved to {demo_file}")

    return demo_artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED_DEMO)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()
    run_code_demo(seed=args.seed, batch_size=args.batch_size, skip_llm=args.no_llm)
