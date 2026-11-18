"""Populate Synthetic_User_Data directory with dataset-calibrated synthetic user profiles.

Generates 10,000 synthetic users calibrated by empirical distributions from
FitRec (workout duration & activity preferences), PAMAP2 (exertion & target RPE),
and WESAD (stress prevalence).
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Baseline.config import SYNTHETIC_DATA_DIR, OUTPUTS_DIR
from Baseline.environment.synthetic_users import (
    generate_synthetic_scenario,
    load_dataset_calibration,
)

def populate_synthetic_users(
    count: int = 10000,
    start_seed: int = 50000,
    output_dir: Path = SYNTHETIC_DATA_DIR,
) -> dict:
    """Generate and serialize calibrated synthetic user profiles."""
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    print(f"Generating {count:,} synthetic users (start_seed={start_seed})")
    print(f"Output Directory: {output_dir}")

    # Load dataset calibration
    cal = load_dataset_calibration(OUTPUTS_DIR)
    print(f"Dataset Calibration Status:")
    print(f"  * FitRec (Durations & Activity Bias): {'CALIBRATED' if cal['fitrec_calibrated'] else 'FALLBACK'}")
    print(f"  * PAMAP2 (Exertion Dynamics):          {'CALIBRATED' if cal['pamap2_calibrated'] else 'FALLBACK'}")
    print(f"  * WESAD  (Empirical Stress Ratio):     {'CALIBRATED' if cal['wesad_calibrated'] else 'FALLBACK'}")

    jsonl_path = output_dir / f"synthetic_users_{count // 1000}k.jsonl"
    csv_path = output_dir / f"synthetic_users_{count // 1000}k.csv"
    sample_path = output_dir / "synthetic_users_sample.json"
    summary_path = output_dir / "synthetic_users_summary.json"

    # Aggregators for summary statistics
    fitness_counts = {}
    goal_counts = {}
    time_counts = {}
    stress_vals = []
    fatigue_vals = []
    rpe_vals = []
    sample_users = []

    print(f"\nGenerating {count:,} profiles...")
    csv_headers = [
        "user_id", "seed", "fitness_level", "goal", "available_time_min",
        "available_equipment", "preferred_categories", "dispreferred_categories",
        "fatigue", "stress", "recent_completion_rate", "target_rpe", "restrictions"
    ]

    with open(jsonl_path, "w", encoding="utf-8") as f_jsonl, \
         open(csv_path, "w", newline="", encoding="utf-8") as f_csv:

        csv_writer = csv.DictWriter(f_csv, fieldnames=csv_headers)
        csv_writer.writeheader()

        for i in range(count):
            seed = start_seed + i
            user = generate_synthetic_scenario(seed=seed, calibration=cal)
            u_dict = user.to_dict()

            # Write JSONL
            f_jsonl.write(json.dumps(u_dict) + "\n")

            # Write CSV (flatten lists to semicolon-delimited strings)
            csv_row = dict(u_dict)
            csv_row["available_equipment"] = ";".join(u_dict["available_equipment"])
            csv_row["preferred_categories"] = ";".join(u_dict["preferred_categories"])
            csv_row["dispreferred_categories"] = ";".join(u_dict["dispreferred_categories"])
            csv_row["restrictions"] = ";".join(u_dict["restrictions"])
            csv_row.pop("step", None)
            csv_writer.writerow(csv_row)

            # Accumulate stats
            fitness_counts[user.fitness_level] = fitness_counts.get(user.fitness_level, 0) + 1
            goal_counts[user.goal] = goal_counts.get(user.goal, 0) + 1
            time_counts[user.available_time_min] = time_counts.get(user.available_time_min, 0) + 1
            stress_vals.append(user.stress)
            fatigue_vals.append(user.fatigue)
            rpe_vals.append(user.target_rpe)

            if i < 25:
                sample_users.append(u_dict)

            if (i + 1) % 2500 == 0:
                print(f"  Processed {i + 1:,} / {count:,} profiles...")

    elapsed = time.perf_counter() - t0

    # Save sample JSON
    with open(sample_path, "w", encoding="utf-8") as f_samp:
        json.dump(sample_users, f_samp, indent=2)

    # Save summary metadata
    summary_data = {
        "metadata": {
            "total_users": count,
            "starting_seed": start_seed,
            "elapsed_seconds": round(elapsed, 2),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "calibration": cal,
        },
        "distributions": {
            "fitness_level": fitness_counts,
            "goals": goal_counts,
            "available_time_minutes": time_counts,
            "fatigue_mean": round(sum(fatigue_vals) / count, 3),
            "stress_mean": round(sum(stress_vals) / count, 3),
            "target_rpe_mean": round(sum(rpe_vals) / count, 2),
        },
        "files_generated": [
            str(jsonl_path.name),
            str(csv_path.name),
            str(sample_path.name),
            str(summary_path.name),
        ],
    }
    with open(summary_path, "w", encoding="utf-8") as f_sum:
        json.dump(summary_data, f_sum, indent=2)

    print(f"Cohort generation completed in {elapsed:.2f} seconds ({count / elapsed:.0f} users/sec).")
    print(f"  -> JSONL:   {jsonl_path} ({jsonl_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  -> CSV:     {csv_path} ({csv_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  -> Sample:  {sample_path}")
    print(f"  -> Summary: {summary_path}")

    return summary_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Populate Synthetic_User_Data folder with calibrated users.")
    parser.add_argument("--count", type=int, default=10000, help="Number of synthetic users to generate")
    parser.add_argument("--seed", type=int, default=50000, help="Starting random seed")
    args = parser.parse_args()

    populate_synthetic_users(count=args.count, start_seed=args.seed)
