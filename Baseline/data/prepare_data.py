"""Dataset preparation and inspection CLI driver for FitRec, PAMAP2, and WESAD."""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Baseline.config import OUTPUTS_DIR
from Baseline.data.fitrec import inspect_fitrec
from Baseline.data.pamap2 import inspect_pamap2
from Baseline.data.wesad import inspect_wesad

def prepare_datasets(
    dataset_choice: str = "all",
    fitrec_sample: int = 5000,
    output_dir: Path = OUTPUTS_DIR,
) -> dict:
    """Inspect and preprocess selected datasets into compact cached summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    print(f"Preparing datasets (selection={dataset_choice}, sample_limit={fitrec_sample})")

    # 1. FitRec
    if dataset_choice in ["all", "fitrec"]:
        print("\n[1/3] Inspecting FitRec / Endomondo...")
        t0 = time.perf_counter()
        res_fitrec = inspect_fitrec(max_records=fitrec_sample, output_dir=output_dir)
        res_fitrec["elapsed_sec"] = round(time.perf_counter() - t0, 2)
        results["FitRec"] = res_fitrec
        print(f"  Status: {res_fitrec['status']} ({res_fitrec.get('records_parsed', 0)} records in {res_fitrec['elapsed_sec']}s)")
        if res_fitrec["status"] == "SUCCESS":
            sports = list(res_fitrec.get("sport_distribution", {}).keys())[:4]
            hr_mean = res_fitrec.get("heart_rate_bpm_stats", {}).get("mean", "N/A")
            print(f"  Top sports: {sports} | Mean HR: {hr_mean} bpm")

    # 2. PAMAP2
    if dataset_choice in ["all", "pamap2"]:
        print("\n[2/3] Inspecting PAMAP2 Physical Activity...")
        t0 = time.perf_counter()
        res_pamap2 = inspect_pamap2(output_dir=output_dir)
        res_pamap2["elapsed_sec"] = round(time.perf_counter() - t0, 2)
        results["PAMAP2"] = res_pamap2
        print(f"  Status: {res_pamap2['status']} ({res_pamap2.get('subjects_parsed', 0)} subjects in {res_pamap2['elapsed_sec']}s)")
        if res_pamap2["status"] == "SUCCESS":
            acts = list(res_pamap2.get("per_activity_heart_rate", {}).keys())[:4]
            print(f"  Sampled activities with HR: {acts}")

    # 3. WESAD
    if dataset_choice in ["all", "wesad"]:
        print("\n[3/3] Inspecting WESAD Wearable Stress & Affect...")
        t0 = time.perf_counter()
        res_wesad = inspect_wesad(output_dir=output_dir)
        res_wesad["elapsed_sec"] = round(time.perf_counter() - t0, 2)
        results["WESAD"] = res_wesad
        print(f"  Status: {res_wesad['status']} ({res_wesad.get('subjects_parsed', 0)} subjects in {res_wesad['elapsed_sec']}s)")
        if res_wesad["status"] == "SUCCESS":
            s2_cond = res_wesad.get("per_subject", {}).get("S2", {}).get("physiological_by_condition", {})
            print(f"  S2 stress conditions summarized: {list(s2_cond.keys())}")

    # Summary manifest
    manifest_path = output_dir / "dataset_preparation_summary.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "dataset_results": results,
            },
            f,
            indent=2,
        )

    print(f"\nPreparation summary manifest saved to {manifest_path}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect and prepare datasets for HybridTrainer.")
    parser.add_argument("--dataset", choices=["all", "fitrec", "pamap2", "wesad"], default="all")
    parser.add_argument("--sample-size", type=int, default=5000, help="Record sample limit for FitRec")
    args = parser.parse_args()

    prepare_datasets(dataset_choice=args.dataset, fitrec_sample=args.sample_size)
