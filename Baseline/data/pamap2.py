"""PAMAP2 wearable activity dataset inspection and summary preprocessing."""

import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, List
from Baseline.config import PAMAP2_PROTOCOL_DIR, OUTPUTS_DIR

PAMAP2_ACTIVITY_LABELS = {
    0: "transient",
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "Nordic walking",
    9: "watching TV",
    10: "computer work",
    11: "car driving",
    12: "ascending stairs",
    13: "descending stairs",
    16: "vacuum cleaning",
    17: "ironing",
    18: "folding laundry",
    19: "house cleaning",
    20: "playing soccer",
    24: "rope jumping",
}

def inspect_pamap2(
    protocol_dir: Optional[Path] = None,
    max_rows_per_subject: int = 50000,
    stride: int = 5,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Inspect and extract activity-level physiological summaries from PAMAP2 protocol files."""
    p_dir = Path(protocol_dir) if protocol_dir else PAMAP2_PROTOCOL_DIR
    out_dir = Path(output_dir) if output_dir else OUTPUTS_DIR

    if not p_dir.exists():
        return {
            "dataset": "PAMAP2",
            "status": "SKIPPED",
            "reason": f"Protocol directory not found at {p_dir}",
            "subjects_parsed": 0,
        }

    dat_files = sorted(list(p_dir.glob("subject1*.dat")))
    if not dat_files:
        return {
            "dataset": "PAMAP2",
            "status": "SKIPPED",
            "reason": f"No subject .dat files found in {p_dir}",
            "subjects_parsed": 0,
        }

    per_subject_summary = {}
    activity_hr_aggregates: Dict[int, List[float]] = {}
    total_samples = 0
    total_hr_nans = 0

    for dat_file in dat_files:
        subj_id = dat_file.stem
        subj_rows = 0
        subj_nan_hr = 0
        subj_valid_hr = 0
        subj_activities = set()

        try:
            with open(dat_file, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if subj_rows >= max_rows_per_subject:
                        break
                    if idx % stride != 0:
                        continue

                    parts = line.split()
                    if len(parts) < 3:
                        continue

                    subj_rows += 1
                    try:
                        act_id = int(parts[1])
                        hr_str = parts[2]
                        subj_activities.add(act_id)

                        if hr_str.lower() == "nan" or hr_str == "":
                            subj_nan_hr += 1
                        else:
                            hr_val = float(hr_str)
                            if 30 <= hr_val <= 230:
                                subj_valid_hr += 1
                                if act_id not in activity_hr_aggregates:
                                    activity_hr_aggregates[act_id] = []
                                activity_hr_aggregates[act_id].append(hr_val)
                            else:
                                subj_nan_hr += 1
                    except (ValueError, IndexError):
                        continue
        except Exception as e:
            continue

        total_samples += subj_rows
        total_hr_nans += subj_nan_hr

        per_subject_summary[subj_id] = {
            "sampled_rows": subj_rows,
            "valid_hr_samples": subj_valid_hr,
            "nan_hr_samples": subj_nan_hr,
            "hr_missingness_rate_pct": round((subj_nan_hr / max(1, subj_rows)) * 100.0, 1),
            "activities_recorded": sorted(list(subj_activities)),
        }

    # Aggregate HR stats per activity
    activity_stats = {}
    for act_id, hrs in sorted(activity_hr_aggregates.items()):
        act_name = PAMAP2_ACTIVITY_LABELS.get(act_id, f"activity_{act_id}")
        if hrs:
            n = len(hrs)
            mean_hr = sum(hrs) / n
            std_hr = math.sqrt(sum((h - mean_hr) ** 2 for h in hrs) / max(1, n - 1)) if n > 1 else 0.0
            activity_stats[act_name] = {
                "activity_id": act_id,
                "sample_count": n,
                "mean_hr_bpm": round(mean_hr, 1),
                "std_hr_bpm": round(std_hr, 1),
                "min_hr_bpm": round(min(hrs), 1),
                "max_hr_bpm": round(max(hrs), 1),
            }

    summary = {
        "dataset": "PAMAP2",
        "status": "SUCCESS",
        "protocol_dir": str(p_dir),
        "subjects_parsed": len(per_subject_summary),
        "total_sampled_rows": total_samples,
        "overall_hr_missingness_rate_pct": round((total_hr_nans / max(1, total_samples)) * 100.0, 1),
        "per_subject": per_subject_summary,
        "per_activity_heart_rate": activity_stats,
    }

    cache_path = out_dir / "pamap2_summary.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
