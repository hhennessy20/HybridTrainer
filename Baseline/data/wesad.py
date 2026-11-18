"""WESAD wearable stress and affect dataset inspection and summary preprocessing."""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
from Baseline.config import WESAD_DIR, OUTPUTS_DIR

WESAD_LABEL_NAMES = {
    0: "transient",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
    6: "undefined_6",
    7: "undefined_7",
}

def inspect_wesad(
    wesad_root: Optional[Path] = None,
    subject_ids: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Inspect and extract condition-specific physiological summaries from WESAD pickle files."""
    root = Path(wesad_root) if wesad_root else WESAD_DIR
    out_dir = Path(output_dir) if output_dir else OUTPUTS_DIR

    if not root.exists():
        return {
            "dataset": "WESAD",
            "status": "SKIPPED",
            "reason": f"WESAD directory not found at {root}",
            "subjects_parsed": 0,
        }

    # Find subject directories
    target_subs = subject_ids or ["S2", "S3"]
    subject_dirs = [root / sid for sid in target_subs if (root / sid / f"{sid}.pkl").exists()]

    if not subject_dirs:
        # Search all
        all_pkls = list(root.glob("S*/*.pkl"))
        if not all_pkls:
            return {
                "dataset": "WESAD",
                "status": "SKIPPED",
                "reason": f"No subject .pkl files found in {root}",
                "subjects_parsed": 0,
            }
        subject_dirs = [p.parent for p in all_pkls[:2]]

    per_subject_summary = {}

    for sdir in subject_dirs:
        sid = sdir.name
        pkl_file = sdir / f"{sid}.pkl"

        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f, encoding="latin1")

            labels = data.get("label", np.array([]))
            signals = data.get("signal", {})
            chest_sigs = signals.get("chest", {})
            wrist_sigs = signals.get("wrist", {})

            # Count samples per label
            unique_labels, counts = np.unique(labels, return_counts=True)
            label_dist = {
                WESAD_LABEL_NAMES.get(int(lbl), f"label_{lbl}"): int(cnt)
                for lbl, cnt in zip(unique_labels, counts)
            }

            # Sampling rates context: Chest is 700 Hz, Wrist ranges from 4 to 64 Hz
            chest_eda = chest_sigs.get("EDA")
            chest_temp = chest_sigs.get("Temp")

            physio_by_condition = {}
            for target_lbl, cond_name in [(1, "baseline"), (2, "stress"), (3, "amusement"), (4, "meditation")]:
                mask = (labels == target_lbl)
                if np.any(mask):
                    cond_stats = {}
                    if chest_eda is not None and len(chest_eda) == len(labels):
                        eda_subset = chest_eda[mask]
                        cond_stats["chest_eda_mean_uS"] = round(float(np.mean(eda_subset)), 4)
                        cond_stats["chest_eda_std_uS"] = round(float(np.std(eda_subset)), 4)
                    if chest_temp is not None and len(chest_temp) == len(labels):
                        temp_subset = chest_temp[mask]
                        cond_stats["chest_temp_mean_celsius"] = round(float(np.mean(temp_subset)), 2)
                        cond_stats["chest_temp_std_celsius"] = round(float(np.std(temp_subset)), 2)

                    physio_by_condition[cond_name] = cond_stats

            per_subject_summary[sid] = {
                "total_chest_samples_700hz": len(labels),
                "recorded_minutes_approx": round(len(labels) / (700.0 * 60.0), 1),
                "label_counts": label_dist,
                "available_signals": {
                    "chest": list(chest_sigs.keys()),
                    "wrist": list(wrist_sigs.keys()),
                },
                "physiological_by_condition": physio_by_condition,
            }

            # Explicitly delete to release memory
            del data
            del labels
            del chest_sigs
            del wrist_sigs

        except Exception as e:
            per_subject_summary[sid] = {"error": str(e)}

    summary = {
        "dataset": "WESAD",
        "status": "SUCCESS",
        "wesad_root": str(root),
        "subjects_parsed": len(per_subject_summary),
        "per_subject": per_subject_summary,
        "methodological_note": (
            "WESAD provides laboratory affective stress responses (EDA, Temp, ECG), not exercise fatigue "
            "or physical recovery. It serves as descriptive context for synthetic stress variation."
        ),
    }

    cache_path = out_dir / "wesad_summary.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
