"""FitRec / Endomondo dataset inspection and summary preprocessing."""

import ast
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional
from Baseline.config import FITREC_JSON_PATH, OUTPUTS_DIR

def inspect_fitrec(
    filepath: Optional[Path] = None,
    max_records: int = 5000,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Inspect and extract compact descriptive distributions from FitRec dataset.

    Parses up to max_records line-delimited records to remain fast and memory-efficient.
    """
    path = Path(filepath) if filepath else FITREC_JSON_PATH
    out_dir = Path(output_dir) if output_dir else OUTPUTS_DIR

    if not path.exists():
        return {
            "dataset": "FitRec",
            "status": "SKIPPED",
            "reason": f"File not found at {path}",
            "records_parsed": 0,
        }

    sports_count: Dict[str, int] = {}
    genders_count: Dict[str, int] = {}
    users_seen = set()

    heart_rates = []
    durations_min = []
    speeds_kmh = []

    records_parsed = 0
    parse_errors = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if records_parsed >= max_records:
                break
            line = line.strip()
            if not line:
                continue

            try:
                # File uses Python literal dict format per line
                record = ast.literal_eval(line)
            except Exception:
                parse_errors += 1
                continue

            records_parsed += 1

            sport = str(record.get("sport", "unknown")).lower()
            sports_count[sport] = sports_count.get(sport, 0) + 1

            gender = str(record.get("gender", "unknown")).lower()
            genders_count[gender] = genders_count.get(gender, 0) + 1

            uid = record.get("userId")
            if uid is not None:
                users_seen.add(uid)

            # Heart rates
            hr_series = record.get("heart_rate") or record.get("tar_heart_rate")
            if isinstance(hr_series, list) and hr_series:
                # Sample middle value or mean
                valid_hrs = [float(h) for h in hr_series if 30 <= float(h) <= 230]
                if valid_hrs:
                    heart_rates.append(sum(valid_hrs) / len(valid_hrs))

            # Duration from timestamps
            ts_series = record.get("timestamp")
            if isinstance(ts_series, list) and len(ts_series) >= 2:
                try:
                    dur_min = (float(ts_series[-1]) - float(ts_series[0])) / 60.0
                    if 1.0 <= dur_min <= 360.0:
                        durations_min.append(dur_min)
                except (ValueError, TypeError):
                    pass

            # Speed
            speed_series = record.get("speed") or record.get("derived_speed")
            if isinstance(speed_series, list) and speed_series:
                valid_speeds = [float(s) for s in speed_series if 0.5 <= float(s) <= 80.0]
                if valid_speeds:
                    speeds_kmh.append(sum(valid_speeds) / len(valid_speeds))

    # Helper stats
    def calc_stats(vals):
        if not vals:
            return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        n = len(vals)
        mean_val = sum(vals) / n
        std_val = math.sqrt(sum((v - mean_val) ** 2 for v in vals) / max(1, n - 1)) if n > 1 else 0.0
        return {
            "count": n,
            "mean": round(mean_val, 2),
            "std": round(std_val, 2),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
            "median": round(sorted(vals)[n // 2], 2),
        }

    summary = {
        "dataset": "FitRec",
        "status": "SUCCESS",
        "source_path": str(path),
        "records_parsed": records_parsed,
        "parse_errors": parse_errors,
        "unique_users_in_sample": len(users_seen),
        "sport_distribution": sports_count,
        "gender_distribution": genders_count,
        "heart_rate_bpm_stats": calc_stats(heart_rates),
        "duration_min_stats": calc_stats(durations_min),
        "speed_kmh_stats": calc_stats(speeds_kmh),
    }

    # Save compact JSON cache
    cache_path = out_dir / "fitrec_summary.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
