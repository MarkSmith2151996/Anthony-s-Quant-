"""Final leaderboard and summary reporting."""

import logging
from datetime import datetime
from pathlib import Path

from pipeline.config import _save_json

logger = logging.getLogger("mcq_pipeline")


def save_final_leaderboard(validated_strategies, output_dir: str):
    """Save the final leaderboard to reports/leaderboard.json."""
    if not validated_strategies:
        return

    leaderboard = {
        "generated_at": datetime.now().isoformat(),
        "type": "final_leaderboard",
        "count": len(validated_strategies),
        "entries": [],
    }

    for rank, (sd, res) in enumerate(validated_strategies, 1):
        mc_score = res.get("mc_score")
        entry = {
            "rank": rank,
            "strategy_name": res["strategy_name"],
            "mc_composite_score": round(mc_score.composite_score, 2) if mc_score else 0,
            "mc_grade": mc_score.grade if mc_score else "N/A",
            "mc_pass_rate": round(mc_score.mc_result.prop_firm_pass_rate * 100, 1)
            if mc_score else 0,
            "mc_median_return": round(mc_score.mc_result.median_return, 2) if mc_score else 0,
            "mc_probability_of_ruin": round(mc_score.mc_result.probability_of_ruin * 100, 1)
            if mc_score else 0,
            "wf_efficiency": round(res.get("wf_efficiency", 0), 3),
            "pct_profitable_windows": round(res.get("pct_profitable_windows", 0), 1),
            "regime_sensitivity": round(res.get("regime_sensitivity", 1.0), 3),
            "overfit_verdict": res.get("overfit_verdict", "unknown"),
        }

        # Include strategy definition if available
        if sd:
            entry["strategy"] = sd

        leaderboard["entries"].append(entry)

    filepath = Path(output_dir) / "leaderboard.json"
    _save_json(leaderboard, filepath)
    logger.info("Final leaderboard saved to %s", filepath)


def print_final_summary(stage_timings: dict[str, float], total_time: float):
    """Print a summary table with timing per stage."""
    print()
    print("=" * 62)
    print("  PIPELINE SUMMARY")
    print("=" * 62)
    print()
    print(f"  {'Stage':<30s}  {'Time':>10s}  {'% of Total':>10s}")
    print("  " + "-" * 52)

    for stage, elapsed in stage_timings.items():
        pct = (elapsed / total_time * 100) if total_time > 0 else 0
        time_str = f"{elapsed:.1f}s"
        if elapsed >= 60:
            mins = int(elapsed // 60)
            secs = elapsed % 60
            time_str = f"{mins}m {secs:.0f}s"
        print(f"  {stage:<30s}  {time_str:>10s}  {pct:>9.1f}%")

    print("  " + "-" * 52)

    total_str = f"{total_time:.1f}s"
    if total_time >= 60:
        mins = int(total_time // 60)
        secs = total_time % 60
        total_str = f"{mins}m {secs:.0f}s"
    print(f"  {'TOTAL':<30s}  {total_str:>10s}  {'100.0':>9s}%")
    print()
    print(f"  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)
    print()
