#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    rows = report["per_seed"]
    agg = report["aggregated"]
    score = report.get("score", {})

    lines = []
    lines.append("# P-04 Final Report Summary")
    lines.append("")
    lines.append("| Seed | Baseline | Agent | Δ | Base Spread | Agent Spread | Ratio |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")

    for r in rows:
        lines.append(
            f"| {r['seed']} | "
            f"{r['baseline_accuracy']:.3f} | "
            f"{r['agent_accuracy']:.3f} | "
            f"{r['delta']:+.3f} | "
            f"{r['spread_baseline']:.2f} | "
            f"{r['spread_agent']:.2f} | "
            f"{r['spread_reduction']:.2f}x |"
        )

    lines.append("")
    lines.append("## Aggregated")
    lines.append("")
    lines.append(f"- Mean Δ: `{agg['mean_delta']:+.4f}`")
    lines.append(f"- Min Δ: `{agg['min_delta']:+.4f}`")
    lines.append(f"- Mean spread reduction: `{agg['mean_spread']:.4f}x`")
    lines.append(f"- Min spread reduction: `{agg['min_spread']:.4f}x`")

    if score:
        lines.append(f"- Retrieval points: `{score.get('retrieval_pts')}`")
        lines.append(f"- Anisotropy points: `{score.get('anisotropy_pts')}`")
        lines.append(f"- Total automated: `{score.get('total_automated')} / {score.get('max_automated')}`")

    text = "\n".join(lines) + "\n"

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)

    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
