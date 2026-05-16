# PCAM Precision Agent

This agent uses a hybrid reliability-weighted retrieval policy and guarded Hessian geometry policy.

The public synthetic corruption process masks dimensions and adds Gaussian noise, so large-magnitude coordinates in the corrupted query are more likely to be reliable. The agent predicts the intended attractor using a weighted cosine classifier:

    score_i = <x_i, q * |q|^p>

Then it computes the local PCAM descent direction and raises precision on coordinates whose flow improves the target-vs-neighbour margin. If the reliability-weighted target differs from the plain cosine target, the agent steers more strongly, because those are the cases where identity precision often falls into a twin neighbour. If both targets agree, it steers conservatively to avoid seed-level regressions.

For near-attractor probes, used by the anisotropy check, the agent precomputes a diagonal Hessian scaling for each stored pattern. The scaling is accepted only when it actually reduces the local condition spread of sqrt(Pi) H sqrt(Pi); otherwise the agent returns identity precision.

The implementation is NumPy-only and does not modify or retrain the frozen PCAM model.

## Run

Quick check:

    python self_check.py --adapter adapters.myteam:Engine --quick

Full multi-seed check:

    python run.py --adapter adapters.myteam:Engine \
      --seeds 7 13 31 97 211 503 1009 \
      --out report.json

Report summary:

    python scripts/summarize_report.py results/final_20_seed_report.json \
      --out results/final_20_seed_summary.md
