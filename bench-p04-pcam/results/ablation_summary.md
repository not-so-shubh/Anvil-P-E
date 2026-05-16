# Ablation Summary

| Agent | Mean Δ | Min Δ | Spread | Diagnosis |
|---|---:|---:|---:|---|
| Identity precision | +0.000 | +0.000 | 1.000x | baseline |
| Geometry only | low | low | ~1.02x | helps anisotropy slightly |
| Reliability classifier only | not run | not run | not run | omitted from final ablation |
| Flow-margin final | +0.1007 | +0.0533 | 1.0256x | final submission |

## Final design choice

The final agent uses flow-margin steering for corrupted retrieval queries and a guarded Hessian branch for near-attractor probes.

The main scoring strength is retrieval consistency. The public diagonal anisotropy check gives only a small spread improvement, so the agent prioritizes avoiding retrieval regressions while preserving spread above baseline.
