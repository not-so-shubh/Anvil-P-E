# PCAM PrecisionPilot Submission

## Entrypoint

```text
adapters.myteam:Engine
```

## Method

PrecisionPilot uses precision as inference-time steering for the frozen PCAM model.

- Guarded reliability-weighted retrieval precision
- Competitor-aware precision steering
- Conservative near-attractor geometry fallback
- Precision vectors are positive, clipped-safe, and mean-normalized by the harness

For corrupted retrieval queries, the agent predicts the intended attractor with a reliability-weighted classifier, then uses the local PCAM descent direction to increase precision on coordinates that improve the target-vs-competitor flow margin. For near-attractor probes, it uses a guarded Hessian geometry fallback and returns identity precision if the geometry candidate does not improve local spread.

The submission does not retrain or modify the PCAM model.

## Final Result

20-seed stress run:

| Metric | Value |
|---|---:|
| Mean Δ accuracy | +0.1007 |
| Min Δ accuracy | +0.0533 |
| Mean spread reduction | 1.0256x |
| Min spread reduction | 1.0164x |
| Retrieval score | 70 / 70 |
| Anisotropy score | 0.22 / 20 |
| Total automated score | 70.22 / 90 |

## Geometry Ceiling Diagnostic

`scripts/geometry_ceiling.py` is diagnostic-only. It does not affect scoring or the submitted adapter.

The script aggressively searches diagonal Hessian preconditioners using identity, inverse diagonal, row norms, inverse-Hessian diagonal, Osborne equilibration, eigenvector updates, gradient descent, Adam, coordinate descent, and deterministic random log-normal starts.

Public 7-seed ceiling result:

```text
mean best ratio: 1.0285x
min best ratio:  1.0174x
max best ratio:  1.0441x
```

Conclusion: public diagonal anisotropy headroom appears low under the clipped, mean-normalized diagonal precision constraint. The final submitted agent therefore prioritizes robust retrieval instead of adding slow geometry code.

## Reproduction

Quick check:

```bash
python self_check.py --adapter adapters.myteam:Engine --quick
```

Final 20-seed stress run:

```bash
python run.py --adapter adapters.myteam:Engine \
  --seeds 7 13 31 42 97 101 202 211 303 404 503 777 1009 1337 2027 2718 3141 4096 9001 9999 \
  --out results/final_20_seed_report.json
```

Geometry ceiling diagnostic:

```bash
python scripts/geometry_ceiling.py --seeds 7 13 31 97 211 503 1009
```

## Safety

- NumPy only
- Deterministic with default settings
- No PCAM model changes
- No benchmark harness changes
- Falls back to identity precision on invalid input
