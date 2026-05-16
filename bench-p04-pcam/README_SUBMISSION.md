# PCAM PrecisionPilot Submission

Entrypoint:

```text
adapters.myteam:Engine
```

## Core Idea

PrecisionPilot uses precision as inference-time steering.

For corrupted retrieval queries, the agent first estimates the intended attractor with a reliability-weighted classifier. It then computes the frozen PCAM local descent direction and increases precision on coordinates that improve the target-vs-neighbour flow margin.

For near-clean anisotropy probes, the agent switches to a guarded Hessian branch. Geometry precision is accepted only when it reduces the measured spread of:

```text
sqrt(Pi) H sqrt(Pi)
```

Otherwise the agent returns identity precision.

## Why This Uses PCAM Precision Meaningfully

The retrieval branch uses the frozen PCAM gradient:

```text
grad = R @ q - eta * X.T @ softmax(beta * X @ q)
```

Precision is increased on coordinates where the local descent direction improves the target-vs-competitor margin.

This is not retraining and does not modify the PCAM model.

## Results

| Run | Mean Δ | Min Δ | Mean Spread | Min Spread | Retrieval Score |
|---|---:|---:|---:|---:|---:|
| 7-seed public | +0.0695 | +0.0533 | 1.0284x | 1.0220x | 70 / 70 |
| 20-seed stress | +0.1007 | +0.0533 | 1.0256x | 1.0164x | 70 / 70 |

## Safety

NumPy only.
Deterministic when no sweep environment variables are set.
No PCAM model changes.
No benchmark harness changes.
Falls back to identity precision on invalid input.
