#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import make_patterns
from pcam_model import PCAMModel, build_default_R


@dataclass
class SearchResult:
    spread: float
    raw_pi: np.ndarray
    name: str


def project_pi(raw_pi: np.ndarray, pi_min: float, pi_max: float) -> np.ndarray:
    pi = np.asarray(raw_pi, dtype=np.float64)
    pi = np.nan_to_num(pi, nan=1.0, posinf=pi_max, neginf=pi_min)
    pi = np.clip(pi, pi_min, pi_max)
    mean = float(np.mean(pi))
    if mean <= 1e-12 or not np.isfinite(mean):
        return np.ones_like(pi)
    return pi / mean


def spread_for_raw(H: np.ndarray,
                   raw_pi: np.ndarray,
                   pi_min: float,
                   pi_max: float) -> float:
    pi = project_pi(raw_pi, pi_min, pi_max)
    root = np.sqrt(pi)
    S = (root[:, None] * H) * root[None, :]
    S = 0.5 * (S + S.T)

    try:
        vals = np.linalg.eigvalsh(S)
    except np.linalg.LinAlgError:
        return float("inf")

    vals = vals[vals > 1e-9]
    if vals.size < 2:
        return float("inf")
    return float(vals[-1] / vals[0])


def condition_and_grad(H: np.ndarray,
                       y: np.ndarray,
                       pi_min: float,
                       pi_max: float) -> tuple[float, np.ndarray, np.ndarray]:
    raw = np.exp(np.clip(y, np.log(pi_min), np.log(pi_max)))
    pi = project_pi(raw, pi_min, pi_max)
    root = np.sqrt(pi)
    S = (root[:, None] * H) * root[None, :]
    S = 0.5 * (S + S.T)

    try:
        vals, vecs = np.linalg.eigh(S)
    except np.linalg.LinAlgError:
        return float("inf"), np.zeros_like(y), raw

    pos = np.flatnonzero(vals > 1e-9)
    if pos.size < 2:
        return float("inf"), np.zeros_like(y), raw

    imin = int(pos[0])
    imax = int(pos[-1])
    cond = float(vals[imax] / vals[imin])

    # Approximate derivative of log(condition) wrt log diagonal precision.
    grad = vecs[:, imax] * vecs[:, imax] - vecs[:, imin] * vecs[:, imin]
    grad = grad - np.mean(grad)
    if not np.isfinite(cond) or not np.all(np.isfinite(grad)):
        return float("inf"), np.zeros_like(y), raw
    return cond, grad, raw


def add_candidate(best: SearchResult,
                  H: np.ndarray,
                  raw_pi: np.ndarray,
                  name: str,
                  pi_min: float,
                  pi_max: float) -> SearchResult:
    spread = spread_for_raw(H, raw_pi, pi_min, pi_max)
    if spread < best.spread:
        return SearchResult(spread=spread, raw_pi=np.asarray(raw_pi).copy(), name=name)
    return best


def osborne_equilibrate(H: np.ndarray,
                        pi_min: float,
                        pi_max: float,
                        iters: int = 80) -> np.ndarray:
    n = H.shape[0]
    y = np.zeros(n, dtype=np.float64)
    off = np.abs(H).copy()
    np.fill_diagonal(off, 0.0)

    for _ in range(iters):
        scale = np.exp(0.5 * y)
        A = (scale[:, None] * off) * scale[None, :]
        row = np.sum(A, axis=1)
        positive = row[row > 1e-12]
        if positive.size == 0:
            break
        target = float(np.exp(np.mean(np.log(positive))))
        delta = -0.5 * (np.log(np.maximum(row, 1e-12)) - np.log(target))
        y += np.clip(delta, -0.25, 0.25)
        y -= np.mean(y)
        y = np.clip(y, np.log(pi_min), np.log(pi_max))

    return np.exp(y)


def gradient_descent(H: np.ndarray,
                     y0: np.ndarray,
                     pi_min: float,
                     pi_max: float,
                     iters: int,
                     lr0: float,
                     name: str) -> SearchResult:
    y = np.asarray(y0, dtype=np.float64).copy()
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y -= np.mean(y)
    best = SearchResult(float("inf"), np.exp(y), name)
    lr = lr0

    for _ in range(iters):
        cond, grad, raw = condition_and_grad(H, y, pi_min, pi_max)
        if cond < best.spread:
            best = SearchResult(cond, raw.copy(), name)
        gnorm = float(np.linalg.norm(grad))
        if gnorm <= 1e-12 or not np.isfinite(gnorm):
            break
        y -= lr * grad
        y -= np.mean(y)
        y = np.clip(y, np.log(pi_min), np.log(pi_max))
        lr *= 0.97

    return best


def adam_descent(H: np.ndarray,
                 y0: np.ndarray,
                 pi_min: float,
                 pi_max: float,
                 iters: int,
                 lr: float,
                 name: str) -> SearchResult:
    y = np.asarray(y0, dtype=np.float64).copy()
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y -= np.mean(y)
    y = np.clip(y, np.log(pi_min), np.log(pi_max))

    m = np.zeros_like(y)
    v = np.zeros_like(y)
    beta1 = 0.9
    beta2 = 0.999
    best = SearchResult(float("inf"), np.exp(y), name)

    for t in range(1, iters + 1):
        cond, grad, raw = condition_and_grad(H, y, pi_min, pi_max)
        if cond < best.spread:
            best = SearchResult(cond, raw.copy(), name)
        gnorm = float(np.linalg.norm(grad))
        if gnorm <= 1e-12 or not np.isfinite(gnorm):
            break

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        mhat = m / (1.0 - beta1 ** t)
        vhat = v / (1.0 - beta2 ** t)
        y -= lr * mhat / (np.sqrt(vhat) + 1e-8)
        y -= np.mean(y)
        y = np.clip(y, np.log(pi_min), np.log(pi_max))

    return best


def coordinate_descent(H: np.ndarray,
                       y0: np.ndarray,
                       pi_min: float,
                       pi_max: float,
                       passes: int,
                       step0: float,
                       name: str) -> SearchResult:
    y = np.asarray(y0, dtype=np.float64).copy()
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y -= np.mean(y)
    y = np.clip(y, np.log(pi_min), np.log(pi_max))
    best = SearchResult(
        spread_for_raw(H, np.exp(y), pi_min, pi_max),
        np.exp(y),
        name,
    )
    step = step0

    for _ in range(passes):
        improved = False
        for i in range(y.size):
            for sign in (1.0, -1.0):
                yy = y.copy()
                yy[i] += sign * step
                yy -= np.mean(yy)
                yy = np.clip(yy, np.log(pi_min), np.log(pi_max))
                spread = spread_for_raw(H, np.exp(yy), pi_min, pi_max)
                if spread < best.spread:
                    y = yy
                    best = SearchResult(spread, np.exp(yy), name)
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-4:
                break

    return best


def initial_candidates(H: np.ndarray,
                       rng: np.random.Generator,
                       random_starts: int,
                       pi_min: float,
                       pi_max: float) -> list[tuple[str, np.ndarray]]:
    eps = 1e-9
    n = H.shape[0]
    candidates: list[tuple[str, np.ndarray]] = []
    candidates.append(("identity", np.ones(n)))

    diag = np.abs(np.diag(H))
    candidates.append(("inverse_abs_diag", 1.0 / np.maximum(diag, eps)))

    row_l2 = np.sqrt(np.sum(H * H, axis=1))
    candidates.append(("inverse_row_l2", 1.0 / np.maximum(row_l2, eps)))

    row_l1 = np.sum(np.abs(H), axis=1)
    candidates.append(("inverse_row_l1", 1.0 / np.maximum(row_l1, eps)))

    try:
        H_inv = np.linalg.inv(H)
        inv_diag = np.abs(np.diag(H_inv))
        candidates.append(("abs_diag_invH", np.maximum(inv_diag, eps)))
        candidates.append(("inverse_abs_diag_invH", 1.0 / np.maximum(inv_diag, eps)))
    except np.linalg.LinAlgError:
        pass

    candidates.append(("osborne_equilibration", osborne_equilibrate(H, pi_min, pi_max)))

    try:
        vals, vecs = np.linalg.eigh(H)
        grad = vecs[:, -1] * vecs[:, -1] - vecs[:, 0] * vecs[:, 0]
        grad -= np.mean(grad)
        for scale in (0.5, 1.0, 2.0, 4.0):
            candidates.append((f"top_bottom_plus_{scale}", np.exp(scale * grad)))
            candidates.append((f"top_bottom_minus_{scale}", np.exp(-scale * grad)))
    except np.linalg.LinAlgError:
        pass

    scales = (0.25, 0.5, 0.9, 1.3, 1.8, 2.4)
    for i in range(random_starts):
        scale = scales[i % len(scales)]
        y = rng.standard_normal(n) * scale
        y -= np.mean(y)
        candidates.append((f"random_lognormal_{i:02d}", np.exp(y)))

    return candidates


def search_best(H: np.ndarray,
                pi_min: float,
                pi_max: float,
                rng: np.random.Generator,
                random_starts: int,
                grad_iters: int,
                coord_passes: int) -> SearchResult:
    best = SearchResult(
        spread=spread_for_raw(H, np.ones(H.shape[0]), pi_min, pi_max),
        raw_pi=np.ones(H.shape[0]),
        name="identity",
    )
    candidates = initial_candidates(H, rng, random_starts, pi_min, pi_max)

    for name, raw in candidates:
        best = add_candidate(best, H, raw, name, pi_min, pi_max)
        y0 = np.log(np.maximum(raw, 1e-12))

        gd = gradient_descent(H, y0, pi_min, pi_max, grad_iters, 0.55, f"{name}+gd")
        if gd.spread < best.spread:
            best = gd

        adam = adam_descent(H, y0, pi_min, pi_max, grad_iters, 0.16, f"{name}+adam")
        if adam.spread < best.spread:
            best = adam

    # Coordinate descent is expensive; run it only from the best basin found.
    coord = coordinate_descent(
        H,
        np.log(np.maximum(best.raw_pi, 1e-12)),
        pi_min,
        pi_max,
        coord_passes,
        0.75,
        f"{best.name}+coord",
    )
    if coord.spread < best.spread:
        best = coord

    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate diagonal anisotropy ceiling for P-04")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[7, 13, 31, 97, 211, 503, 1009])
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--n-anisotropy", type=int, default=16)
    parser.add_argument("--random-starts", type=int, default=12)
    parser.add_argument("--grad-iters", type=int, default=180)
    parser.add_argument("--coord-passes", type=int, default=14)
    args = parser.parse_args()

    all_ratios: list[float] = []

    for seed in args.seeds:
        X = make_patterns(K=args.K, N=args.N, seed=seed)
        R = build_default_R(N=args.N, seed=seed)
        model = PCAMModel(X, R)

        rng = np.random.default_rng(seed)
        indices = rng.choice(
            args.K,
            size=min(args.n_anisotropy, args.K),
            replace=False,
        ).tolist()

        for idx in indices:
            pattern = model.X[idx]
            H = model.hessian(pattern)
            H = 0.5 * (H + H.T)
            eig_H = np.linalg.eigvalsh(H)
            if eig_H.min() <= 0:
                continue

            base = spread_for_raw(H, np.ones(model.N), model.pi_min, model.pi_max)
            local_rng = np.random.default_rng(seed * 1009 + idx)
            best = search_best(
                H,
                model.pi_min,
                model.pi_max,
                local_rng,
                args.random_starts,
                args.grad_iters,
                args.coord_passes,
            )
            ratio = base / best.spread if best.spread > 0 else 0.0
            all_ratios.append(ratio)
            print(
                f"seed={seed:>5d} pattern={idx:>2d} "
                f"base={base:>9.4f} best={best.spread:>9.4f} "
                f"ratio={ratio:>7.4f}x method={best.name}"
            )

    if not all_ratios:
        print("No stable patterns found.")
        return 1

    mean_ratio = float(np.mean(all_ratios))
    min_ratio = float(np.min(all_ratios))
    max_ratio = float(np.max(all_ratios))
    print()
    print(f"mean best ratio: {mean_ratio:.4f}x")
    print(f"min best ratio:  {min_ratio:.4f}x")
    print(f"max best ratio:  {max_ratio:.4f}x")

    if mean_ratio < 1.2:
        print("Decision: below 1.2x; stop chasing public anisotropy.")
    elif mean_ratio >= 5.0:
        print("Decision: >=5x; focus everything on this geometry branch.")
    elif mean_ratio >= 2.0:
        print("Decision: >=2x; integrate the winning method into myteam.py.")
    else:
        print("Decision: modest ceiling; integrate only if retrieval remains untouched.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
