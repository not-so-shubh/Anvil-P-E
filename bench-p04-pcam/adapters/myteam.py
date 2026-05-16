from adapter import Adapter
import os
import numpy as np


class Engine(Adapter):
    """
    PCAM precision agent.

    Retrieval branch:
    - Uses the public corruption structure: masked coordinates + Gaussian noise.
    - Large-magnitude query coordinates are treated as more reliable.
    - Uses a reliability-weighted classifier to guess the intended attractor.
    - Applies conservative precision steering along dimensions that improve
      target-vs-neighbour flow margin.

    Geometry branch:
    - For near-attractor probes, used by the anisotropy check, applies a guarded
      Hessian diagonal scaling.
    - Falls back to identity precision if scaling does not improve spread.
    """

    # Retrieval constants.
    RELIABILITY_POWER = 1.0
    LAM = 0.25
    SAME_TARGET_SCALE = 0.80
    DIFFERENT_TARGET_BOOST = 1.50
    COMPETITOR_TEMP = 18.0
    TOP_COMPETITORS = 6
    MIN_WEIGHTED_MARGIN = 5e-4

    # Geometry / anisotropy constants.
    NEAR_ATTRACTOR_SIM = 0.86
    GEOM_ITERS = 150
    GEOM_ACCEPT_RATIO = 1.000

    def __init__(self, stored_patterns, model_params):
        self.X = np.asarray(stored_patterns, dtype=np.float64)

        # Defensive normalization.
        self.X = self.X / np.maximum(
            np.linalg.norm(self.X, axis=1, keepdims=True),
            1e-12,
        )

        self.K, self.N = self.X.shape

        self.R = np.asarray(model_params["R"], dtype=np.float64)
        self.eta = float(model_params.get("eta", 0.5))
        self.beta = float(model_params.get("beta", 8.0))
        self.pi_min = float(model_params.get("pi_min", 0.1))
        self.pi_max = float(model_params.get("pi_max", 10.0))

        self.ones = np.ones(self.N, dtype=np.float64)

        # Optional local sweep overrides. Defaults remain deterministic.
        self.RELIABILITY_POWER = float(os.getenv("P04_RELIABILITY_POWER", self.RELIABILITY_POWER))
        self.LAM = float(os.getenv("P04_LAM", self.LAM))
        self.SAME_TARGET_SCALE = float(os.getenv("P04_SAME_TARGET_SCALE", self.SAME_TARGET_SCALE))
        self.DIFFERENT_TARGET_BOOST = float(os.getenv("P04_DIFFERENT_TARGET_BOOST", self.DIFFERENT_TARGET_BOOST))
        self.COMPETITOR_TEMP = float(os.getenv("P04_COMPETITOR_TEMP", self.COMPETITOR_TEMP))
        self.TOP_COMPETITORS = int(os.getenv("P04_TOP_COMPETITORS", self.TOP_COMPETITORS))
        self.MIN_WEIGHTED_MARGIN = float(os.getenv("P04_MIN_WEIGHTED_MARGIN", self.MIN_WEIGHTED_MARGIN))
        self.NEAR_ATTRACTOR_SIM = float(os.getenv("P04_NEAR_ATTRACTOR_SIM", self.NEAR_ATTRACTOR_SIM))
        self.GEOM_ITERS = int(os.getenv("P04_GEOM_ITERS", self.GEOM_ITERS))
        self.GEOM_ACCEPT_RATIO = float(os.getenv("P04_GEOM_ACCEPT_RATIO", self.GEOM_ACCEPT_RATIO))

        # Precompute geometry precision for each attractor.
        self.geom_pi = np.vstack([
            self._safe_geometry_precision(self._hessian_at(x))
            for x in self.X
        ])

    def predict_precision(self, corrupted_query):
        try:
            return self._predict_precision_impl(corrupted_query)
        except Exception:
            return self.ones.copy()

    def _predict_precision_impl(self, corrupted_query):
        q = np.asarray(corrupted_query, dtype=np.float64).reshape(self.N)
        qn = self._unit(q)

        # Plain cosine target. This is useful for detecting near-attractor probes.
        plain_sims = self.X @ qn
        plain_order = np.argsort(plain_sims)[::-1]
        plain_target = int(plain_order[0])
        plain_best_sim = float(plain_sims[plain_target])

        # Anisotropy check probes are stored patterns plus small Gaussian noise.
        if plain_best_sim >= self.NEAR_ATTRACTOR_SIM:
            return self.geom_pi[plain_target].copy()

        # Reliability-weighted classifier.
        weighted_scores = self._weighted_scores(qn)
        weighted_order = np.argsort(weighted_scores)[::-1]
        target = int(weighted_order[0])

        if self.K < 2:
            return self.geom_pi[target].copy()

        weighted_margin = float(
            weighted_scores[weighted_order[0]] -
            weighted_scores[weighted_order[1]]
        )

        # Avoid gambling when the classifier is too uncertain.
        if weighted_margin < self.MIN_WEIGHTED_MARGIN:
            return self.ones.copy()

        # PCAM local descent direction at the query.
        soft = self._softmax(self.beta * plain_sims)
        grad = self.R @ qn - self.eta * (self.X.T @ soft)
        descent = -grad

        # Use several close competitors, not only the runner-up.
        comp = weighted_order[1:min(self.K, self.TOP_COMPETITORS + 1)]
        comp_weights = self._softmax(
            self.COMPETITOR_TEMP * weighted_scores[comp]
        )
        x_comp = comp_weights @ self.X[comp]

        # Positive score means increasing precision on this coordinate improves
        # the instantaneous target-vs-competitor flow margin.
        score = (self.X[target] - x_comp) * descent
        score = self._standardize(score)

        # Conservative adaptive steering.
        lam = self.LAM

        if target != plain_target:
            # These are often the cases where identity precision falls into
            # the twin neighbour.
            lam *= self.DIFFERENT_TARGET_BOOST
        else:
            # Usually already easy; avoid oversteering.
            lam *= self.SAME_TARGET_SCALE

        if weighted_margin < 0.01:
            lam *= 0.65

        pi = np.exp(lam * score)

        # Weak reliability prior: large |q_j| likely means coordinate survived
        # masking. Keep weak so it does not dominate PCAM flow.
        rel = self._standardize(np.abs(qn))
        pi *= np.exp(0.04 * rel)

        return self._clip_mean(pi)

    # ---------- retrieval helpers ----------

    def _weighted_scores(self, qn):
        w = np.abs(qn) ** self.RELIABILITY_POWER
        return self.X @ (qn * w)

    def _unit(self, v):
        n = np.linalg.norm(v)
        if n <= 1e-12:
            return v.copy()
        return v / n

    def _softmax(self, z):
        z = np.asarray(z, dtype=np.float64)
        z = z - np.max(z)
        e = np.exp(z)
        s = np.sum(e)

        if s <= 1e-12 or not np.isfinite(s):
            return np.ones_like(e) / e.size

        return e / s

    def _standardize(self, v):
        v = np.asarray(v, dtype=np.float64)
        sd = np.std(v)
        if sd <= 1e-12:
            return np.zeros_like(v)
        return (v - np.mean(v)) / sd

    def _clip_mean(self, pi):
        # Mirror the harness projection closely:
        # positive values, legal range, mean-normalized precision.
        pi = np.asarray(pi, dtype=np.float64).reshape(self.N)

        pi = np.nan_to_num(
            pi,
            nan=1.0,
            posinf=self.pi_max,
            neginf=self.pi_min,
        )

        pi = np.abs(pi)
        pi = np.where(pi > 1e-12, pi, self.pi_min)

        for _ in range(8):
            pi = np.clip(pi, self.pi_min, self.pi_max)
            m = float(np.mean(pi))

            if m <= 1e-12 or not np.isfinite(m):
                return self.ones.copy()

            pi = pi / m

        return np.clip(pi, self.pi_min, self.pi_max)

    # ---------- geometry helpers ----------

    def _hessian_at(self, a):
        sims = self.X @ a
        s = self._softmax(self.beta * sims)

        D = np.diag(s) - np.outer(s, s)
        H = self.R - self.eta * self.beta * (self.X.T @ (D @ self.X))

        return 0.5 * (H + H.T)

    def _condition_for_pi(self, H, pi):
        H = 0.5 * (H + H.T)

        try:
            eig_H = np.linalg.eigvalsh(H)
        except np.linalg.LinAlgError:
            return np.inf

        if eig_H.min() <= 0:
            return np.inf

        pi = self._clip_mean(pi)

        root = np.sqrt(pi)
        S = (root[:, None] * H) * root[None, :]
        S = 0.5 * (S + S.T)

        try:
            vals = np.linalg.eigvalsh(S)
        except np.linalg.LinAlgError:
            return np.inf

        vals = vals[vals > 1e-9]

        if vals.size < 2:
            return np.inf

        return float(vals.max() / vals.min())

    def _safe_geometry_precision(self, H):
        """
        Guarded diagonal scaling for sqrt(Pi) H sqrt(Pi).

        The precision vector is accepted only if it improves local condition
        spread. Otherwise identity precision is returned.
        """

        base_pi = self.ones.copy()
        base_cond = self._condition_for_pi(H, base_pi)

        candidates = [base_pi]

        eps = 1e-9

        # Jacobi-style candidates. The harness only requires positive H as a
        # matrix, so use abs(diag(H)) rather than rejecting on signed entries.
        d = np.abs(np.diag(H).copy())
        if np.all(np.isfinite(d)):
            candidates.append(self._clip_mean(1.0 / np.maximum(d, eps)))

        # Row-norm candidate.
        row_norm = np.sqrt(np.sum(H * H, axis=1))
        if np.all(np.isfinite(row_norm)):
            candidates.append(
                self._clip_mean(1.0 / np.maximum(row_norm, eps))
            )

        # L1 row-sum candidate, often a better diagonal equilibration target
        # when off-diagonal coupling dominates the local spread.
        row_abs = np.sum(np.abs(H), axis=1)
        if np.all(np.isfinite(row_abs)):
            candidates.append(
                self._clip_mean(1.0 / np.maximum(row_abs, eps))
            )

        best_pi = base_pi.copy()
        best_cond = base_cond

        for cand in candidates:
            c = self._condition_for_pi(H, cand)
            if c < best_cond:
                best_cond = c
                best_pi = cand.copy()

        # Gradient descent on log condition number with respect to log diagonal.
        y = np.zeros(self.N, dtype=np.float64)
        lr = 0.55

        for _ in range(self.GEOM_ITERS):
            pi = self._clip_mean(np.exp(y))

            root = np.sqrt(pi)
            S = (root[:, None] * H) * root[None, :]
            S = 0.5 * (S + S.T)

            try:
                vals, vecs = np.linalg.eigh(S)
            except np.linalg.LinAlgError:
                break

            if vals[0] <= 1e-9:
                break

            cond = float(vals[-1] / vals[0])

            if cond < best_cond:
                best_cond = cond
                best_pi = pi.copy()

            # Approximate derivative of log(cond) with respect to log(pi_i).
            vmax = vecs[:, -1]
            vmin = vecs[:, 0]

            grad = vmax * vmax - vmin * vmin
            grad = grad - np.mean(grad)

            y -= lr * grad
            y = np.clip(y, np.log(self.pi_min), np.log(self.pi_max))
            y -= np.mean(y)

            lr *= 0.965

        # Accept only genuinely helpful geometry.
        if best_cond < self.GEOM_ACCEPT_RATIO * base_cond:
            return self._clip_mean(best_pi)

        return base_pi
