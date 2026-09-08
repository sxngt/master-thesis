"""Statistical layer of the reward coach (v5): pure functions, unit-tested.

Everything the decision layer and the report need that is *computed* rather
than *prompted* lives here, so that the LLM reasons on evidence with error
bars instead of on raw numbers:

- measurement uncertainty of the objective J from one evaluation
  (binomial for rates, sd/sqrt(n) for means),
- detrended effect of an intervention (observed dJ minus the dJ the local
  learning trend would have produced anyway),
- a per-parameter effect ledger with normal-normal shrinkage towards a prior
  (posterior mean and sd of "what this move does in this run"),
- a saturating learning-curve fit for the projected final J,
- calibration of the LLM's own dJ predictions,
- an in-run command-tracking ratio (measured / commanded speed).

No simulator or LLM dependency (CLAUDE.md: coach logic must be testable
without either).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

Trace = list[tuple[int, float]]  # (env step, J)


# --------------------------------------------------------------- uncertainty
def binomial_se(p: float, n: int) -> float:
    """Standard error of a rate estimated from n Bernoulli episodes (Wilson-free,
    floored at the SE of 1/n so p = 0 or 1 does not report zero uncertainty)."""
    if n <= 0:
        return 0.0
    p = min(max(p, 1.0 / n), 1.0 - 1.0 / n) if n > 1 else 0.5
    return math.sqrt(p * (1.0 - p) / n)


def objective_se(
    kpi: dict[str, float],
    weights: dict[str, float],
    n: int,
    sd_suffix: str = "_sd",
    rate_keys: tuple[str, ...] = ("success_rate",),
) -> float:
    """SE of J = sum w_m KPI_m from one evaluation of n episodes.

    Rates use the binomial SE; means use ``<kpi><sd_suffix>`` / sqrt(n) when
    the evaluator reports the per-episode sd, else contribute 0 (unknown).
    Terms are treated as independent (an upper bound is not needed: the
    process-noise estimate from the trace is combined with this later).
    """
    if n <= 0:
        return 0.0
    var = 0.0
    for m, w in weights.items():
        if m in rate_keys:
            se = binomial_se(float(kpi.get(m, 0.0)), n)
        else:
            sd = kpi.get(f"{m}{sd_suffix}")
            se = float(sd) / math.sqrt(n) if sd is not None else 0.0
        var += (w * se) ** 2
    return math.sqrt(var)


# -------------------------------------------------------------------- trend
def linear_trend(trace: Trace, points: int = 4) -> tuple[float, float]:
    """OLS slope (per env step) and intercept of J over the last ``points`` evaluations.

    Returns (0, last J) with fewer than two points."""
    pts = trace[-points:]
    if len(pts) < 2:
        return 0.0, (pts[-1][1] if pts else 0.0)
    x = np.array([s for s, _ in pts], dtype=float)
    y = np.array([j for _, j in pts], dtype=float)
    x0 = x - x.mean()
    denom = float((x0**2).sum())
    slope = float((x0 * (y - y.mean())).sum() / denom) if denom > 0 else 0.0
    return slope, float(y.mean() - slope * x.mean())


def detrended_effect(
    trace_before: Trace, step_after: int, j_after: float, points: int = 4
) -> tuple[float, float]:
    """Effect of an intervention applied after ``trace_before[-1]``.

    Returns (effect, expected_dj): expected_dj is what the pre-intervention
    linear trend predicts J would have gained by ``step_after`` without the
    intervention; effect = (j_after - j_before) - expected_dj. On a rising
    curve a kept move that merely rode the trend therefore scores ~0, and a
    move that stalled learning scores negative even if J did not fall.
    """
    step_before, j_before = trace_before[-1]
    slope, _ = linear_trend(trace_before, points)
    expected = slope * (step_after - step_before)
    return (j_after - j_before) - expected, expected


# ---------------------------------------------------------- learning curve
@dataclass
class CurveFit:
    asymptote: float
    tau_steps: float
    j0: float
    t0: int
    r2: float
    n: int

    def predict(self, t: float) -> float:
        decay = math.exp(-(t - self.t0) / self.tau_steps)
        return self.asymptote + (self.j0 - self.asymptote) * decay


def fit_saturating(trace: Trace, min_points: int = 4, tau_grid: int = 40) -> CurveFit | None:
    """Fit J(t) = A + B exp(-(t - t0)/tau) to a policy lineage by least squares.

    tau is searched on a log grid from one evaluation interval to 10x the
    span of the trace; A and B are solved in closed form for each tau. Returns
    None with fewer than ``min_points`` evaluations or a degenerate span. The
    asymptote is the projected J of *this* lineage under no further change —
    the report shows it next to the objective ceiling so the LLM can tell
    "still climbing" from "saturated below the ceiling".
    """
    if len(trace) < min_points:
        return None
    t = np.array([s for s, _ in trace], dtype=float)
    y = np.array([j for _, j in trace], dtype=float)
    span = float(t[-1] - t[0])
    if span <= 0:
        return None
    dt_min = max(float(np.min(np.diff(t))), 1.0)
    best = None
    for tau in np.geomspace(dt_min, 10.0 * span, tau_grid):
        e = np.exp(-(t - t[0]) / tau)
        X = np.stack([np.ones_like(e), e], axis=1)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = float(((X @ coef - y) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, tau, coef)
    sse, tau, (a, b) = best
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    return CurveFit(
        asymptote=float(a), tau_steps=float(tau), j0=float(a + b), t0=int(t[0]), r2=r2, n=len(trace)
    )


# ------------------------------------------------------------------ ledger
def move_direction(before: float, after: float) -> str:
    """'up' if the value increased (a penalty weight becoming less negative is
    'up' = weaker), 'down' otherwise."""
    return "up" if after > before else "down"


@dataclass
class EffectLedger:
    """Per-(parameter, direction) record of detrended effects with shrinkage.

    posterior(param, dir) combines the in-run observations x_1..x_n (each with
    measurement sd ``obs_sd``) with a normal prior N(mu0, sd0^2): the usual
    normal-normal update, so one lucky observation cannot dominate and a
    parameter never tried reports the prior. Priors default to N(0, prior_sd)
    and may be overridden per (param, dir) from earlier batches' logs
    (``effect_prior`` in the coach config, e.g. the delta_j_by_direction table
    of the log study).
    """

    prior_sd: float = 0.1
    priors: dict[tuple[str, str], tuple[float, float]] = field(default_factory=dict)
    obs: dict[tuple[str, str], list[tuple[float, float]]] = field(default_factory=dict)

    def add(self, param: str, direction: str, effect: float, obs_sd: float) -> None:
        self.obs.setdefault((param, direction), []).append((effect, max(obs_sd, 1e-6)))

    def posterior(self, param: str, direction: str) -> tuple[float, float, int]:
        mu0, sd0 = self.priors.get((param, direction), (0.0, self.prior_sd))
        prec = 1.0 / sd0**2
        num = mu0 * prec
        xs = self.obs.get((param, direction), [])
        for x, s in xs:
            num += x / s**2
            prec += 1.0 / s**2
        return num / prec, math.sqrt(1.0 / prec), len(xs)

    def rows(self, params: list[str] | None = None) -> list[str]:
        if params is None:
            keys = sorted(self.obs)
        else:
            keys = [(p, d) for p in params for d in ("up", "down")]
        out = []
        for p, d in keys:
            mean, sd, n = self.posterior(p, d)
            if n == 0 and params is None:
                continue
            src = f"{n} obs" if n else "prior only"
            out.append(f"{p} {d}: effect on J {mean:+.3f} +- {sd:.3f} ({src})")
        return out


# ------------------------------------------------------------- calibration
def calibration(pairs: list[tuple[float, float]]) -> dict[str, float]:
    """Agreement between predicted and realised (detrended) dJ."""
    if not pairs:
        return {"n": 0}
    pred = np.array([p for p, _ in pairs])
    real = np.array([r for _, r in pairs])
    sign_ok = float(np.mean(np.sign(pred) == np.sign(real)))
    return {
        "n": len(pairs),
        "sign_accuracy": sign_ok,
        "mae": float(np.mean(np.abs(pred - real))),
        "bias": float(np.mean(pred - real)),
    }


# ---------------------------------------------------------------- tracking
def tracking_ratio(
    pairs: list[tuple[float, float, float]], min_success: float = 0.5
) -> tuple[float, int] | None:
    """Slope through the origin of measured vs commanded speed over reports
    whose success rate is at least ``min_success`` (a fallen policy does not
    track anything). Returns (ratio, n) or None."""
    pts = [(c, v) for c, v, s in pairs if s >= min_success and c > 0]
    if not pts:
        return None
    c = np.array([x for x, _ in pts])
    v = np.array([y for _, y in pts])
    return float((c * v).sum() / (c * c).sum()), len(pts)
