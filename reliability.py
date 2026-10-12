"""Distribution fitting and reliability metrics for Time-to-Failure (TTF) data.

This module has no Streamlit dependency so it can be reused or tested on its own.
All data are treated as complete (uncensored) failure times and parameters are
estimated by Maximum Likelihood (MLE) through ``scipy.stats``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

DISTRIBUTIONS = ["Exponential", "Weibull", "Normal", "Lognormal", "Gamma"]
# Distributions whose support is t > 0 (when the location parameter is 0).
POSITIVE_ONLY = {"Exponential", "Weibull", "Lognormal", "Gamma"}
CRITERIA = {"AIC": "aic", "BIC": "bic", "Kolmogorov-Smirnov": "ks_stat"}


@dataclass
class FitResult:
    """Outcome of fitting one distribution to the data."""

    name: str
    dist: object | None = None  # frozen scipy.stats distribution
    params: dict[str, float] = field(default_factory=dict)
    n_params: int = 0
    loglik: float = np.nan
    aic: float = np.nan
    bic: float = np.nan
    ks_stat: float = np.nan
    ks_pvalue: float = np.nan
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def params_text(self, digits: int = 5) -> str:
        return ", ".join(f"{k} = {v:.{digits}g}" for k, v in self.params.items())


# --------------------------------------------------------------------------
# Individual fitters: return (frozen distribution, parameter dict, #free params)
# --------------------------------------------------------------------------
def _fit_exponential(t, free_loc):
    loc, scale = stats.expon.fit(t) if free_loc else stats.expon.fit(t, floc=0)
    params = {"λ (laju kegagalan)": 1.0 / scale, "θ = 1/λ (skala)": scale}
    if free_loc:
        params["γ (lokasi)"] = loc
    return stats.expon(loc=loc, scale=scale), params, 1 + free_loc


def _fit_weibull(t, free_loc):
    kwargs = {} if free_loc else {"floc": 0}
    beta, loc, eta = stats.weibull_min.fit(t, **kwargs)
    params = {"β (bentuk)": beta, "η (skala)": eta}
    if free_loc:
        params["γ (lokasi)"] = loc
    return stats.weibull_min(beta, loc=loc, scale=eta), params, 2 + free_loc


def _fit_normal(t, free_loc):
    mu, sigma = stats.norm.fit(t)
    return stats.norm(loc=mu, scale=sigma), {"μ (rata-rata)": mu, "σ (simp. baku)": sigma}, 2


def _fit_lognormal(t, free_loc):
    kwargs = {} if free_loc else {"floc": 0}
    sigma, loc, scale = stats.lognorm.fit(t, **kwargs)
    params = {"μ (rata-rata ln t)": float(np.log(scale)), "σ (simp. baku ln t)": sigma}
    if free_loc:
        params["γ (lokasi)"] = loc
    return stats.lognorm(sigma, loc=loc, scale=scale), params, 2 + free_loc


def _fit_gamma(t, free_loc):
    kwargs = {} if free_loc else {"floc": 0}
    alpha, loc, theta = stats.gamma.fit(t, **kwargs)
    params = {"α (bentuk)": alpha, "θ (skala)": theta}
    if free_loc:
        params["γ (lokasi)"] = loc
    return stats.gamma(alpha, loc=loc, scale=theta), params, 2 + free_loc


_FITTERS = {
    "Exponential": _fit_exponential,
    "Weibull": _fit_weibull,
    "Normal": _fit_normal,
    "Lognormal": _fit_lognormal,
    "Gamma": _fit_gamma,
}


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def fit_distribution(name: str, t: np.ndarray, free_loc: bool = False) -> FitResult:
    """Fit a single distribution by MLE and compute goodness-of-fit statistics.

    Never raises: any problem is reported through ``FitResult.error``.
    """
    t = np.asarray(t, dtype=float)
    if name in POSITIVE_ONLY and np.any(t <= 0):
        return FitResult(name, error="Membutuhkan t > 0, tetapi data memuat nilai ≤ 0.")
    try:
        with np.errstate(all="ignore"):
            dist, params, k = _FITTERS[name](t, free_loc)
            loglik = float(np.sum(dist.logpdf(t)))
        if not np.isfinite(loglik) or not all(np.isfinite(v) for v in params.values()):
            return FitResult(name, error="Estimasi parameter tidak konvergen.")
        n = t.size
        ks = stats.kstest(t, dist.cdf)
        return FitResult(
            name=name,
            dist=dist,
            params={key: float(v) for key, v in params.items()},
            n_params=k,
            loglik=loglik,
            aic=2 * k - 2 * loglik,
            bic=k * np.log(n) - 2 * loglik,
            ks_stat=float(ks.statistic),
            ks_pvalue=float(ks.pvalue),
        )
    except Exception as exc:  # scipy optimisers can fail on degenerate data
        return FitResult(name, error=f"Gagal melakukan fitting: {exc}")


def fit_all(t: np.ndarray, free_loc: bool = False) -> list[FitResult]:
    """Fit the five candidate distributions."""
    return [fit_distribution(name, t, free_loc) for name in DISTRIBUTIONS]


def rank_results(results: list[FitResult], criterion: str = "AIC") -> list[FitResult]:
    """Successful fits sorted from best to worst (lower criterion value = better)."""
    attr = CRITERIA[criterion]
    return sorted((r for r in results if r.ok), key=lambda r: getattr(r, attr))


def evaluate(dist, t) -> dict[str, np.ndarray]:
    """Reliability metrics at time(s) ``t`` for a frozen distribution.

    F(t): unreliability (CDF), R(t) = 1 - F(t), f(t): PDF, h(t) = f(t)/R(t).
    """
    t = np.asarray(t, dtype=float)
    with np.errstate(all="ignore"):
        F = dist.cdf(t)
        R = dist.sf(t)  # numerically safer than 1 - cdf in the far tail
        f = dist.pdf(t)
        h = np.where(R > 0, f / R, np.nan)
    return {"F": F, "R": R, "f": f, "h": h}


def empirical_cdf(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted failure times and Benard's median-rank estimate of F(t)."""
    x = np.sort(np.asarray(t, dtype=float))
    n = x.size
    ranks = np.arange(1, n + 1)
    return x, (ranks - 0.3) / (n + 0.4)
