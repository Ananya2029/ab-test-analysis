"""Statistics for analysing an A/B test on binary (conversion/retention) and continuous metrics."""
from dataclasses import dataclass

import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


# ---------------------------------------------------------------- sanity checks
def srm_test(n_a: int, n_b: int, expected_share_a: float = 0.5):
    """Sample ratio mismatch: chi-square goodness-of-fit of the observed split vs the planned one.
    A tiny p-value means the randomisation or logging may be broken and results can't be trusted."""
    total = n_a + n_b
    expected = [total * expected_share_a, total * (1 - expected_share_a)]
    chi2, p = stats.chisquare([n_a, n_b], f_exp=expected)
    return float(chi2), float(p)


# ---------------------------------------------------------------- binary metrics
@dataclass
class PropTest:
    rate_a: float
    rate_b: float
    diff: float          # rate_b - rate_a (absolute, in proportion units)
    ci_low: float
    ci_high: float
    rel_lift: float      # diff / rate_a
    z: float
    p_value: float


def two_proportion_test(x_a: int, n_a: int, x_b: int, n_b: int, alpha: float = 0.05) -> PropTest:
    """Two-sided z-test (pooled SE for the test, unpooled SE for the confidence interval)."""
    pa, pb = x_a / n_a, x_b / n_b
    pooled = (x_a + x_b) / (n_a + n_b)
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (pb - pa) / se_pooled
    p = 2 * stats.norm.sf(abs(z))
    se = np.sqrt(pa * (1 - pa) / n_a + pb * (1 - pb) / n_b)
    zc = stats.norm.ppf(1 - alpha / 2)
    d = pb - pa
    return PropTest(pa, pb, d, d - zc * se, d + zc * se, d / pa, float(z), float(p))


def bootstrap_diff(a: np.ndarray, b: np.ndarray, stat=np.mean, n_boot: int = 10_000,
                   alpha: float = 0.05, seed: int = 42):
    """Percentile bootstrap CI for stat(b) - stat(a). Returns (estimate, low, high, samples)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        samples[i] = stat(rng.choice(b, b.size)) - stat(rng.choice(a, a.size))
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(stat(b) - stat(a)), float(low), float(high), samples


def bayesian_ab(x_a: int, n_a: int, x_b: int, n_b: int, n_samples: int = 200_000, seed: int = 42):
    """Beta(1,1) priors -> Beta posteriors. Returns P(B > A), the 95% credible interval of B - A,
    and the expected loss (in proportion units) of choosing each variant."""
    rng = np.random.default_rng(seed)
    a = rng.beta(1 + x_a, 1 + n_a - x_a, n_samples)
    b = rng.beta(1 + x_b, 1 + n_b - x_b, n_samples)
    diff = b - a
    return {
        "p_b_better": float((diff > 0).mean()),
        "ci_low": float(np.quantile(diff, 0.025)),
        "ci_high": float(np.quantile(diff, 0.975)),
        "expected_loss_choose_a": float(np.maximum(diff, 0).mean()),
        "expected_loss_choose_b": float(np.maximum(-diff, 0).mean()),
        "posterior_diff": diff,
    }


# ---------------------------------------------------------------- power & design
def minimum_detectable_effect(p_base: float, n_per_group: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest absolute lift the test could reliably detect (solved on Cohen's h)."""
    h = NormalIndPower().solve_power(nobs1=n_per_group, alpha=alpha, power=power, ratio=1.0)
    # invert h = 2*asin(sqrt(p2)) - 2*asin(sqrt(p1)) for p2 > p1
    p2 = np.sin(np.arcsin(np.sqrt(p_base)) + h / 2) ** 2
    return float(p2 - p_base)


def required_sample_size(p_base: float, abs_lift: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Users per group needed to detect an absolute lift with the given power."""
    h = abs(proportion_effectsize(p_base + abs_lift, p_base))
    return int(np.ceil(NormalIndPower().solve_power(effect_size=h, alpha=alpha, power=power, ratio=1.0)))


# ---------------------------------------------------------------- the peeking problem
def peeking_false_positive_rate(p: float = 0.19, n_per_group: int = 20_000, n_looks: int = 20,
                                n_sims: int = 2_000, alpha: float = 0.05, seed: int = 42):
    """Simulate A/A tests (no real difference). Compare the false-positive rate of
    (a) testing once at the end vs (b) checking after every batch and stopping at the first p < alpha."""
    rng = np.random.default_rng(seed)
    looks = np.linspace(n_per_group / n_looks, n_per_group, n_looks).astype(int)
    fixed_hits = peek_hits = 0
    for _ in range(n_sims):
        a = rng.random(n_per_group) < p
        b = rng.random(n_per_group) < p
        ca, cb = np.cumsum(a), np.cumsum(b)
        significant = [two_proportion_test(ca[n - 1], n, cb[n - 1], n).p_value < alpha for n in looks]
        peek_hits += any(significant)
        fixed_hits += significant[-1]
    return fixed_hits / n_sims, peek_hits / n_sims
