"""Checks the statistics against textbook values and known behaviour.

    python -m pytest -q
"""
import numpy as np
import pytest
from statsmodels.stats.proportion import proportions_ztest

from ab.stats import (bayesian_ab, bootstrap_diff, minimum_detectable_effect, peeking_false_positive_rate,
                      required_sample_size, srm_test, two_proportion_test)


def test_two_proportion_test_matches_statsmodels():
    r = two_proportion_test(200, 1000, 250, 1000)
    z, p = proportions_ztest([250, 200], [1000, 1000])
    assert r.z == pytest.approx(z) and r.p_value == pytest.approx(p)
    assert r.diff == pytest.approx(0.05) and r.ci_low < 0.05 < r.ci_high


def test_srm_detects_a_broken_split():
    assert srm_test(5000, 5000)[1] == pytest.approx(1.0)
    assert srm_test(5000, 5300)[1] < 0.01


def test_bootstrap_ci_matches_theory():
    # A single 95% interval misses the true value 5% of the time by design, so test its width instead:
    # it should match the analytical 2 x 1.96 x SE of a difference in means.
    rng = np.random.default_rng(0)
    a, b = rng.normal(10, 2, 3000), rng.normal(10.5, 2, 3000)
    est, lo, hi, _ = bootstrap_diff(a, b, n_boot=4000)
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    assert lo < est < hi
    assert (hi - lo) == pytest.approx(2 * 1.96 * se, rel=0.1)


def test_bayesian_probability_is_sensible():
    same = bayesian_ab(500, 1000, 500, 1000, n_samples=50_000)
    better = bayesian_ab(500, 1000, 560, 1000, n_samples=50_000)
    assert 0.45 < same["p_b_better"] < 0.55
    assert better["p_b_better"] > 0.99


def test_power_functions_are_consistent():
    n = required_sample_size(0.19, 0.01)
    mde = minimum_detectable_effect(0.19, n)
    assert mde == pytest.approx(0.01, abs=5e-4)
    # textbook check: 50% vs 55% at alpha 0.05 / 80% power needs about 1,565 users per group
    assert 1540 <= required_sample_size(0.5, 0.05) <= 1590


def test_peeking_inflates_false_positives():
    fixed, peek = peeking_false_positive_rate(n_per_group=2000, n_looks=10, n_sims=400)
    assert fixed < 0.09
    assert peek > fixed + 0.05
