import pymc3 as pm

from cmabeval.base import HasFittableParams


class BESTPairedT(HasFittableParams):
    """Kruschke's BEST (Bayesian Estimation Superseded t-Test) model."""

    def __init__(self, num_samples=1000):
        self.num_samples = num_samples

        self.trace_ = None

    def fit(self, diff):
        sigma_low = diff.std() * 1 / 100
        sigma_high = diff.std() * 100

        with pm.Model():
            group_mean = pm.Normal('group_mean', diff.mean(), diff.std() * 2)
            group_std = pm.Uniform('group_std', lower=sigma_low, upper=sigma_high)
            v = pm.Exponential('v_minus_one', 1 / 29.) + 1
            pm.StudentT('result', nu=v, mu=group_mean, sd=group_std, observed=diff)
            self.trace_ = pm.sample(self.num_samples)

        return self

    def plot(self, **kwargs):
        kwargs.setdefault('color', '#87ceeb')
        kwargs.setdefault('text_size', 14)
        return pm.plot_posterior(
            self.trace_, varnames=['group_mean'], **kwargs)