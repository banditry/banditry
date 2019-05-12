import numpy as np
import pymc3 as pm

from cmabeval.base import HasFittableParams
from cmabeval.experiment import ExperimentMetrics


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


class Comparator:
    """Compare two methods via statistical test on ExperimentMetrics."""

    def __init__(self, champion_metrics, challenger_metrics):
        """
        Args:
            champion_metrics (cmabeval.experiment.ExperimentMetrics)
            challenger_metrics (cmabeval.experiment.ExperimentMetrics)
        """
        self.champion = champion_metrics
        self.challenger = challenger_metrics

    @classmethod
    def load(cls, champion_dirpath, challenger_dirpath):
        champion = ExperimentMetrics.load(champion_dirpath)
        challenger = ExperimentMetrics.load(challenger_dirpath)
        return cls(champion, challenger)

    def compare_rewards(self):
        return self._compare('rewards')

    def compare_decision_time(self):
        return self._compare('time_per_decision')

    def _compare(self, metric_name):
        avg_diffs = self.compute_avg_diffs(metric_name)

        tester = BESTPairedT().fit(avg_diffs)
        tester.plot()
        return tester

    def compute_avg_diffs(self, metric_name):
        common_seeds = self._common_seeds()
        num_replications = len(common_seeds)
        avg_diff = np.ndarray((num_replications,))
        for i, seed in enumerate(common_seeds):
            m1 = self.champion[seed]
            m2 = self.challenger[seed]
            avg_diff[i] = (np.sum(getattr(m1, metric_name)) / m1.num_time_steps -
                           np.sum(getattr(m2, metric_name)) / m2.num_time_steps)
        return avg_diff

    def _common_seeds(self):
        return tuple(sorted(set(self.champion.replications.keys()) &
                            set(self.challenger.replications.keys())))


class MethodComparator:
    """Compare methods across experiments."""

    def __init__(self, comparators):
        self.comparators = comparators

    def compare_rewards(self):
        return self._compare('rewards')

    def compare_decision_time(self):
        return self._compare('time_per_decision')

    def _compare(self, metric_name):
        avg_diffs = self.compute_avg_diffs(metric_name)

        tester = BESTPairedT().fit(avg_diffs)
        tester.plot()
        return tester

    def compute_avg_diffs(self, metric_name):
        return np.concatenate([comp.compute_avg_diffs(metric_name)
                               for comp in self.comparators])
