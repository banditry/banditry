import logging

import numpy as np
import pandas as pd
from scipy import special as sps
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def simulate_gaussian_data(num_arms=5, num_predictors=3, num_time_steps=50,
                           *, prior_means=None, prior_cov=None,
                           context_prior_means=None, context_prior_cov=None,
                           seed=42):
    rng = np.random.RandomState(seed)

    # Generate true effects
    if prior_means is None:
        prior_means = np.zeros(num_predictors, dtype=np.float)

    if prior_cov is None:
        prior_cov = np.identity(num_predictors, dtype=np.float)

    true_effects = rng.multivariate_normal(prior_means, prior_cov)
    print(f'True effects: {np.round(true_effects, 4)}')

    # Generate design matrix
    if context_prior_means is None:
        context_prior_means = np.ones(num_predictors, dtype=np.float) * -3

    if context_prior_cov is None:
        context_prior_cov = np.identity(num_predictors, dtype=np.float)

    arm_contexts = rng.multivariate_normal(context_prior_means, context_prior_cov, size=num_arms)
    print(f'Context matrix size: {arm_contexts.shape}')

    # Generate multiple points for each arm, using round-robin routing.
    arm_per_time_step = rng.choice(num_arms, size=num_time_steps)
    print(f'Samples per arm: {np.bincount(arm_per_time_step)}')

    print(f'True rates: {sps.expit(arm_contexts.dot(true_effects))}')

    design_matrix = arm_contexts[arm_per_time_step]
    print(f'Design matrix size: {design_matrix.shape}')

    logits = design_matrix.dot(true_effects)
    rates = sps.expit(logits)

    ys = rng.binomial(n=1, p=rates)
    print(f'{sum(ys)} successes out of {len(ys)} samples')

    return true_effects, rates, design_matrix, ys


class ModelValidator:
    """Fit Bayesian regression models and validate their outputs."""

    def __init__(self, model, credible_bands=(90, 80, 50)):
        self.model = model
        self.credible_bands = credible_bands

    def validate(self, X, y, rates):
        self.model.fit(X, y)
        self.traceplot()
        self.recapture_plot(X, rates)

    def traceplot(self, fontsize=14):
        plottable_params = [(name, value)
                            for name, value in self.model.iter_params()
                            if hasattr(value, 'shape') and not name.startswith('_')]
        nrows = sum((1 if len(values.shape) == 2 else
                     values.shape[1] if len(values.shape) == 3
                     else 0)
                    for _, values in plottable_params)
        fig, axes = plt.subplots(nrows=nrows, squeeze=False, figsize=(10, nrows * 3.5))
        axes_iter = iter(axes.flat)

        def plot_next(param_df, param_name):
            ax = next(axes_iter)
            param_df.plot(ax=ax)

            # Pretty it up
            name_wo_underscore = param_name.rstrip('_')
            ax.set_title(name_wo_underscore, fontsize=fontsize + 2)
            ax.set_ylabel(f'Support({name_wo_underscore})', fontsize=fontsize)
            ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)

        for param_name, value in plottable_params:
            if len(value.shape) <= 2:
                num_samples, cardinality = value.shape
                individual_names = [f'{param_name}{i}' for i in range(cardinality)]
                param_df = pd.DataFrame(
                    value, columns=pd.Index(individual_names, 'Parameters'),
                    index=pd.Index(np.arange(num_samples), name='Posterior Samples'))
                plot_next(param_df, param_name)
            elif len(value.shape) <= 3:
                num_samples, cardinality = value.shape[0], value.shape[1:]
                for i in range(cardinality[0]):
                    row = value[:, i]  # select all samples
                    individual_names = [f'{param_name[:-1]}_{i}{j}' for j in range(cardinality[1])]
                    param_df = pd.DataFrame(
                        row, columns=pd.Index(individual_names, 'Parameters'),
                        index=pd.Index(np.arange(num_samples), name='Posterior Samples'))
                    plot_next(param_df, f'{param_name[:-1]}_{i}')
            else:
                # TODO: only supports 1D & 2D parameters
                logger.info(f'{param_name} with dimension {value.shape[1:]} not being plotted.'
                            f' Only parameters up to 2D are currently supported in traceplot')

        plt.tight_layout()
        return axes

    def recapture_plot(self, X, rates, fontsize=14):
        rates_trace = self.model.transform(X)
        expected_rates = np.mean(rates_trace, axis=-1)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title('Rate Recapture', fontsize=fontsize + 2)
        ci_labels = ','.join(f'{width}%' for width in self.credible_bands)
        ax.set_xlabel(f'Data (red) and Expected Value with {ci_labels} CIs (blue)',
                      fontsize=fontsize)
        ax.set_ylabel('Rate', fontsize=fontsize)

        # Plot actual rates along with expected value (estimate)
        ax.plot(rates, 's', color='red', alpha=0.8, label='actual')
        ax.plot(expected_rates, 's', color='blue', alpha=0.7, label='expected')
        ax.legend()

        # Plot vertical lines representing credible intervals of various widths.
        # First sort widths in descending order; we'll use this ordering to plot
        # with increasingly bold lines to show increasing credibility.
        widths = list(sorted(self.credible_bands, reverse=True))
        credible_bands = [(50 - width / 2, 50 + width / 2)
                          for width in widths]

        xpoints = np.arange(len(rates))
        for i, q in enumerate(credible_bands, start=1):
            lo, hi = np.percentile(rates_trace, q=q, axis=-1)
            plt.vlines(xpoints, lo, hi, color='blue', alpha=i * 0.2)

        return ax
