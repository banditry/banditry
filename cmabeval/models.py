import logging
import itertools

import numpy as np
import pandas as pd
import scipy as sp
from scipy import optimize, stats
from scipy import special as sps

from cmabeval.base import Seedable, BaseModel, PGBaseModel
from cmabeval.exceptions import NotFitted, InsufficientData
from cmabeval.transformers import Preprocessor

logger = logging.getLogger(__name__)


def draw_omegas(design_matrix, theta, pg_rng):
    num_rows = design_matrix.shape[0]
    omegas = np.ndarray(num_rows)
    logits = design_matrix.dot(theta)
    for i, logit_i in enumerate(logits):
        omegas[i] = pg_rng.pgdraw(1, logit_i)

    return omegas


class Agent:
    def choose_arm(self, context):
        raise NotImplementedError


class EqualAllocationAgent(Agent, Seedable):

    def __init__(self, num_arms, **kwargs):
        Seedable.__init__(self, **kwargs)
        self.num_arms = num_arms

    def choose_arm(self, context):
        return self.rng.random_integers(0, self.num_arms)


class MCMCLogisticRegression(PGBaseModel):

    def __init__(self, num_samples=100, num_burnin=0, **kwargs):
        super().__init__(**kwargs)

        # Set other properties that control fitting
        self.num_samples = num_samples
        self.num_burnin = num_burnin

        self.beta_hat_ = None

    def iter_params(self):
        return ((name, (value if value is None or not hasattr(value, '__getitem__')
                        else value[self.num_burnin:]))
                for name, value in self.__dict__.items()
                if name.endswith('_'))

    def sample_from_prior(self, **kwargs):
        raise NotImplementedError

    def last_beta_sample(self, **kwargs):
        if self.beta_hat_ is None:
            params = self.sample_from_prior(**kwargs)
            if isinstance(params, tuple):
                return params[-1]  # convention is that beta_hat is last param drawn
            else:
                return params  # assumes beta is only param drawn
        else:
            return self.beta_hat_[-1]

    def transform(self, X, num_burnin=None):
        self.raise_if_not_fitted()

        # Optionally override default burnin.
        num_burnin = self.num_burnin if num_burnin is None else num_burnin
        beta_trace = self.beta_hat_[num_burnin:]

        # Compute logits and then transform to rates
        logits = X.dot(beta_trace.T)
        return sps.expit(logits)

    def choose_arm(self, context):
        beta_hat = self.last_beta_sample()

        # Compute logits and then transform to rates
        logits = context.dot(beta_hat)
        rates = sps.expit(logits)

        # Choose best arm for this "plausible model."
        # Break ties randomly.
        return self.rng.choice(np.flatnonzero(rates == rates.max()))


class LogisticRegression(MCMCLogisticRegression):
    """Bayesian logistic regression model, fitted with PG-augmented Gibbs."""

    def __init__(self, m0=None, P0=None, interactions=False, **kwargs):
        """
        Args:
            m0 (np.ndarray): prior mean
            P0 (np.ndarray): prior covariance matrix
        """
        super().__init__(**kwargs)

        # Hyperparameters
        self.m0 = m0
        self.P0 = P0
        self.interactions = interactions

        self.preprocessor_ = None

    def sample_from_prior(self):
        return self.rng.multivariate_normal(self.m0, self.P0)

    def fit(self, df, y):
        """Fit the model using Gibbs sampler.

        Args:
            df (pd.DataFrame): design matrix
            y (np.ndarray): responses (binary rewards)

        Returns:
            self: reference to fitted model object (this instance).
        """
        preprocessor = Preprocessor(self.interactions)
        X = preprocessor.fit_transform(df)

        # Precompute some values that will be re-used in loops
        P0_inv = np.linalg.inv(self.P0)
        P0_inv_m0 = P0_inv.dot(self.m0)
        kappas = (y - 0.5).T
        XTkappa = X.T.dot(kappas)
        y_omega = XTkappa + P0_inv_m0
        num_predictors = X.shape[1]

        # Init memory for parameter traces
        beta_hat = np.ndarray((self.num_samples + 1, num_predictors))

        # Init trace from prior
        beta_hat[0] = self.sample_from_prior()

        gammas = self.rng.normal(0, 1, size=(self.num_samples, num_predictors))
        for s in range(1, self.num_samples + 1):
            omegas = draw_omegas(X, beta_hat[s - 1], self.pg_rng)
            V_omega_inv = (X.T * omegas).dot(X) + P0_inv

            try:
                L = sp.linalg.cholesky(V_omega_inv, lower=True)
            except sp.linalg.LinAlgError as err:  # V_omega_inv not positive semi-definite
                raise InsufficientData(err)

            # Solve system of equations to sample beta from multivariate normal
            eta = sp.linalg.solve_triangular(L, y_omega, lower=True)
            beta_hat[s] = sp.linalg.solve_triangular(
                L, eta + gammas[s - 1], lower=True, trans='T')

        # Set fitted parameters on instance
        self.beta_hat_ = beta_hat[1:]  # discard initial sample from prior
        self.preprocessor_ = preprocessor

        return self

    def choose_arm(self, context):
        self.raise_if_not_fitted()
        preprocessed_context = self.preprocessor_.transform(context)
        beta_hat = self.last_beta_sample()

        # Compute logits and then transform to rates
        logits = preprocessed_context.dot(beta_hat)
        rates = sps.expit(logits)

        # Choose best arm for this "plausible model." Break ties randomly.
        return self.rng.choice(np.flatnonzero(rates == rates.max()))


class LogisticRegressionNIW(MCMCLogisticRegression):
    """Bayesian logistic regression model, fitted with PG-augmented Gibbs."""

    def __init__(self, eta0=0.01, mu0=None, nu0=None, Lambda0=None, **kwargs):
        """
        Args:
            eta0 (int): prior mean "strength".
            mu0 (np.ndarray[ndim=1]): prior mean vector of coefficients.
            nu0 (int): prior degrees of freedom; controls strength of IW prior on Sigma.
            Lambda0 (np.ndarray[ndim=2]): prior scatter matrix for IW prior on Sigma.
        """
        super().__init__(**kwargs)

        # Hyperparameters
        self.eta0 = eta0
        self.mu0 = mu0
        self.nu0 = nu0
        self.Lambda0 = Lambda0

        # Set up empty parameters
        self.Sigma_hat_ = None
        self.mu_hat_ = None

    def sample_from_prior(self, Psi0=None):
        if Psi0 is None:
            Psi0 = np.linalg.inv(self.Lambda0)  # scale matrix of Inverse Wishart

        Sigma = stats.invwishart.rvs(self.nu0, Psi0, random_state=self.rng)
        mu = self.rng.multivariate_normal(self.mu0, Sigma / self.eta0)
        beta = self.rng.multivariate_normal(mu, Sigma)

        return Sigma, mu, beta

    def last_sample(self, **kwargs):
        try:
            self.raise_if_not_fitted()
            return self.Sigma_hat_[-1], self.mu_hat_[-1], self.beta_hat_[-1]
        except NotFitted:
            return self.sample_from_prior(**kwargs)

    def fit(self, X, y):
        """Fit the model using Gibbs sampler.

        Args:
            X (np.ndarray): design matrix
            y (np.ndarray): responses (binary rewards)

        Returns:
            self: reference to fitted model object (this instance).

        WARNING: calling fit multiple times in a row may produce different results
            if the hyperparameters haven't been set. The first time, they'll be set
            from the data, and the next time, those values will be re-used.
        """
        num_predictors = X.shape[1]

        # Set hyperparameters from data if no values have been set.
        if self.mu0 is None:
            self.mu0 = np.zeros(num_predictors, dtype=np.float)

        if self.nu0 is None:
            self.nu0 = num_predictors + 2

        if self.Lambda0 is None:
            self.Lambda0 = np.identity(num_predictors, dtype=np.float)

        # Precompute some values that will be re-used in loops
        kappas = (y - 0.5).T
        XTkappa = X.T.dot(kappas)

        Psi0 = np.linalg.inv(self.Lambda0)  # scale matrix of Inverse Wishart
        eta_t = self.eta0 + 1
        nu_t = self.nu0 + 1

        # Init memory for parameter traces
        Sigma_hat = np.ndarray((self.num_samples + 1, num_predictors, num_predictors))
        mu_hat = np.ndarray((self.num_samples + 1, num_predictors))
        beta_hat = np.ndarray((self.num_samples + 1, num_predictors))

        # Init traces from priors
        Sigma_hat[0], mu_hat[0], beta_hat[0] = self.last_sample(Psi0=Psi0)

        # Assign the instance parameters to be views of the traces that
        # exclude the initial samples from the prior
        self.Sigma_hat_ = Sigma_hat[1:]
        self.mu_hat_ = mu_hat[1:]
        self.beta_hat_ = beta_hat[1:]

        for s in range(1, self.num_samples + 1):
            omegas = draw_omegas(X, beta_hat[s - 1], self.pg_rng)

            # Draw betas
            Lambda = np.linalg.inv(Sigma_hat[s - 1])
            # TODO: speed this up by computing inverse via Cholesky decomposition
            V_omega = np.linalg.inv((X.T * omegas).dot(X) + Lambda)
            m_omega = V_omega.dot(XTkappa + Lambda.dot(mu_hat[s - 1]))
            beta_hat[s] = self.rng.multivariate_normal(m_omega, V_omega)

            # Draw mu and Sigma
            beta_mean = beta_hat[s].mean()
            mu_t = (self.eta0 * self.mu0 + beta_mean) / eta_t

            beta_resids = (beta_hat[s] - beta_mean)[:, None]  # d x 1
            S = beta_resids.dot(beta_resids.T)
            prior_beta_resids = (beta_hat[s] - self.mu0)[:, None]  # d x 1
            mean_virtual_scatter = (self.eta0 / eta_t) * prior_beta_resids.dot(prior_beta_resids.T)
            Lambda_t = self.Lambda0 + S + mean_virtual_scatter

            # TODO: speed things up by sampling Lambda_hat[s] from Wishart, then inverting that
            # to get Sigma -- then Lambda_hat[s] can be re-used for the beta draws.
            Sigma_hat[s] = stats.invwishart.rvs(
                nu_t, np.linalg.inv(Lambda_t), random_state=self.rng)
            mu_hat[s] = self.rng.multivariate_normal(mu_t, Sigma_hat[s] / eta_t)

        return self


"""
Perhaps break the design up into a distribution object like
scipy's frozen distributions and a trace object that contains the actual
samples for the trace?

Then you can have one distribution representing the prior values based
on the hyperparameters and another based on the posterior values created
for each sample. One thought is to have these be mutable, in order to re-use
this object across samples. However, this doesn't really work out because
after we make the first update, we'd lose the initial hyperparameters. So
another option is to just have a `draw_from_posterior` method that does the
updates and then takes a draw based on those updated values, then discards
the updated values.
"""
class IGGDist:

    __slots__ = ['mapping', 'a0', 'b0', 'c0', 'd0']

    def __init__(self, mapping, a0=0.01, b0=0, c0=0.01, d0=0.1):
        """
        Args:
            mapping (np.ndarray): index corresponds to coefficient index,
                and value corresponds to coefficient group membership.
            a0 (float): > 0, shape parameter for coefficient IG prior.
            b0 (float): >= 0, additive factor on the IG rate prior. This is not the
                same as the rate itself and is broken out to facilitate re-use of
                this class for both the prior and the posterior distributions. In
                the posterior, this will capture the observed sum of squared
                deviations. In the prior, it should always be 0.
            c0 (float): > 0, shape parameter for IG rate prior (Gamma).
            d0 (float): > 0, rate parameter for IG rate prior (Gamma).
        """
        self.mapping = mapping
        self.a0 = a0
        self.b0 = b0
        self.c0 = c0
        self.d0 = d0

    @property
    def num_groups(self):
        return len(np.unique(self.mapping))

    @property
    def num_coefficients(self):
        return len(self.mapping)

    def group_masks(self):
        return [(self.mapping == i) for i in range(self.num_groups)]

    def rvs(self, size=None, rng=None):
        """
        Args:
            size (int): number of samples to draw.
            rng (np.random.RandomState): RNG to use for drawing samples.

        Returns:
            tuple[np.ndarray]: samples for each parameter in the distribution.
        """
        if rng is None:
            rng = np.random.RandomState()

        # Pooled prior across variance scales
        b_size = ((self.num_groups,) if size is None else
                  (self.num_groups, size))
        b = stats.gamma.rvs(
            self.c0, scale=(1 / self.d0), size=b_size, random_state=rng)
        scale = b + self.b0

        sigma_sq = stats.invgamma.rvs(self.a0, scale=scale, random_state=rng)

        return b, sigma_sq

    def update(self, sigma_sq, beta):
        c_t, d_t = self._update_b(sigma_sq)
        a_t, b_t = self._update_sigma_sq(beta)
        return self.__class__(self.mapping, a_t, b_t, c_t, d_t)

    def _update_b(self, sigma_sq):
        c_t = self.c0 + self.num_groups * self.a0

        # d_t is the rate parameter, which is the inverse of the scale
        sum_sigma_sq = np.sum(sigma_sq)
        d_t = self.d0 + (1 / sum_sigma_sq if sum_sigma_sq > 0 else 0)
        return c_t, d_t

    def _update_sigma_sq(self, beta):
        group_masks = self.group_masks()
        a_t = self.a0 + 0.5 * np.array([np.sum(mask)
                                        for mask in group_masks])
        beta_sum_squares = []
        with np.errstate(all='raise'):
            for mask in group_masks:
                try:
                    beta_sum_squares.append(np.sum(np.square(beta[mask])))
                except FloatingPointError:
                    beta_sum_squares.append(np.finfo(np.float).max)

        b_t = 0.5 * np.array(beta_sum_squares)
        return a_t, b_t


"""
There's something fundamentally whacked out about the current design of these
classes. In general, it's impossible to draw samples from the prior and set up
memory for the parameter traces before actually seeing what the data is going
to look like. These procedures make more sense to have on objects that are
created for each fit using the data passed during that fit.
"""
class LogisticRegressionIGG(MCMCLogisticRegression):
    """Bayesian logistic regression model with NIG prior, fitted with PG-augmented Gibbs."""

    def __init__(self, mapping_type='pooled', a0=0.01, c0=0.01, d0=0.01, interactions=False,
                 **kwargs):
        """
        Args:
            mapping_type (str): specify which coefficients will be mapped together.
                'pooled': all grouped together
                'unpooled': each coefficient has unique scale parameter
                'by_order': effects at each order of interaction are grouped
            a0 (float): 1/2 prior sample size for effect variance.
            c0 (float): 1/2 prior sample size for effect variance scale.
            d0 (float): 1/2 prior sum of variance for effect variance scale.
            interactions (bool): pass True to include 2nd-order interaction terms
        """
        super().__init__(**kwargs)

        # Hyperparameters
        self.mapping_type = mapping_type
        self.a0 = a0
        self.c0 = c0
        self.d0 = d0
        self.interactions = interactions

        # Set up empty parameters
        self.preprocessor_ = None
        self._mapping_ = None
        self.sigma_sq_hat_ = None
        self.b_hat_ = None

    def get_prior(self, mapping=None):
        if self._mapping_ is None:
            if mapping is None:
                raise NotFitted("if model is not yet fit, must pass mapping")
        else:
            mapping = self._mapping_

        return IGGDist(mapping, self.a0, self.c0, self.d0)

    def sample_from_prior(self, mapping=None):
        prior = self.get_prior(mapping)
        b, sigma_sq = prior.rvs(rng=self.rng)

        # Broadcast group variances to number of coefficients
        # print(sigma_sq, b_size, rate)
        if prior.num_groups == 1:
            variances = np.ones(prior.num_coefficients) * sigma_sq
        else:
            variances = sigma_sq[prior.mapping]

        # Use shape from variances in case we're drawing multiple samples
        # print(variances.shape)
        prior_means = np.zeros(variances.shape)

        # Draw coefficients; this incorporates shared information across groups
        # via common broadcasted variance terms.
        beta = self.rng.normal(prior_means, variances)

        return b, sigma_sq, beta

    def last_sample(self, mapping=None):
        try:
            self.raise_if_not_fitted()
            return self.b_hat_[-1], self.sigma_sq_hat_[-1], self.beta_hat_[-1]
        except NotFitted:
            return self.sample_from_prior(mapping)

    def construct_coefficient_mapping(self, dmat, real_colnames):
        num_predictors = dmat.shape[1]
        if self.mapping_type == 'pooled':
            return np.zeros(num_predictors, dtype=np.int)
        elif self.mapping_type == 'unpooled':
            return np.arange(num_predictors, dtype=np.int)
        elif self.mapping_type == 'by_type':
            # prior for each data type
            named_terms = {name: term for name, term in
                           zip(dmat.design_info.term_names, dmat.design_info.terms)}
            column_indices = np.arange(dmat.shape[1])
            real_indices = np.array(sorted(itertools.chain.from_iterable(
                column_indices[dmat.design_info.term_slices[named_terms[name]]]
                for name in real_colnames)))
            mapping = np.zeros(dmat.shape[1], dtype=np.int)
            mapping[real_indices] = 1
            return mapping
        elif self.mapping_type == 'by_order':
            raise ValueError("mapping_type='by_order' not currently supported")
        else:
            raise ValueError(f"unrecognized mapping_type {self.mapping_type}")

    def fit(self, df, y):
        """Fit the model using Gibbs sampler.

        Args:
            df (pd.DataFrame): design matrix
            y (np.ndarray): responses (binary rewards)

        Returns:
            self: reference to fitted model object (this instance).

        WARNING: calling fit multiple times in a row may produce different results
            if the hyperparameters haven't been set. The first time, they'll be set
            from the data, and the next time, those values will be re-used.
        """
        preprocessor = Preprocessor(self.interactions)
        X = dmat = preprocessor.fit_transform(df)

        real_colnames = list(sorted(df.select_dtypes(['int', 'float']).columns))
        self._mapping_ = self.construct_coefficient_mapping(dmat, real_colnames)
        prior = self.get_prior()

        # Precompute some values that will be re-used in loops
        kappas = (y - 0.5).T
        y_omega = X.T.dot(kappas)

        # Init memory for parameter traces
        num_predictors = X.shape[1]
        b_hat = np.ndarray((self.num_samples + 1, prior.num_groups))
        sigma_sq_hat = np.ndarray((self.num_samples + 1, prior.num_groups))
        beta_hat = np.ndarray((self.num_samples + 1, num_predictors))

        # Init traces from priors
        b_hat[0], sigma_sq_hat[0], beta_hat[0] = self.last_sample()

        # Pre-draw random variables for multivariate normal sampling for efficiency
        gammas = self.rng.normal(0, 1, size=(self.num_samples, num_predictors))
        for s in range(1, self.num_samples + 1):
            # Draw beta
            omegas = draw_omegas(X, beta_hat[s - 1], self.pg_rng)

            V_omega_inv = (X.T * omegas).dot(X)  # augmented scatter matrix
            Lambda_diag = 1 / sigma_sq_hat[s - 1][self._mapping_]
            np.fill_diagonal(V_omega_inv, np.diag(V_omega_inv) + Lambda_diag)  # add in Lambda

            post_dist = None
            try:
                L = sp.linalg.cholesky(V_omega_inv, lower=True)
            except sp.linalg.LinAlgError:
                try:
                    if post_dist is None:
                        raise

                    logger.debug(f'V_omega_inv not positive semi-definite for sample {s}; '
                                 'attempting to resample last sample hyperparams to resolve')

                    # Re-use previous computation of augmented scatter matrix
                    # by first subtracting out contribution from sampled variance.
                    np.fill_diagonal(V_omega_inv, np.diag(V_omega_inv) - Lambda_diag)
                    b_hat[s - 1], sigma_sq_hat[s - 1] = post_dist.rvs(rng=self.rng)

                    # Now add in contribution from new sample
                    Lambda_diag = 1 / sigma_sq_hat[s - 1][self._mapping_]
                    np.fill_diagonal(V_omega_inv, np.diag(V_omega_inv) + Lambda_diag)

                    # Attempt Cholesky again
                    L = sp.linalg.cholesky(V_omega_inv, lower=True)
                except sp.linalg.LinAlgError:  # resample failed
                    logger.debug(f'resample failed for sample {s}; '
                                 'adding machine epsilon to diagonal to attempt to resolve')
                    try:
                        adjustment = np.eye(num_predictors) * np.finfo(np.float).eps
                        L = sp.linalg.cholesky(V_omega_inv + adjustment, lower=True)
                    except sp.linalg.LinAlgError as err:  # adjustment failed
                        raise InsufficientData(err)

            # Solve system of equations to sample beta from multivariate normal
            eta = sp.linalg.solve_triangular(L, y_omega, lower=True)
            beta_hat[s] = sp.linalg.solve_triangular(
                L, eta + gammas[s - 1], lower=True, trans='T')

            # Draw b and sigma_sq
            post_dist = prior.update(sigma_sq_hat[s - 1], beta_hat[s])
            b_hat[s], sigma_sq_hat[s] = post_dist.rvs(rng=self.rng)

        # Assign the instance parameters to be views of the traces that
        # exclude the initial samples from the prior
        self.sigma_sq_hat_ = sigma_sq_hat[1:]
        self.b_hat_ = b_hat[1:]
        self.beta_hat_ = beta_hat[1:]

        # Store transformer for use in transform
        self.preprocessor_ = preprocessor

        return self

    def choose_arm(self, context):
        self.raise_if_not_fitted()
        preprocessed_context = self.preprocessor_.transform(context)
        beta_hat = self.last_beta_sample()

        # Compute logits and then transform to rates
        logits = preprocessed_context.dot(beta_hat)
        rates = sps.expit(logits)

        # Choose best arm for this "plausible model." Break ties randomly.
        return self.rng.choice(np.flatnonzero(rates == rates.max()))


# This is the form from Bishop's PRML (p. 218)
def posterior_neg_log_likelihood(w, X, t, m, q):
    logits = X.dot(w)
    y = sps.expit(logits)
    # diff = w - m
    # P0 = np.diag(1 / q)
    # diff.dot(P0).dot(diff) -
    return (0.5 * q.dot((w - m) ** 2) -
            np.sum(t * np.log(y) + (1 - t) * np.log1p(-y)))


# TODO: make beta_dist a parameter
# TODO: add rvs method and move `num_samples` from constructor to that
# TODO: move over the ModelValidator into new module and update to use rvs method for plots.
# TODO: generalize `transform` method by having it first call a `get_beta_trace` method and adding **kwargs to transform and passing them through
# TODO: update `choose_arm` to use new `rvs` method to get the beta sample; unify interface with LogisticRegression model
class LaplaceLogisticRegression(BaseModel):
    """Bayesian logistic regression model, fitted with Laplace approximation."""

    def __init__(self, m0=None, q0=None, num_samples=100, **kwargs):
        """
        Args:
            m0 (np.ndarray): prior mean
            q0 (np.ndarray): prior precision matrix diagonal
        """
        super().__init__(**kwargs)

        # Hyperparameters
        self.m0 = m0
        self.q0 = q0

        # Set other properties that control fitting
        self.num_samples = num_samples

        # Set up empty parameters
        self.beta_dist = stats.multivariate_normal(self.m0, np.diag(1 / self.q0))
        self.beta_hat_ = None

    def reset(self):
        super().reset()
        self.beta_dist = stats.multivariate_normal(self.m0, np.diag(1 / self.q0))
        self.beta_hat_ = None
        return self

    def fit(self, X, y):
        """Fit the model using Laplacian approximation.

        Args:
            X (np.ndarray): design matrix
            y (np.ndarray): responses (binary rewards)

        Returns:
            self: reference to fitted model object (this instance).
        """
        # First we need to find the mode of the posterior distribution.
        num_predictors = X.shape[1]
        optimization_result = optimize.minimize(
            posterior_neg_log_likelihood,
            x0=np.random.normal(0, 0.001, size=num_predictors),
            args=(X, y, self.m0, self.q0))
        mean_map_estimate = optimization_result.x

        # Next we "fit" a Gaussian centered at this posterior mode.
        # The computations below compute the covariance matrix by
        # taking the inverse of the matrix of second derivatives of
        # the negative log likelihood (see Bishop 4.5 for more details).
        m = mean_map_estimate  # retain notation from Chapelle paper
        p = sps.expit(X.dot(m))[:, None]
        q = self.q0 + np.sum(X ** 2 * p * (1 - p), axis=0)
        cov = np.diag(1 / q)  # q is precision

        # Set fitted parameters on instance
        self.beta_dist = stats.multivariate_normal(m, cov)
        self.beta_hat_ = self.beta_dist.rvs(self.num_samples, random_state=self.rng)
        return self

    def transform(self, X):
        self.raise_if_not_fitted()

        # Compute logits and then transform to rates
        logits = X.dot(self.beta_hat_.T)
        return sps.expit(logits)

    def choose_arm(self, context):
        beta_sample = self.beta_dist.rvs(random_state=self.rng)
        logits = context.dot(beta_sample)
        rates = sps.expit(logits)
        return np.argmax(rates)
