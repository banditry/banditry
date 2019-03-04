import numpy as np
from scipy import stats
from scipy import special as sps
from scipy import optimize
from pypolyagamma import PyPolyaGamma


class NotFitted(Exception):
    """Raise when a model has not been fit and a method
    is being called that depends on it having been fit.
    """
    pass


class Seedable:
    """Inherit from this class to get methods useful for objects
    using random seeds.
    """
    def __init__(self, seed=42):
        self._initial_seed = seed
        self.rng = np.random.RandomState(self._initial_seed)

    def seed(self, seed):
        self.rng.seed(seed)
        return self

    def reset(self):
        self.seed(self._initial_seed)
        return self


class PGSeedable(Seedable):
    """Alternative to Seedable for classes that also use a PyPolyaGamma
    RNG object to generate Polya-Gamma random variates.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pg_rng = PyPolyaGamma(seed=self.rng.randint(0, 2 ** 32))

    def seed(self, seed):
        super().seed(seed)
        self.pg_rng = PyPolyaGamma(seed=self.rng.randint(0, 2 ** 32))
        return self

    # Use custom pickling to handle non-serializable PG RNG
    # WARNING: pickle + unpickle will reset seed
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['pg_rng']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.reset()


class HasFittableParams:
    """Provides some useful helper methods and properties."""

    @property
    def param_names(self):
        return [name for name in self.__dict__ if name.endswith('_')]

    def iter_params(self):
        return ((name, value)
                for name, value in self.__dict__.items()
                if name.endswith('_'))

    def reset_params(self):
        for name in self.param_names:
            setattr(self, name, None)

    def raise_if_not_fitted(self):
        empty_params = [name for name, value in self.iter_params() if value is None]
        if empty_params:
            raise NotFitted(f"some parameters are None: {empty_params}")


class BaseModel(Seedable, HasFittableParams):

    def reset(self):
        Seedable.reset(self)
        HasFittableParams.reset_params(self)
        return self


class PGBaseModel(PGSeedable, HasFittableParams):

    def reset(self):
        PGSeedable.reset(self)
        HasFittableParams.reset_params(self)
        return self


def draw_omegas(design_matrix, theta, pg_rng):
    num_rows = design_matrix.shape[0]
    omegas = np.ndarray(num_rows)
    logits = design_matrix.dot(theta)
    for i, logit_i in enumerate(logits):
        omegas[i] = pg_rng.pgdraw(1, logit_i)

    return omegas


class MCMCLogisticRegression(PGBaseModel):

    def __init__(self, num_samples=100, num_burnin=0, **kwargs):
        super().__init__(**kwargs)

        # Set other properties that control fitting
        self.num_samples = num_samples
        self.num_burnin = num_burnin

    def iter_params(self):
        return ((name, value if value is None else value[self.num_burnin:])
                for name, value in self.__dict__.items()
                if name.endswith('_'))


class LogisticRegression(MCMCLogisticRegression):
    """Bayesian logistic regression model, fitted with PG-augmented Gibbs."""

    def __init__(self, m0=None, P0=None, **kwargs):
        """
        Args:
            m0 (np.ndarray): prior mean
            P0 (np.ndarray): prior covariance matrix
        """
        super().__init__(**kwargs)

        # Hyperparameters
        self.m0 = m0
        self.P0 = P0

        # Set up empty parameters
        self.beta_hat_ = None

    def sample_from_prior(self):
        return self.rng.multivariate_normal(self.m0, self.P0)

    def fit(self, X, y):
        """Fit the model using Gibbs sampler.

        Args:
            X (np.ndarray): design matrix
            y (np.ndarray): responses (binary rewards)

        Returns:
            self: reference to fitted model object (this instance).
        """
        # Precompute some values that will be re-used in loops
        P0_inv = np.linalg.inv(self.P0)
        P0_inv_m0 = P0_inv.dot(self.m0)
        kappas = (y - 0.5).T
        num_predictors = X.shape[1]

        # Init memory for parameter traces
        beta_hat = np.ndarray((self.num_samples + 1, num_predictors))

        # Init trace from prior
        beta_hat[0] = self.sample_from_prior()

        for i in range(1, self.num_samples + 1):
            omegas = draw_omegas(X, beta_hat[i - 1], self.pg_rng)

            # TODO: speed this up by computing inverse via Cholesky decomposition
            V_omega = np.linalg.inv((X.T * omegas).dot(X) + P0_inv)
            m_omega = V_omega.dot(X.T.dot(kappas) + P0_inv_m0)

            beta_hat[i] = self.rng.multivariate_normal(m_omega, V_omega)

        # Set fitted parameters on instance
        self.beta_hat_ = beta_hat[1:]  # discard initial sample from prior
        return self

    def transform(self, X, num_burnin=None):
        self.raise_if_not_fitted()

        # Optionally override default burnin.
        num_burnin = self.num_burnin if num_burnin is None else num_burnin
        beta_trace = self.beta_hat_[num_burnin:]

        # Compute logits and then transform to rates
        logits = X.dot(beta_trace.T)
        return sps.expit(logits)

    def choose_arm(self, context):
        if self.beta_hat_ is None:
            beta_hat = self.rng.multivariate_normal(self.m0, self.P0)
        else:
            beta_hat = self.beta_hat_[-1]

        # Compute logits and then transform to rates
        logits = context.dot(beta_hat)
        rates = sps.expit(logits)

        # Choose best arm for this "plausible model."
        return np.argmax(rates)


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
        self.beta_hat_ = None

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

    def transform(self, X, num_burnin=None):
        self.raise_if_not_fitted()

        # Optionally override default burnin.
        num_burnin = self.num_burnin if num_burnin is None else num_burnin
        beta_trace = self.beta_hat_[num_burnin:]

        # Compute logits and then transform to rates
        logits = X.dot(beta_trace.T)
        return sps.expit(logits)

    def choose_arm(self, context):
        if self.beta_hat_ is None:
            _, _, beta_hat = self.sample_from_prior()
        else:
            beta_hat = self.beta_hat_[-1]

        # Compute logits and then transform to rates
        logits = context.dot(beta_hat)
        rates = sps.expit(logits)

        # Choose best arm for this "plausible model."
        return np.argmax(rates)


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
