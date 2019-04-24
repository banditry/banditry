import os
import json
import logging
import datetime
from concurrent import futures

import numpy as np
import pandas as pd
from scipy import special as sps
import matplotlib.pyplot as plt

from cmabeval.base import Seedable
from cmabeval.exceptions import NotFitted, InsufficientData
from cmabeval import serialize, versioning

logger = logging.getLogger(__name__)
ISO_8601_FMT = '%Y-%m-%dT%H:%M:%S.%f'


def plot_cum_regret(rewards, optimal_rewards, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.pop('figsize', None))

    regret = optimal_rewards - rewards
    cum_regret = np.cumsum(regret, axis=-1)
    pd.DataFrame(cum_regret.T).plot(
        ax=ax,
        color=kwargs.get('color', 'red'),
        alpha=kwargs.get('alpha', 0.5))

    fontsize = kwargs.pop('fontsize', 14)
    ax.set_ylabel('Cumulative Regret', fontsize=fontsize)
    ax.set_xlabel('Trial Number', fontsize=fontsize)
    ax.get_legend().remove()
    ax.set_title(kwargs.get('title', ''), fontsize=fontsize + 2)

    return ax


class GaussianSimulationFactory(Seedable):
    """Simulate data according to contextual Gaussian distributions.

    A factory creates individual environments.
    This particular factory creates `GaussianSimulationEnvironment`s.
    """

    def __init__(self, num_arms=100, num_predictors=10, num_time_steps=1000,
                 *, prior_effect_means=None, prior_effect_cov=None,
                 prior_context_means=None, prior_context_cov=None, **kwargs):
        super().__init__(**kwargs)

        self.num_arms = num_arms
        self.num_predictors = num_predictors
        self.num_time_steps = num_time_steps

        # Set prior parameters for effects
        self.prior_effect_means = prior_effect_means
        if self.prior_effect_means is None:
            self.prior_effect_means = np.zeros(
                self.num_predictors, dtype=np.float)

        self.prior_effect_cov = prior_effect_cov
        if self.prior_effect_cov is None:
            self.prior_effect_cov = np.identity(
                self.num_predictors, dtype=np.float)

        # Set prior parameters for arm contexts
        self.prior_context_means = prior_context_means
        if self.prior_context_means is None:
            self.prior_context_means = np.ones(self.num_predictors, dtype=np.float) * -3

        self.prior_context_cov = prior_context_cov
        if self.prior_context_cov is None:
            self.prior_context_cov = np.identity(self.num_predictors, dtype=np.float)

    def __call__(self):
        # Generate true effects
        true_effects = self.rng.multivariate_normal(
            self.prior_effect_means, self.prior_effect_cov)
        logger.info(f'True effects: {np.round(true_effects, 4)}')

        # Generate design matrix
        arm_contexts = self.rng.multivariate_normal(
            self.prior_context_means, self.prior_context_cov, size=self.num_arms)
        logger.info(f'Context matrix size: {arm_contexts.shape}')

        return GaussianSimulationEnvironment(
            true_effects, arm_contexts, seed=self.rng.randint(0, 2**32))


class GaussianSimulationEnvironment(Seedable):
    """An environment with Gaussian-distributed rewards related to
    contextual covariates linearly through a logistic link function.

    To replicate an experiment with the same environment but different
    random seeds, simply change the random seed after the first experiment
    is complete. If running in parallel, create multiple of these objects
    with different random seeds but the same parameters otherwise.
    """

    def __init__(self, true_effects, arm_contexts, **kwargs):
        super().__init__(**kwargs)

        self.true_effects = true_effects
        self.arm_contexts = arm_contexts
        self.arm_rates = self._recompute_arm_rates()
        self.optimal_arm = np.argmax(self.arm_rates)
        self.optimal_rate = self.arm_rates[self.optimal_arm]

    def _recompute_arm_rates(self):
        logits = self.arm_contexts.dot(self.true_effects)
        return sps.expit(logits)

    @property
    def num_arms(self):
        return self.arm_contexts.shape[0]

    @property
    def num_predictors(self):
        return self.arm_contexts.shape[1]

    def __str__(self):
        return (f'{self.__class__.__name__}'
                f', num_predictors={self.num_predictors}'
                f', num_arms={self.num_arms}'
                f', max_arm_rate={np.round(np.max(self.arm_rates), 5)}'
                f', mean_arm_rate={np.round(np.mean(self.arm_rates), 5)}')

    def __repr__(self):
        return self.__str__()

    def random_arm(self):
        return self.rng.choice(self.num_arms)

    def choose_arm(self, i):
        self._validate_arm_index(i)

        # Generate data for optimal arm.
        y_optimal = self.rng.binomial(n=1, p=self.optimal_rate)

        # Generate data for selected arm.
        context = self.arm_contexts[i]
        if i == self.optimal_arm:
            y = y_optimal
        else:
            y = self.rng.binomial(n=1, p=self.arm_rates[i])

        return context, y, y_optimal

    def _validate_arm_index(self, i):
        if i < 0 or i >= self.num_arms:
            raise ValueError(
                f'arm a must satisfy: 0 < a < {self.num_arms}; got {i}')


class ReplicationMetrics:
    """Record metrics from a single replication of an experiment.

    These consist of:

    1.  design matrix rows observed,
    2.  actions taken,
    3.  associated rewards,
    4.  optimal action possible,
    5.  optimal reward possible, and
    6.  compute time consumed to take action.

    Replication metrics serialize as a DataFrame with two types of columns:

    1.  design matrix rows stored using their column names or placeholder
        column names with the format 'p{column_index}', and
    2.  metadata columns (2-6 above) stored using the naming convention
        _{metadata_element_name}_

    """
    _action_colname = '_action_'
    _optimal_action_colname = '_optimal_action_'
    _reward_colname = '_reward_'
    _optimal_reward_colname = '_optimal_reward_'
    _compute_time_colname = '_compute_time_'

    metadata_colnames = [_action_colname, _optimal_action_colname,
                         _reward_colname, _optimal_reward_colname,
                         _compute_time_colname]

    def __init__(self, seed, num_time_steps, num_predictors, predictor_colnames=None):
        self.seed = seed
        if predictor_colnames is None:
            self._predictor_colnames = [f'p{i}' for i in range(num_predictors)]
        else:
            self._predictor_colnames = list(predictor_colnames)
        self.design_matrix = pd.DataFrame(index=pd.Index(range(num_time_steps), name='time_step'),
                                          columns=self._predictor_colnames, dtype=np.float)

        self.actions = np.ndarray(num_time_steps, dtype=np.uint)
        self.optimal_actions = np.ndarray(num_time_steps, dtype=np.uint)
        self.rewards = np.ndarray(num_time_steps, dtype=np.float)
        self.optimal_rewards = np.ndarray(num_time_steps, dtype=np.float)
        self.time_per_decision = np.ndarray(num_time_steps, dtype=np.float)

        self.start = None
        self.end = None

    @property
    def num_time_steps(self):
        return self.design_matrix.shape[0]

    @property
    def num_predictors(self):
        return self.design_matrix.shape[1]

    def __repr__(self):
        return f'{self.__class__.__name__}(' \
               f'seed={self.seed}, ' \
               f'num_time_steps={self.num_time_steps}, ' \
               f'num_predictors={self.num_predictors}' \
               f')'

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if not hasattr(other, 'as_df'):
            return False

        df1 = self.as_df()
        df2 = other.as_df()
        return (df1.index.equals(df2.index) and
                df1.columns.equals(df2.columns) and
                np.allclose((df1.values - df2.values).astype(float), 0))

    @property
    def predictor_colnames(self):
        return self._predictor_colnames

    @property
    def colnames(self):
        return self.predictor_colnames + self.metadata_colnames

    def as_df(self):
        df = pd.DataFrame(index=pd.Index(np.arange(self.num_time_steps), name='time_step'),
                          columns=self.colnames, dtype=np.float)
        df.loc[:, self.predictor_colnames] = self.design_matrix
        df.loc[:, self._action_colname] = self.actions
        df.loc[:, self._optimal_action_colname] = self.optimal_actions
        df.loc[:, self._reward_colname] = self.rewards
        df.loc[:, self._optimal_reward_colname] = self.optimal_rewards
        df.loc[:, self._compute_time_colname] = self.time_per_decision
        return df

    @classmethod
    def from_df(cls, df, seed=None):
        num_time_steps = df.shape[0]
        num_predictors = df.shape[1] - len(cls.metadata_colnames)

        # get predictor colnames from DF to avoid losing them
        colnames = list(df.columns)
        for name in cls.metadata_colnames:
            colnames.remove(name)

        instance = cls(seed, num_time_steps, num_predictors, predictor_colnames=colnames)

        instance.design_matrix.loc[:] = df.loc[:, instance.predictor_colnames]
        instance.actions[:] = df[instance._action_colname]
        instance.optimal_actions[:] = df[instance._optimal_action_colname]
        instance.rewards[:] = df[instance._reward_colname]
        instance.optimal_rewards[:] = df[instance._optimal_reward_colname]
        instance.time_per_decision[:] = df[instance._compute_time_colname]

        return instance

    def save(self, path):
        # TODO: save loses start and end
        # TODO: flexible version of this: path = self._standardize_path(path)
        df = self.as_df()
        logger.info(f'saving metrics to {path}')
        df.to_csv(path, index=False)

    def _standardize_path(self, path):
        name, ext = os.path.splitext(path)
        return f'{name}_{self.seed}.csv'

    @classmethod
    def load(cls, path):
        df = pd.read_csv(path)
        instance = cls.from_df(df)
        instance.seed = cls._seed_from_path(path)
        return instance

    @classmethod
    def _seed_from_path(cls, path):
        base = os.path.basename(path).split('_')[0]
        return int(os.path.splitext(base)[0])


class ExperimentMetrics:
    """Record metrics from multiple replications of the same experiment.

    Experiment metrics consist of:

    1.  ReplicationMetrics for each replication of the experiment
        1.  metadata associated with the replication (e.g. random
            seed and start, end timestamps
    2.  metadata associated with the overall experiment, such as
        the simulation name, model identifiers and hyperparams, etc.

    Experiment metrics serialize as a directory containing:

    1.  an index.json file containing metadata about each replication
    2.  the serialized ReplicationMetrics CSV for each replication,
        in a subdirectory named 'replications'

    """
    def __init__(self, metadata):
        self.metadata = metadata
        self.replications = {}

    def __repr__(self):
        return f'{self.__class__.__name__}(metadata={self.metadata})'

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return (self.equal_metadata(other) and
                self.replications == other.replications)

    def equal_metadata(self, other):
        kwargs = dict(cls=serialize.NumpyEncoder, sort_keys=True)
        return (json.dumps(self.metadata, **kwargs) ==
                json.dumps(other.metadata, **kwargs))

    def __getitem__(self, seed):
        return self.replications[seed]

    def __iter__(self):
        return iter(self.replications.values())

    def add_replication(self, metrics):
        self.replications[metrics.seed] = metrics

    def add_replications(self, metrics):
        for m in metrics:
            self.add_replication(m)

    def plot_cum_regret(self):
        rewards = np.array([m.rewards for m in self])
        optimals = np.array([m.optimal_rewards for m in self])
        return plot_cum_regret(rewards, optimals)

    def save(self, dirpath):
        """Save each metrics object at `dirpath/<seed>.csv`.

        This operation is not atomic -- failures will leave any outputs
        generated so far.

        Raises:
            OSError: if `dirpath` exists.
        """
        if not self.replications:
            raise ValueError('There are no replications to save')

        # TODO: better top-level log
        logger.info(f'Saving {len(self.replications)} replications to {dirpath}')
        os.makedirs(dirpath)

        replications_path = os.path.join(dirpath, 'replications')
        os.makedirs(replications_path)

        replication_metadata = self.write_index(dirpath)
        self.write_replication_metrics(dirpath, replication_metadata)

    def write_index(self, dirpath):
        replication_metadata = [
            {
                'metrics_path': os.path.join('replications', f'{metrics.seed}.csv'),
                'seed': metrics.seed,
                'start_time': metrics.start.isoformat(),
                'end_time': metrics.end.isoformat()
            }
            for metrics in self.replications.values()
        ]

        index = {'metadata': self.metadata,
                 'replications': replication_metadata}
        index_fpath = os.path.join(dirpath, 'index.json')
        logger.info(f'writing index.json to {index_fpath}')
        with open(index_fpath, 'w') as f:
            json.dump(index, f, indent=4, cls=serialize.NumpyEncoder)

        return replication_metadata

    def write_replication_metrics(self, dirpath, replication_metadata):
        paths = [os.path.join(dirpath, meta['metrics_path'])
                 for meta in replication_metadata]

        replications_path = os.path.join(dirpath, 'replications')
        logger.info(f'writing {len(self.replications)} to {replications_path}')
        with futures.ThreadPoolExecutor() as pool:
            submitted = []
            for metrics, path in zip(self.replications.values(), paths):
                submitted.append(pool.submit(metrics.save, path))

            futures.wait(submitted)

    @classmethod
    def load(cls, dirpath):
        index = cls.load_index(dirpath)
        metrics = cls.load_replication_metrics(dirpath, index['replications'])

        exp_metrics = cls(index['metadata'])
        exp_metrics.add_replications(metrics)
        return exp_metrics

    @classmethod
    def load_index(cls, dirpath):
        index_fpath = os.path.join(dirpath, 'index.json')
        logger.info(f'loading experiment metrics index from {index_fpath}')
        with open(index_fpath) as f:
            return json.load(f, object_hook=serialize.decode_object)

    @classmethod
    def load_replication_metrics(cls, dirpath, replication_meta):
        replications_path = os.path.join(dirpath, 'replications')
        metrics_paths = [os.path.join(replications_path, path)
                         for path in os.listdir(replications_path)]
        logger.info(f'loading {len(metrics_paths)} metrics from {replications_path}')

        meta_map = {m['seed']: m for m in replication_meta}
        with futures.ThreadPoolExecutor() as pool:
            all_metrics = pool.map(ReplicationMetrics.load, metrics_paths)
            for metrics, path in zip(all_metrics, metrics_paths):
                meta = meta_map[metrics.seed]
                metrics.start = datetime.datetime.strptime(meta['start_time'], ISO_8601_FMT)
                metrics.end = datetime.datetime.strptime(meta['end_time'], ISO_8601_FMT)
                yield metrics


class Experiment(Seedable):
    def __init__(self, model, env, *,
                 logging_frequency=100, max_workers=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.env = env

        self.logging_frequency = logging_frequency
        self.max_workers = max_workers
        if self.max_workers is None:
            import multiprocessing as mp
            self.max_workers = mp.cpu_count() - 1

    @property
    def num_time_steps(self):
        return self.env.spec.max_episode_steps

    @property
    def metadata(self):
        return {
            'env_name': self.env.env.spec.id,
            'num_time_steps': self.num_time_steps,
            'model_name': f'{self.model.__class__.__module__}.{self.model.__class__.__name__}',
            'model_hash': versioning.hash_class(self.model.__class__),
            'hyperparams': self.model.get_hyperparams()
        }

    def run(self, num_replications=1):
        rep_nums = np.arange(num_replications)
        with futures.ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            all_metrics = list(pool.map(self.run_once, rep_nums))

        num_failed = sum(1 for m in all_metrics if m is None)
        logger.info(f'{num_failed} of {num_replications} failed')

        successful_replication_metrics = [m for m in all_metrics if m is not None]
        exp_metrics = ExperimentMetrics(self.metadata)
        exp_metrics.add_replications(successful_replication_metrics)
        return exp_metrics

    def run_once(self, seed):
        self.model.seed(seed)
        self.env.seed(seed)
        replication_name = f'Replication_{seed}'

        try:
            self._unsafe_run_once(replication_name)
        except Exception as exc:
            logger.error(f'{replication_name} failed due to: {exc}')
            logger.exception(exc)
            return None

        return self.env.metrics

    def _unsafe_run_once(self, replication_name):
        obs = self.env.reset()
        for t in range(1, self.num_time_steps + 1):
            if t % self.logging_frequency == 0:
                logger.info(f'{replication_name} at t={t}')

            try:
                action = self.model.choose_arm(obs)
            except NotFitted:
                action = self.env.action_space.sample()

            obs, reward, done, info = self.env.step(action)
            if done:
                logger.info(f"{replication_name} finished after {t} timesteps")
                return

            past_contexts = self.env.metrics.design_matrix.iloc[:t]
            past_rewards = self.env.metrics.rewards[:t]
            try:
                self.model.fit(past_contexts, past_rewards)
            except InsufficientData as exc:
                logger.info(f'In {replication_name} at time step {t}, '
                            f'unable to fit model due to: {exc}')
            except Exception:
                logger.error(f'model fitting failed at time step {t}, '
                             f'unexpected exception')
                raise
