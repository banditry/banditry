import time
import datetime

import gym
import numpy as np
import pandas as pd
from scipy import stats, special

from cmabeval.base import Seedable
from cmabeval.experiment import ReplicationMetrics


class MetricsRecorder(gym.Wrapper):
    """Record metrics from an agent interacting with the wrapped environment."""

    def __init__(self, env):
        super().__init__(env)
        self.metrics = None
        self._current_step = 0
        self._compute_start = None
        self._last_observation = None

    def step(self, action):
        assert self._compute_start is not None, "Cannot call env.step() before calling reset()"
        time_for_decision = time.time() - self._compute_start
        observation, reward, done, info = self.env.step(action)

        # Record outcomes for this step
        t = self._current_step  # will be 0 on first call to step
        self.metrics.design_matrix.iloc[t] = self._last_observation.iloc[action]
        self.metrics.time_per_decision[t] = time_for_decision
        self.metrics.actions[t] = action
        self.metrics.optimal_actions[t] = info.get('optimal_action', np.nan)
        self.metrics.rewards[t] = reward
        self.metrics.optimal_rewards[t] = info.get('optimal_reward', np.nan)

        # Move to next step and restart timer
        self._current_step += 1
        if not done:
            self._last_observation = observation
            self._compute_start = time.time()  # reset compute timer
        else:
            self._compute_start = None
            self.metrics.end = datetime.datetime.now()

        return observation, reward, done, info

    def reset(self, **kwargs):
        self._last_observation = self.env.reset(**kwargs)
        self.metrics = ReplicationMetrics(
            self.env.initial_seed, self.env.num_time_steps, self.env.num_predictors,
            predictor_colnames=self._last_observation.columns)
        self.metrics.start = datetime.datetime.now()

        self._current_step = 0
        self._compute_start = time.time()
        return self._last_observation


class ContextualBanditEnv(Seedable, gym.Env):

    def render(self, mode='human'):
        raise NotImplementedError

    def __init__(self, num_arms, num_context, num_time_steps, **kwargs):
        Seedable.__init__(self, **kwargs)  # implements seed and reset

        self.num_arms = num_arms
        self.num_context = num_context
        self.num_predictors = num_arms + num_context
        self.num_time_steps = num_time_steps

        self._context_colnames = [f'p{i}' for i in range(num_context)]
        self._base_obs = pd.Series(range(num_arms), dtype='category').to_frame('arm')
        self._last_observation = None

        self.action_space = gym.spaces.Discrete(self.num_arms)
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(self.num_predictors,), dtype=np.float)
        self.reward_range = (0, 1)

        self._last_observation = None

        # Use a fixed random seed for this part so environment is always the same
        rng = np.random.RandomState(42)

        # Set up context distribution
        shared_variance = 0.5
        self.context_dist = stats.truncnorm(0, 10, loc=0, scale=shared_variance)

        # Set up arm effects.
        self.arm_effects = np.ndarray((self.num_arms, self.num_context))

        # All but one of the arms will have the same effects.
        effect_dist = stats.norm(-1, 0.5)
        shared_effects = effect_dist.rvs(size=self.num_context, random_state=rng)
        self.arm_effects[:-1] = (np.tile(shared_effects, self.num_arms - 1)
                                   .reshape(self.num_arms - 1, self.num_context))

        # The last one will have just slightly better effects.
        self.arm_effects[-1] = shared_effects + stats.truncnorm.rvs(
            0.4, 0.7, loc=0.5, scale=0.1,
            size=self.num_context, random_state=rng)

    def _next_observation(self):
        context = self.context_dist.rvs(size=self.num_context, random_state=self.rng)
        self._last_context = pd.Series(context)

        obs = self._base_obs.copy()
        for i, name in enumerate(self._context_colnames):
            obs[name] = context[i]

        self._last_observation = obs
        return self._last_observation

    def reset(self, **kwargs):
        Seedable.reset(self)
        return self._next_observation()

    def step(self, action):
        rates = special.expit(self.arm_effects.dot(self._last_context))
        rewards = self.rng.binomial(n=1, p=rates)
        optimal_action = rates.argmax()
        optimal_reward = rewards[optimal_action]
        actual_reward = rewards[action]

        info = dict(optimal_action=optimal_action,
                    optimal_reward=optimal_reward)
        next_observation = self._next_observation()

        done = False  # will be handled by wrapper
        return next_observation, actual_reward, done, info
