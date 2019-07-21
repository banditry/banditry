import time
import datetime

import gym
import numpy as np
import pandas as pd
from scipy import stats, special

from banditry.base import Seedable
from banditry.experiment import ReplicationMetrics


def register_env(env, num_arms, num_context, num_time_steps=1000,
                 num_replications=100, version=0, seed=42):
    env_name = (f'CMAB1Best{env.__name__}'
                f'N{num_arms}C{num_context}T{num_time_steps}-v{version}')
    gym.envs.registry.env_specs.pop(env_name, None)
    gym.envs.register(
        env_name,
        trials=num_replications, max_episode_steps=num_time_steps,
        entry_point=env, kwargs=dict(
            num_arms=num_arms, num_context=num_context,
            num_time_steps=num_time_steps, seed=seed))

    return env_name


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
        dtypes = {colname: self._last_observation[colname].dtype
                  for colname in self._last_observation.columns}
        self.metrics.design_matrix = self.metrics.design_matrix.astype(dtypes)

        self.metrics.start = datetime.datetime.now()

        self._current_step = 0
        self._compute_start = time.time()
        return self._last_observation


class SeedableDiscrete(gym.spaces.Discrete, Seedable):

    def __init__(self, n, **kwargs):
        gym.spaces.Discrete.__init__(self, n)
        Seedable.__init__(self, **kwargs)

    def sample(self):
        return self.rng.randint(self.n)


class ContextualBanditEnv(Seedable, gym.Env):

    def render(self, mode='human'):
        raise NotImplementedError

    def __init__(self, num_arms, num_context, num_time_steps, **kwargs):
        Seedable.__init__(self, **kwargs)  # implements seed and reset

        self.num_arms = num_arms
        self.num_context = num_context
        self.num_predictors = 1 + num_context  # 1 for arm categorical
        self.num_time_steps = num_time_steps

        self._context_colnames = [f'p{i}' for i in range(num_context)]
        self._base_obs = pd.Series(range(num_arms), dtype='category').to_frame('arm')
        self._last_observation = None

        self.reward_range = (0, 1)
        self.action_space = SeedableDiscrete(self.num_arms)
        self.observation_space = self.create_observation_space()

        self._last_observation = None

        # Use a fixed random seed for this part so environment is always the same
        rng = np.random.RandomState(42)
        self.context_dist = self.create_context_dist(rng)
        self.interaction_effects = self.create_interaction_effects(rng)
        self.arm_effects = self.create_arm_effects(rng)

    def create_observation_space(self):
        # TODO: fix bounds on this so `sample` actually stays in bounds
        # Also, make it seedable like the action space for repeatability
        return gym.spaces.Box(
            low=0, high=np.inf, shape=(self.num_predictors,), dtype=np.float)

    def create_context_dist(self, rng):
        return stats.truncnorm(0, 10, loc=0, scale=0.5)

    def create_interaction_effects(self, rng):
        return np.zeros((self.num_arms, self.num_context))

    def create_arm_effects(self, rng):
        return np.zeros(self.num_arms)

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
        self.action_space.reset()
        return self._next_observation()

    def step(self, action):
        logits = (self.arm_effects[action] +
                  self.interaction_effects.dot(self._last_context))
        rates = special.expit(logits)
        rewards = self.rng.binomial(n=1, p=rates)

        actual_reward = rewards[action]
        optimal_action = rates.argmax()
        optimal_reward = rewards[optimal_action]

        info = dict(optimal_action=optimal_action,
                    optimal_reward=optimal_reward)
        next_observation = self._next_observation()

        done = False  # will be handled by wrapper
        return next_observation, actual_reward, done, info


class OnlyInteractionEffects(ContextualBanditEnv):

    def create_interaction_effects(self, rng):
        effects = np.ndarray((self.num_arms, self.num_context))
        effect_dist = stats.norm(-1, 0.5)

        shared_effects = effect_dist.rvs(size=self.num_context, random_state=rng)
        effects[:-1] = (np.tile(shared_effects, self.num_arms - 1)
                        .reshape(self.num_arms - 1, self.num_context))

        # The last one will have just slightly better effects.
        effects[-1] = shared_effects + stats.truncnorm.rvs(
            0.4, 0.7, loc=0.5, scale=0.1,
            size=self.num_context, random_state=rng)

        return effects


class OnlyArmEffects(ContextualBanditEnv):

    def create_arm_effects(self, rng):
        return np.linspace(-4, -2, num=self.num_arms)


class ArmAndInteractionEffects(ContextualBanditEnv):

    def create_arm_effects(self, rng):
        return np.linspace(-2, -4, num=self.num_arms)

    def create_interaction_effects(self, rng):
        effects = np.ndarray((self.num_arms, self.num_context))
        effect_dist = stats.norm(-1, 0.5)

        shared_effects = effect_dist.rvs(size=self.num_context, random_state=rng)
        effects[:-1] = (np.tile(shared_effects, self.num_arms - 1)
                        .reshape(self.num_arms - 1, self.num_context))

        # The last one will have just slightly better effects.
        effects[-1] = shared_effects + stats.truncnorm.rvs(
            0.4, 0.7, loc=0.5, scale=0.1,
            size=self.num_context, random_state=rng)

        return effects
