import numpy as np
from pypolyagamma.pypolyagamma import PyPolyaGamma

from cmabeval.models import NotFitted


class Seedable:
    """Inherit from this class to get methods useful for objects
    using random seeds.
    """
    def __init__(self, seed=42):
        self._initial_seed = seed
        self.rng = np.random.RandomState(self._initial_seed)

    @property
    def initial_seed(self):
        return self._initial_seed

    def seed(self, seed):
        self._initial_seed = seed
        self.rng.seed(self._initial_seed)
        return self

    def reset(self):
        self.rng.seed(self._initial_seed)
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