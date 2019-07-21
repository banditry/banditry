import inspect
import itertools

import numpy as np
from pypolyagamma.pypolyagamma import PyPolyaGamma

from banditry.exceptions import NotFitted


def get_parents_with_hyperparams(klass):
    cutoff_set = {PGBaseModel, BaseModel}
    super_classes = klass.mro()

    idx = None
    for base_class in cutoff_set:
        try:
            idx = super_classes.index(base_class)
            break
        except ValueError:
            pass

    return super_classes[1:idx]


def get_hyperparam_names(klass):
    parents = get_parents_with_hyperparams(klass)
    all_classes = [klass] + parents
    return set(itertools.chain.from_iterable(
        inspect.getfullargspec(class_ref).args[1:]  # exclude self
        for class_ref in all_classes))


def get_hyperparams(model):
    names = get_hyperparam_names(model.__class__)
    return {name: getattr(model, name) for name in names}


class HasHyperparams:

    def get_hyperparams(self):
        return get_hyperparams(self)


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


class BaseModel(Seedable, HasFittableParams, HasHyperparams):

    def reset(self):
        Seedable.reset(self)
        HasFittableParams.reset_params(self)
        return self


class PGBaseModel(PGSeedable, HasFittableParams, HasHyperparams):

    def reset(self):
        PGSeedable.reset(self)
        HasFittableParams.reset_params(self)
        return self
