import pytest
import numpy as np
import pandas as pd

from cmabeval import models
from cmabeval.transformers import Preprocessor


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def rand_df(rng):
    num_arms = 5
    num_context = 10
    real_colnames = [f'p{i}' for i in range(num_context)]
    context_data = rng.normal(0, 1, size=(num_arms, num_context))
    arm_data = np.arange(num_arms)
    data = pd.DataFrame(index=pd.Index(arm_data),
                        columns=pd.Index(['arm'] + real_colnames))

    data.loc[:, real_colnames] = context_data
    data['arm'] = arm_data
    return data.astype({
        'arm': 'category',
        **{name: 'float' for name in real_colnames}})


@pytest.fixture
def rand_dmat(rand_df):
    return Preprocessor().fit_transform(rand_df)


def test_build_by_type_mapping(rand_dmat):
    expected = np.array([0] * 5 + [1] * 10)
    mapping = models.build_by_type_mapping(rand_dmat)
    assert np.array_equal(mapping, expected)
