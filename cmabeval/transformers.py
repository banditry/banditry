import logging
import itertools

import patsy

logger = logging.getLogger(__name__)


class Preprocessor:
    """Dummy-encode categoricals, standardize reals."""

    def __init__(self, interactions=False):
        self.interactions = interactions

        self.design_info_ = None

    def fit(self, df):
        dmat = self.do_encoding(df)
        self.design_info_ = dmat.design_info
        return self

    def do_encoding(self, df):
        patsy_formula = build_patsy_formula(df, self.interactions)
        return patsy.dmatrix(patsy_formula, df)

    def transform(self, df):
        return patsy.build_design_matrices([self.design_info_], df)[0]

    def fit_transform(self, df):
        dmat = self.do_encoding(df)
        self.design_info_ = dmat.design_info
        return dmat


def build_patsy_formula(df, interactions=False):
    real_colnames = list(sorted(df.select_dtypes(['int', 'float']).columns))
    if df.shape[0] > 1:
        real_factors = [f'standardize({name})' for name in real_colnames]
    else:
        real_factors = real_colnames

    cat_factors = list(sorted(set(df.columns) - set(real_colnames)))
    all_factors = cat_factors + real_factors
    factors = ['0'] + all_factors

    # TODO: scale interaction terms involving reals
    if interactions:
        # Interact all covariates; use un-standardized reals
        interaction_factors = [f'{f1}:{f2}' for f1, f2, in
                               itertools.product(real_colnames + cat_factors)]
    else:
        interaction_factors = []

    factors += interaction_factors
    formula = ' + '.join(factors)
    logger.debug(f'constructed patsy from {len(cat_factors)} categorical factors, '
                 f'{len(real_factors)} real factors, and '
                 f'{len(interaction_factors)} interaction factors: "{formula}"')

    return formula
