# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from sklearn.decomposition import SparseCoder
from sklearn.ensemble import (
    VotingClassifier, AdaBoostRegressor,
)
try:
    from sklearn.ensemble import VotingRegressor
except ImportError:
    # Available only in 0.21
    VotingRegressor = 'VotingRegressor'
try:
    from sklearn.ensemble import StackingRegressor, StackingClassifier
except ImportError:
    StackingRegressor = 'StackingRegressor'
    StackingClassifier = 'StackingClassifier'
from sklearn.feature_selection import SelectFromModel, RFE, RFECV
from sklearn.linear_model import (
    LogisticRegression, SGDClassifier, LinearRegression
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.multiclass import (
    OneVsRestClassifier, OneVsOneClassifier,
    OutputCodeClassifier
)
from sklearn.multioutput import (
    MultiOutputRegressor, MultiOutputClassifier,
)
try:
    from sklearn.multioutput import (
        ClassifierChain, RegressorChain
    )
except ImportError:
    # Available only in 0.21
    ClassifierChain = 'ClassifierChain'
    RegressorChain = 'RegressorChain'
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeRegressor


def build_custom_scenarios():
    """
    Defines parameters values for some operators.
    """
    return {
        # skips
        SparseCoder: None,
        # scenarios
        AdaBoostRegressor: [
            ('default', {
                'n_estimators': 5,
            }),
        ],
        ClassifierChain: [
            ('logreg', {
                'base_estimator': LogisticRegression(solver='liblinear'),
            })
        ],
        GridSearchCV: [
            ('cl', {
                'estimator': LogisticRegression(solver='liblinear'),
                'param_grid': {'fit_intercept': [False, True]},
            }),
            ('reg', {
                'estimator': LinearRegression(),
                'param_grid': {'fit_intercept': [False, True]},
            }),
        ],
        LocalOutlierFactor: [
            ('novelty', {
                'novelty': True,
            }),
        ],
        LogisticRegression: [
            ('liblinear', {
                'solver': 'liblinear',
            }),
        ],
        MultiOutputClassifier: [
            ('logreg', {
                'estimator': LogisticRegression(solver='liblinear'),
            })
        ],
        MultiOutputRegressor: [
            ('linreg', {
                'estimator': LinearRegression(),
            })
        ],
        NuSVC: [
            ('prob', {
                'probability': True,
            }),
        ],
        OneVsOneClassifier: [
            ('logreg', {
                'estimator': LogisticRegression(solver='liblinear'),
            })
        ],
        OneVsRestClassifier: [
            ('logreg', {
                'estimator': LogisticRegression(solver='liblinear'),
            })
        ],
        OutputCodeClassifier: [
            ('logreg', {
                'estimator': LogisticRegression(solver='liblinear'),
            })
        ],
        RandomizedSearchCV: [
            ('cl', {
                'estimator': LogisticRegression(solver='liblinear'),
                'param_distributions': {'fit_intercept': [False, True]},
            }),
            ('reg', {
                'estimator': LinearRegression(),
                'param_distributions': {'fit_intercept': [False, True]},
            }),
        ],
        RegressorChain: [
            ('linreg', {
                'base_estimator': LinearRegression(),
            })
        ],
        RFE: [
            ('cl', {
                'estimator': LogisticRegression(solver='liblinear'),
            }),
            ('reg', {
                'estimator': LinearRegression(),
            })
        ],
        RFECV: [
            ('cl', {
                'estimator': LogisticRegression(solver='liblinear'),
            }),
            ('reg', {
                'estimator': LinearRegression(),
            })
        ],
        SelectFromModel: [
            ('rf', {
                'estimator': DecisionTreeRegressor(),
            }),
        ],
        SGDClassifier: [
            ('log', {
                'loss': 'log',
            }),
        ],
        StackingClassifier: [
            ('logreg', {
                'estimators': [LogisticRegression()],
            }),
        ],
        StackingRegressor: [
            ('linreg', {
                'estimators': [LinearRegression()],
            }),
        ],
        SVC: [
            ('prob', {
                'probability': True,
            }),
        ],
        VotingClassifier: [
            ('logreg-noflatten', {
                'voting': 'soft',
                'flatten_transform': False,
                'estimators': [
                    ('lr1', LogisticRegression(solver='liblinear')),
                    ('lr2', LogisticRegression(
                        solver='liblinear', fit_intercept=False)),
                ],
            })
        ],
        VotingRegressor: [
            ('linreg', {
                'estimators': [
                    ('lr1', LinearRegression()),
                    ('lr2', LinearRegression(fit_intercept=False)),
                ],
            })
        ],
    }


_extra_parameters = build_custom_scenarios()
