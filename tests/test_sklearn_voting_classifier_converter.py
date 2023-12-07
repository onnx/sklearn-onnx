# SPDX-License-Identifier: Apache-2.0

import unittest
from textwrap import dedent
from io import StringIO
import packaging.version as pv
import numpy
import pandas
from sklearn import __version__ as sklver
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from onnxruntime import InferenceSession
from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.proto import onnx_proto
from skl2onnx.common._apply_operation import apply_mul
from skl2onnx.common.data_types import guess_data_type
from test_utils import (
    dump_multiple_classification,
    dump_binary_classification,
    dump_data_and_model,
    TARGET_OPSET,
)


class CustomTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        TransformerMixin.__init__(self)
        BaseEstimator.__init__(self)

    def fit(self, X, y, sample_weight=None):
        pass

    def transform(self, X):
        return X * numpy.array([[0.5, 0.1, 10], [0.5, 0.1, 10]]).T


def custom_transform_shape_calculator(operator):
    operator.outputs[0].type = FloatTensorType([3, 2])


def custom_tranform_converter(scope, operator, container):
    input = operator.inputs[0]
    output = operator.outputs[0]

    weights_name = scope.get_unique_variable_name("weights")
    atype = onnx_proto.TensorProto.FLOAT
    weights = [0.5, 0.1, 10]
    shape = [len(weights), 1]
    container.add_initializer(weights_name, atype, shape, weights)
    apply_mul(scope, [input.full_name, weights_name], output.full_name, container)


class TestVotingClassifierConverter(unittest.TestCase):
    def test_operator_mul(self):
        model = CustomTransform()
        Xd = numpy.array([[1, 2], [3, 4], [4, 5]])

        model_onnx = convert_sklearn(
            model,
            "CustomTransform",
            [("input", FloatTensorType([None, Xd.shape[1]]))],
            custom_shape_calculators={
                CustomTransform: custom_transform_shape_calculator
            },
            custom_conversion_functions={CustomTransform: custom_tranform_converter},
            target_opset=TARGET_OPSET,
        )
        dump_data_and_model(
            Xd.astype(numpy.float32), model, model_onnx, basename="CustomTransformerMul"
        )

    @unittest.skipIf(TARGET_OPSET < 9, reason="not available")
    def test_voting_hard_binary(self):
        model = VotingClassifier(
            voting="hard",
            flatten_transform=False,
            estimators=[
                ("lr", LogisticRegression()),
                ("lr2", LogisticRegression(fit_intercept=False)),
            ],
        )
        # predict_proba is not defined when voting is hard.
        dump_binary_classification(
            model, suffix="Hard", comparable_outputs=[0], target_opset=TARGET_OPSET
        )

    @unittest.skipIf(TARGET_OPSET < 9, reason="not available")
    def test_voting_hard_binary_weights(self):
        model = VotingClassifier(
            voting="hard",
            flatten_transform=False,
            weights=numpy.array([1000, 1]),
            estimators=[
                ("lr", LogisticRegression()),
                ("lr2", LogisticRegression(fit_intercept=False)),
            ],
        )
        # predict_proba is not defined when voting is hard.
        dump_binary_classification(
            model,
            suffix="WeightsHard",
            comparable_outputs=[0],
            target_opset=TARGET_OPSET,
        )

    def test_voting_soft_binary(self):
        model = VotingClassifier(
            voting="soft",
            flatten_transform=False,
            estimators=[
                ("lr", LogisticRegression()),
                ("lr2", LogisticRegression(fit_intercept=False)),
            ],
        )
        dump_binary_classification(
            model, suffix="Soft", comparable_outputs=[0, 1], target_opset=TARGET_OPSET
        )

    def test_voting_soft_binary_weighted(self):
        model = VotingClassifier(
            voting="soft",
            flatten_transform=False,
            weights=numpy.array([1.8, 0.2]),
            estimators=[
                ("lr", LogisticRegression()),
                ("lr2", LogisticRegression(fit_intercept=False)),
            ],
        )
        dump_binary_classification(
            model, suffix="WeightedSoft", target_opset=TARGET_OPSET
        )

    @unittest.skipIf(TARGET_OPSET < 9, reason="not available")
    def test_voting_hard_multi(self):
        # predict_proba is not defined when voting is hard.
        model = VotingClassifier(
            voting="hard",
            flatten_transform=False,
            estimators=[
                ("lr", LogisticRegression()),
                ("lr2", DecisionTreeClassifier()),
            ],
        )
        dump_multiple_classification(
            model, suffix="Hard", comparable_outputs=[0], target_opset=TARGET_OPSET
        )

    @unittest.skipIf(TARGET_OPSET < 9, reason="not available")
    def test_voting_hard_multi_weighted(self):
        # predict_proba is not defined when voting is hard.
        model = VotingClassifier(
            voting="hard",
            flatten_transform=False,
            weights=numpy.array([1000, 1]),
            estimators=[
                ("lr", LogisticRegression()),
                ("lr2", DecisionTreeClassifier()),
            ],
        )
        dump_multiple_classification(
            model,
            suffix="WeightedHard",
            comparable_outputs=[0],
            target_opset=TARGET_OPSET,
        )

    def test_voting_soft_multi(self):
        model = VotingClassifier(
            voting="soft",
            flatten_transform=False,
            estimators=[
                ("lr", LogisticRegression()),
                ("lr2", LogisticRegression()),
            ],
        )
        dump_multiple_classification(model, suffix="Soft", target_opset=TARGET_OPSET)

    def test_voting_soft_multi_string(self):
        model = VotingClassifier(
            voting="soft",
            flatten_transform=False,
            estimators=[
                ("lr", LogisticRegression()),
                ("lr2", LogisticRegression()),
            ],
        )
        dump_multiple_classification(
            model, label_string=True, suffix="Soft", target_opset=TARGET_OPSET
        )

    def test_voting_soft_multi_weighted(self):
        model = VotingClassifier(
            voting="soft",
            flatten_transform=False,
            weights=numpy.array([1.8, 0.2]),
            estimators=[
                ("lr", LogisticRegression()),
                ("lr2", LogisticRegression()),
            ],
        )
        dump_multiple_classification(
            model, suffix="WeightedSoft", target_opset=TARGET_OPSET
        )

    def test_voting_soft_multi_weighted4(self):
        model = VotingClassifier(
            voting="soft",
            flatten_transform=False,
            weights=numpy.array([2.7, 0.3, 0.5, 0.5]),
            estimators=[
                ("lr", LogisticRegression()),
                ("lra", LogisticRegression()),
                ("lrb", LogisticRegression()),
                ("lr2", LogisticRegression()),
            ],
        )
        dump_multiple_classification(
            model, suffix="Weighted4Soft", target_opset=TARGET_OPSET
        )

    def test_voting_soft_multi_weighted42(self):
        model = VotingClassifier(
            voting="soft",
            flatten_transform=False,
            weights=numpy.array([27, 0.3, 0.5, 0.5]),
            estimators=[
                ("lr", LogisticRegression()),
                ("lra", LogisticRegression()),
                ("lrb", LogisticRegression()),
                ("lr2", LogisticRegression()),
            ],
        )
        dump_multiple_classification(
            model, suffix="Weighted42Soft", target_opset=TARGET_OPSET
        )

    @unittest.skipIf(
        pv.Version(sklver) < pv.Version("1.1.0"),
        reason="need more recent version of scikit-learn",
    )
    def test_voting_classifier_one_more_input(self):
        csv_x = dedent(
            """
        checking_status,duration,credit_history,purpose,credit_amount,savings_status,###
        employment,installment_commitment,personal_status,other_parties,residence_since,###
        property_magnitude,age,other_payment_plans,housing,existing_credits,job,###
        num_dependents,own_telephone,foreign_worker
        <0,6,critical/other existing credit,radio/tv,1169.0,no known savings,>=7,4,###
        male single,none,4,real estate,67,none,own,2,skilled,1,yes,yes
        0<=X<200,48,existing paid,radio/tv,5951.0,<100,1<=X<4,2,female div/dep/mar,###
        none,2,real estate,22,none,own,1,skilled,1,none,yes
        no checking,12,critical/other existing credit,education,2096.0,<100,4<=X<7,2,###
        male single,none,3,real estate,49,none,own,1,unskilled resident,2,none,yes
        <0,42,existing paid,furniture/equipment,7882.0,<100,4<=X<7,2,male single,###
        guarantor,4,life insurance,45,none,for free,1,skilled,2,none,yes
        <0,24,delayed previously,new car,4870.0,<100,1<=X<4,3,male single,none,4,###
        no known property,53,none,for free,2,skilled,2,none,yes
        no checking,36,existing paid,education,9055.0,no known savings,###
        1<=X<4,2,male single,###
        none,4,no known property,35,none,for free,1,unskilled resident,2,yes,yes
        no checking,24,existing paid,furniture/equipment,2835.0,500<=X<1000,###
        >=7,3,male single,###
        none,4,life insurance,53,none,own,1,skilled,1,none,yes
        0<=X<200,36,existing paid,used car,6948.0,<100,1<=X<4,2,###
        male single,none,2,car,35,###
        none,rent,1,high qualif/self emp/mgmt,1,yes,yes
        no checking,12,existing paid,radio/tv,3059.0,>=1000,4<=X<7,2,###
        male div/sep,none,4,###
        real estate,61,none,own,1,unskilled resident,1,none,yes
        0<=X<200,30,critical/other existing credit,new car,5234.0,<100,unemployed,4,###
        male mar/wid,none,2,car,28,none,own,2,high qualif/self emp/mgmt,1,none,yes
        """.replace(
                "###\n        ", ""
            )
        )

        X = pandas.read_csv(StringIO(csv_x))
        self.assertEqual(X.shape, (10, 20))
        y = [
            "good",
            "bad",
            "bad",
            "bad",
            "bad",
            "good",
            "good",
            "good",
            "good",
            "bad",
        ]

        model1 = Pipeline(
            steps=[
                (
                    "concat",
                    ColumnTransformer(
                        [("concat", "passthrough", list(range(X.shape[1])))],
                        sparse_threshold=0,
                    ),
                ),
                (
                    "voting",
                    VotingClassifier(
                        flatten_transform=False,
                        estimators=[
                            (
                                "est",
                                Pipeline(
                                    steps=[
                                        # This encoder is placed before
                                        # SimpleImputer because
                                        # onnx does not support text for Imputer.
                                        ("encoder", OrdinalEncoder()),
                                        (
                                            "imputer",
                                            SimpleImputer(strategy="most_frequent"),
                                        ),
                                        (
                                            "rf",
                                            RandomForestClassifier(
                                                n_estimators=4,
                                                max_depth=4,
                                                random_state=0,
                                            ),
                                        ),
                                    ],
                                ),
                            ),
                        ],
                    ),
                ),
            ]
        )

        model2 = Pipeline(
            steps=[
                (
                    "concat",
                    ColumnTransformer(
                        [
                            (
                                "concat",
                                Pipeline(
                                    steps=[
                                        # This encoder is placed before
                                        # simpleImputer because
                                        # onnx does not support text for Imputer.
                                        ("encoder", OrdinalEncoder()),
                                        (
                                            "imputer",
                                            SimpleImputer(strategy="most_frequent"),
                                        ),
                                    ],
                                ),
                                list(range(X.shape[1])),
                            )
                        ],
                        sparse_threshold=0,
                    ),
                ),
                (
                    "voting",
                    VotingClassifier(
                        flatten_transform=False,
                        estimators=[
                            (
                                "est",
                                RandomForestClassifier(
                                    n_estimators=4,
                                    max_depth=4,
                                    random_state=0,
                                ),
                            ),
                        ],
                    ),
                ),
            ]
        )

        models = [model1, model2]
        for model in models:
            model.fit(X, y)
            expected = model.predict(X)
            schema = guess_data_type(X)

            onnx_model = to_onnx(
                model=model,
                initial_types=schema,
                options={"zipmap": False},
                target_opset=TARGET_OPSET,
            )

            sess = InferenceSession(
                onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            inputs = {c: X[c].to_numpy().reshape((-1, 1)) for c in X.columns}
            got = sess.run(None, inputs)
            self.assertEqual(expected.tolist(), got[0].tolist())


if __name__ == "__main__":
    TestVotingClassifierConverter().test_voting_classifier_one_more_input()
    unittest.main(verbosity=2)
