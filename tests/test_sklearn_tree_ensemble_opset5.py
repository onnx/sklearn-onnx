# SPDX-License-Identifier: Apache-2.0
"""
Tree models converted with operator TreeEnsemble (ai.onnx.ml opset 5).
"""

import unittest
import packaging.version as pv
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import onnx
from onnx.reference import ReferenceEvaluator
from onnxruntime import InferenceSession, __version__ as ort_version
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from skl2onnx import to_onnx
from skl2onnx.common.data_types import Int64TensorType
from skl2onnx.common.tree_ensemble import convert_to_tree_ensemble_attributes
from test_utils import TARGET_IR, TARGET_OPSET

LEGACY = {"TreeEnsembleClassifier", "TreeEnsembleRegressor"}


def _data(kind, n_classes=2, n_targets=1, missing=False):
    if kind == "classifier":
        X, y = make_classification(
            n_samples=300,
            n_features=6,
            n_informative=4,
            n_classes=n_classes,
            random_state=0,
        )
    else:
        X, y = make_regression(
            n_samples=300, n_features=6, n_targets=n_targets, random_state=0
        )
    if missing:
        X[np.random.default_rng(0).random(X.shape) < 0.05] = np.nan
    return X, y


CLASSIFIERS = [
    (DecisionTreeClassifier(max_depth=4, random_state=0), 2, False),
    (DecisionTreeClassifier(max_depth=4, random_state=0), 3, False),
    (RandomForestClassifier(n_estimators=5, max_depth=4, random_state=0), 2, False),
    (RandomForestClassifier(n_estimators=5, max_depth=4, random_state=0), 3, False),
    (ExtraTreesClassifier(n_estimators=5, max_depth=4, random_state=0), 3, False),
    (GradientBoostingClassifier(n_estimators=10, random_state=0), 2, False),
    (GradientBoostingClassifier(n_estimators=10, random_state=0), 3, False),
    (
        GradientBoostingClassifier(n_estimators=10, loss="exponential", random_state=0),
        2,
        False,
    ),
    (HistGradientBoostingClassifier(max_iter=10, random_state=0), 2, True),
    (HistGradientBoostingClassifier(max_iter=10, random_state=0), 3, True),
]

REGRESSORS = [
    (DecisionTreeRegressor(max_depth=4, random_state=0), 1, False),
    (DecisionTreeRegressor(max_depth=4, random_state=0), 2, False),
    (RandomForestRegressor(n_estimators=5, max_depth=4, random_state=0), 1, False),
    (RandomForestRegressor(n_estimators=5, max_depth=4, random_state=0), 2, False),
    (GradientBoostingRegressor(n_estimators=10, random_state=0), 1, False),
    (HistGradientBoostingRegressor(max_iter=10, random_state=0), 1, True),
]


def _convert(model, X, ml_opset, **kwargs):
    return to_onnx(
        model, X[:1], target_opset={"": TARGET_OPSET, "ai.onnx.ml": ml_opset}, **kwargs
    )


def _run(onx, X):
    sess = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, {sess.get_inputs()[0].name: X})


@unittest.skipIf(
    pv.Version(ort_version) < pv.Version("1.21.0"),
    reason="onnxruntime implements TreeEnsemble from 1.21.0",
)
class TestSklearnTreeEnsembleOpset5(unittest.TestCase):
    def _check_operators(self, onx):
        op_types = {node.op_type for node in onx.graph.node}
        self.assertIn("TreeEnsemble", op_types)
        self.assertFalse(op_types & LEGACY, op_types)
        onnx.checker.check_model(onx)

    def test_classifiers(self):
        for model, n_classes, missing in CLASSIFIERS:
            X, y = _data("classifier", n_classes=n_classes, missing=missing)
            model.fit(X, y)
            for dtype in (np.float32, np.float64):
                if dtype == np.float64 and isinstance(
                    model, GradientBoostingClassifier
                ):
                    # The converter declares double probabilities and
                    # computes floats, with any opset.
                    continue
                with self.subTest(
                    model=type(model).__name__, n_classes=n_classes, dtype=dtype
                ):
                    Xt = X.astype(dtype)
                    onx = _convert(model, Xt, 5, options={"zipmap": False})
                    self._check_operators(onx)
                    label, proba = _run(onx, Xt)
                    assert_array_equal(label, model.predict(Xt))
                    assert_allclose(proba, model.predict_proba(Xt), atol=1e-5)
                    if dtype == np.float64 and hasattr(model, "_predictors"):
                        # Same type as the input, the comparisons and the sums
                        # are done in double.
                        assert_allclose(
                            proba, model.predict_proba(Xt), rtol=0, atol=1e-12
                        )
                    if dtype == np.float32:
                        # Same outputs as the deprecated operator.
                        label3, proba3 = _run(
                            _convert(model, Xt, 3, options={"zipmap": False}), Xt
                        )
                        assert_array_equal(label, label3)
                        assert_allclose(proba, proba3, atol=1e-6)

    def test_classifier_zipmap_and_string_labels(self):
        X, y = _data("classifier", n_classes=3)
        labels = np.array(["a", "b", "c"])[y]
        for model in (
            RandomForestClassifier(n_estimators=3, max_depth=3, random_state=0),
            HistGradientBoostingClassifier(max_iter=5, random_state=0),
        ):
            model.fit(X, labels)
            # ZipMap only accepts floats.
            for dtype in (np.float32,):
                with self.subTest(model=type(model).__name__, dtype=dtype):
                    Xt = X.astype(dtype)
                    onx = _convert(model, Xt, 5)
                    self._check_operators(onx)
                    label, proba = _run(onx, Xt)
                    assert_array_equal(label, model.predict(Xt))
                    expected = model.predict_proba(Xt)
                    for i, row in enumerate(proba):
                        assert_allclose(
                            [row[c] for c in model.classes_], expected[i], atol=1e-5
                        )

    def test_one_class(self):
        X, _ = _data("classifier")
        model = RandomForestClassifier(n_estimators=3, random_state=0)
        model.fit(X, np.full(X.shape[0], 7))
        Xt = X.astype(np.float32)
        onx = _convert(model, Xt, 5, options={"zipmap": False})
        self._check_operators(onx)
        label, proba = _run(onx, Xt)
        assert_array_equal(label, model.predict(Xt))
        assert_allclose(proba, model.predict_proba(Xt), atol=1e-6)

    def test_regressors(self):
        for model, n_targets, missing in REGRESSORS:
            X, y = _data("regressor", n_targets=n_targets, missing=missing)
            model.fit(X, y)
            for dtype in (np.float32, np.float64):
                with self.subTest(
                    model=type(model).__name__, n_targets=n_targets, dtype=dtype
                ):
                    Xt = X.astype(dtype)
                    onx = _convert(model, Xt, 5)
                    self._check_operators(onx)
                    (pred,) = _run(onx, Xt)
                    expected = model.predict(Xt).reshape(pred.shape)
                    assert_allclose(pred, expected, rtol=1e-5, atol=1e-4)
                    if dtype == np.float32:
                        (pred3,) = _run(_convert(model, Xt, 3), Xt)
                        assert_allclose(pred, pred3, rtol=1e-6, atol=1e-5)

    def test_integer_input(self):
        rng = np.random.default_rng(0)
        X = rng.integers(0, 10, size=(200, 4)).astype(np.int64)
        y = (X[:, 0] + X[:, 1] > 9).astype(np.int64)
        model = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)
        onx = to_onnx(
            model,
            initial_types=[("X", Int64TensorType([None, 4]))],
            target_opset={"": TARGET_OPSET, "ai.onnx.ml": 5},
            options={"zipmap": False},
        )
        self._check_operators(onx)
        label, proba = _run(onx, X)
        assert_array_equal(label, model.predict(X))
        assert_allclose(proba, model.predict_proba(X), atol=1e-6)

    def test_decision_path_and_leaf(self):
        X, y = _data("classifier", n_classes=3)
        Xt = X.astype(np.float32)
        options = {"zipmap": False, "decision_path": True, "decision_leaf": True}
        for model in (
            DecisionTreeClassifier(max_depth=3, random_state=0),
            RandomForestClassifier(n_estimators=3, max_depth=3, random_state=0),
        ):
            model.fit(X, y)
            with self.subTest(model=type(model).__name__):
                onx = _convert(model, Xt, 5, options=options)
                self._check_operators(onx)
                got = _run(onx, Xt)
                expected = _run(_convert(model, Xt, 3, options=options), Xt)
                for g, e in zip(got, expected):
                    if g.dtype.kind in "fc":
                        assert_allclose(g, e, atol=1e-6)
                    else:
                        assert_array_equal(g, e)

    def test_multi_output_classifier(self):
        X, y = _data("classifier", n_classes=3)
        Y = np.stack([y, (y + 1) % 3], axis=1)
        model = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, Y)
        Xt = X.astype(np.float32)
        onx = _convert(model, Xt, 5, options={"zipmap": False})
        self._check_operators(onx)
        label, proba = _run(onx, Xt)
        assert_array_equal(label, model.predict(Xt))
        expected = _run(_convert(model, Xt, 3, options={"zipmap": False}), Xt)
        assert_allclose(proba, expected[1], atol=1e-6)

    def test_one_vs_rest_integer_input(self):
        # The inner forest declares integer probabilities, the tree
        # operators return floats.
        rng = np.random.default_rng(0)
        X = rng.integers(0, 10, size=(200, 4)).astype(np.int64)
        y = (X[:, 0] + X[:, 1]) % 3
        model = OneVsRestClassifier(
            RandomForestClassifier(n_estimators=3, max_depth=3, random_state=0)
        ).fit(X, y)
        onx = to_onnx(
            model,
            initial_types=[("X", Int64TensorType([None, 4]))],
            target_opset={"": TARGET_OPSET, "ai.onnx.ml": 5},
            options={id(model): {"zipmap": False}},
        )
        self._check_operators(onx)
        label, proba = _run(onx, X)
        assert_array_equal(label, model.predict(X))
        assert_allclose(proba, model.predict_proba(X), atol=1e-5)

    def test_hist_gradient_boosting_thresholds(self):
        # Inputs equal to a threshold, or one float step away from it,
        # follow the same branch as scikit-learn.
        X, y = _data("classifier", missing=True)
        model = HistGradientBoostingClassifier(max_iter=20, random_state=0)
        model.fit(X, y)
        rng = np.random.default_rng(1)
        columns = []
        for thresholds in model._bin_mapper.bin_thresholds_:
            value = np.asarray(thresholds)[rng.integers(0, len(thresholds), 2000)]
            columns.append(value)
        X64 = np.stack(columns, axis=1)
        X32 = X64.astype(np.float32)
        step = rng.integers(-1, 2, X32.shape)
        X32 = np.where(
            step < 0,
            np.nextafter(X32, np.float32(-np.inf)),
            np.where(step > 0, np.nextafter(X32, np.float32(np.inf)), X32),
        )
        for Xt, atol in ((X64, 1e-12), (X32, 1e-5)):
            with self.subTest(dtype=Xt.dtype):
                onx = _convert(model, Xt, 5, options={"zipmap": False})
                label, proba = _run(onx, Xt)
                assert_allclose(proba, model.predict_proba(Xt), rtol=0, atol=atol)
                assert_array_equal(label, model.predict(Xt))

    def test_smaller_than_deprecated_operator(self):
        X, y = _data("classifier")
        model = HistGradientBoostingClassifier(max_iter=50, random_state=0)
        model.fit(X, y)
        Xt = X.astype(np.float32)
        size5 = len(_convert(model, Xt, 5).SerializeToString())
        size3 = len(_convert(model, Xt, 3).SerializeToString())
        self.assertLess(size5, size3)

    def test_reference_evaluator(self):
        X, y = _data("classifier", n_classes=3)
        model = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=0)
        model.fit(X, y)
        Xt = X.astype(np.float32)
        onx = _convert(model, Xt, 5, options={"zipmap": False})
        expected = _run(onx, Xt)
        got = ReferenceEvaluator(onx).run(None, {"X": Xt})
        assert_array_equal(got[0], expected[0])
        assert_allclose(got[1], expected[1], atol=1e-6)

    def test_root_branches_have_different_indices(self):
        # Tree 0 has one split and two leaves. The root of tree 1 has a leaf
        # and a split as children which would both get index 2:
        # onnxruntime < 1.24 then reads the root as a leaf.
        attrs = {
            "nodes_treeids": [0, 0, 0, 1, 1, 1, 1, 1],
            "nodes_nodeids": [0, 1, 2, 0, 1, 2, 3, 4],
            "nodes_featureids": [0, 0, 0, 1, 0, 2, 0, 0],
            "nodes_modes": ["BRANCH_LEQ", "LEAF", "LEAF"]
            + ["BRANCH_LEQ", "BRANCH_LEQ", "LEAF", "LEAF", "LEAF"],
            "nodes_values": [0.5, 0, 0, 0.5, 0.5, 0, 0, 0],
            "nodes_truenodeids": [1, 0, 0, 1, 2, 0, 0, 0],
            "nodes_falsenodeids": [2, 0, 0, 4, 3, 0, 0, 0],
            "nodes_missing_value_tracks_true": [0] * 8,
            "target_treeids": [0, 0, 1, 1, 1],
            "target_nodeids": [1, 2, 2, 3, 4],
            "target_ids": [0] * 5,
            "target_weights": [1.0, 2.0, 3.0, 4.0, 5.0],
            "n_targets": 1,
        }
        new = convert_to_tree_ensemble_attributes(attrs, np.float32)
        for root in new["tree_roots"]:
            self.assertFalse(
                new["nodes_truenodeids"][root] == new["nodes_falsenodeids"][root]
                and new["nodes_trueleafs"][root] != new["nodes_falseleafs"][root]
            )
        node = onnx.helper.make_node(
            "TreeEnsemble", ["X"], ["Y"], domain="ai.onnx.ml", n_targets=1, **new
        )
        graph = onnx.helper.make_graph(
            [node],
            "g",
            [onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, None)],
            [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)],
        )
        onx = onnx.helper.make_model(
            graph,
            ir_version=TARGET_IR,
            opset_imports=[
                onnx.helper.make_opsetid("", TARGET_OPSET),
                onnx.helper.make_opsetid("ai.onnx.ml", 5),
            ],
        )
        X = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=np.float32)
        (got,) = _run(onx, X)
        # tree 0: x0 <= 0.5 -> 1 else 2
        # tree 1: x1 <= 0.5 -> (x0 <= 0.5 -> 3 else 4) else 5
        assert_allclose(got.ravel(), [1 + 3, 1 + 5, 2 + 4, 2 + 5])


if __name__ == "__main__":
    unittest.main(verbosity=2)
