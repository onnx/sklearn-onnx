# SPDX-License-Identifier: Apache-2.0

"""
Test scikit-learn's IsolationForest.
"""

import logging
import unittest
import packaging.version as pv
import numpy as np
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession
from sklearn import __version__ as sklv

try:
    from sklearn.ensemble import IsolationForest
except ImportError:
    IsolationForest = None
from skl2onnx import to_onnx
from test_utils import dump_data_and_model, TARGET_OPSET, TARGET_OPSET_ML

try:
    from onnxruntime.capi.onnxruntime_pybind11_state import NotImplemented
except ImportError:
    NotImplemented = RuntimeError


sklv2 = ".".join(sklv.split(".")[:2])


class TestSklearnIsolationForest(unittest.TestCase):
    @unittest.skipIf(IsolationForest is None, reason="old scikit-learn")
    @unittest.skipIf(
        pv.Version(sklv2) < pv.Version("0.22.0"), reason="tree structure is different."
    )
    @unittest.skipIf(TARGET_OPSET < 12, reason="not available")
    def test_isolation_forest(self):
        isol = IsolationForest(n_estimators=3, random_state=0)
        data = np.array(
            [[-1.1, -1.2], [0.3, 0.2], [0.5, 0.4], [100.0, 99.0]], dtype=np.float32
        )
        model = isol.fit(data)
        model_onnx = to_onnx(
            model, data, target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML}
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(data, model, model_onnx, basename="IsolationForest")

    @unittest.skipIf(IsolationForest is None, reason="old scikit-learn")
    @unittest.skipIf(
        pv.Version(sklv2) < pv.Version("0.22.0"), reason="tree structure is different."
    )
    @unittest.skipIf(TARGET_OPSET < 12, reason="not available")
    def test_isolation_forest_score_samples(self):
        isol = IsolationForest(n_estimators=3, random_state=0)
        data = np.array(
            [[-1.1, -1.2], [0.3, 0.2], [0.5, 0.4], [100.0, 99.0]], dtype=np.float32
        )
        model = isol.fit(data)
        model_onnx = to_onnx(
            model,
            data,
            options={"score_samples": True},
            target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML},
        )
        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        names = [o.name for o in sess.get_outputs()]
        self.assertEqual(names, ["label", "scores", "score_samples"])
        got = sess.run(None, {"X": data})
        self.assertEqual(len(got), 3)
        expected_label = isol.predict(data)
        expected_decif = isol.decision_function(data)
        expected_score = isol.score_samples(data)
        assert_almost_equal(expected_label, got[0].ravel())
        assert_almost_equal(expected_decif, got[1].ravel())
        assert_almost_equal(expected_score, got[2].ravel())

    @unittest.skipIf(IsolationForest is None, reason="old scikit-learn")
    def test_isolation_forest_single_training_sample(self):
        train = np.array([[1.0, 2.0]], dtype=np.float32)
        test = np.array([[1.0, 2.0], [100.0, -50.0], [-4.0, 10.0]], dtype=np.float32)
        model = IsolationForest(n_estimators=3, random_state=0).fit(train)
        model_onnx = to_onnx(
            model,
            train,
            options={"score_samples": True},
            target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML},
        )
        self._check_outputs(model, model_onnx, test)

    @unittest.skipIf(IsolationForest is None, reason="old scikit-learn")
    @unittest.skipIf(
        pv.Version(sklv2) < pv.Version("0.22.0"), reason="tree structure is different."
    )
    @unittest.skipIf(TARGET_OPSET < 12, reason="not available")
    def test_isolation_forest_op1(self):
        # The converter only needs TreeEnsembleRegressor from ai.onnx.ml,
        # which is available since opset 1.
        isol = IsolationForest(n_estimators=3, random_state=0)
        data = np.array(
            [[-1.1, -1.2], [0.3, 0.2], [0.5, 0.4], [100.0, 99.0]], dtype=np.float32
        )
        model = isol.fit(data)
        model_onnx = to_onnx(
            model, data, target_opset={"": TARGET_OPSET, "ai.onnx.ml": 1}
        )
        self._check_outputs(model, model_onnx, data)

    @unittest.skipIf(IsolationForest is None, reason="old scikit-learn")
    def test_isolation_forest_opset6(self):
        # Add and Mul need numpy-style broadcasting, available since opset 7.
        isol = IsolationForest(n_estimators=3, random_state=0)
        data = np.array(
            [[-1.1, -1.2], [0.3, 0.2], [0.5, 0.4], [100.0, 99.0]], dtype=np.float32
        )
        model = isol.fit(data)
        with self.assertRaises(RuntimeError):
            to_onnx(model, data, target_opset={"": 6, "ai.onnx.ml": 1})

    def _check_outputs(self, model, model_onnx, data, decimal=5):
        sess = InferenceSession(
            model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, {sess.get_inputs()[0].name: data})
        assert_almost_equal(model.predict(data), got[0].ravel())
        assert_almost_equal(
            model.decision_function(data), got[1].ravel(), decimal=decimal
        )
        if len(got) > 2:
            assert_almost_equal(
                model.score_samples(data), got[2].ravel(), decimal=decimal
            )
        return got

    @staticmethod
    def _outlier_data(n_features=6, dtype=np.float32):
        rs = np.random.RandomState(0)
        train = rs.randn(500, n_features)
        test = np.vstack(
            [rs.randn(300, n_features), rs.uniform(-6, 6, size=(50, n_features))]
        )
        return train.astype(dtype), test.astype(dtype)

    @unittest.skipIf(IsolationForest is None, reason="old scikit-learn")
    @unittest.skipIf(
        pv.Version(sklv2) < pv.Version("0.22.0"), reason="tree structure is different."
    )
    @unittest.skipIf(TARGET_OPSET < 12, reason="not available")
    def test_isolation_forest_many_trees(self):
        # The graph must not grow with the number of trees (issue #1128).
        train, test = self._outlier_data()
        sizes = {}
        for n_estimators in [10, 300]:
            model = IsolationForest(n_estimators=n_estimators, random_state=0).fit(
                train
            )
            model_onnx = to_onnx(
                model,
                train[:1],
                options={"score_samples": True},
                target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML},
            )
            sizes[n_estimators] = len(model_onnx.graph.node)
            got = self._check_outputs(model, model_onnx, test)
            # The test data must contain both inliers and outliers.
            self.assertEqual(set(got[0].ravel().tolist()), {-1, 1})
        self.assertEqual(sizes[10], sizes[300])

    @unittest.skipIf(IsolationForest is None, reason="old scikit-learn")
    @unittest.skipIf(
        pv.Version(sklv2) < pv.Version("0.22.0"), reason="tree structure is different."
    )
    @unittest.skipIf(TARGET_OPSET < 12, reason="not available")
    def test_isolation_forest_max_features(self):
        # Each tree uses a subset of the features.
        train, test = self._outlier_data()
        for max_features in [0.5, 2]:
            model = IsolationForest(
                n_estimators=50, max_features=max_features, random_state=0
            ).fit(train)
            model_onnx = to_onnx(
                model,
                train[:1],
                options={"score_samples": True},
                target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML},
            )
            self._check_outputs(model, model_onnx, test)

    @unittest.skipIf(IsolationForest is None, reason="old scikit-learn")
    @unittest.skipIf(
        pv.Version(sklv2) < pv.Version("0.22.0"), reason="tree structure is different."
    )
    @unittest.skipIf(TARGET_OPSET < 12, reason="not available")
    def test_isolation_forest_double(self):
        train, test = self._outlier_data(dtype=np.float64)
        model = IsolationForest(n_estimators=50, random_state=0).fit(train)
        model_onnx = to_onnx(
            model,
            train[:1],
            options={"score_samples": True},
            target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML},
        )
        got = self._check_outputs(model, model_onnx, test)
        self.assertEqual(got[1].dtype, np.float64)
        self.assertEqual(got[2].dtype, np.float64)

    @unittest.skipIf(IsolationForest is None, reason="old scikit-learn")
    @unittest.skipIf(
        pv.Version(sklv2) < pv.Version("0.22.0"), reason="tree structure is different."
    )
    @unittest.skipIf(TARGET_OPSET < 12, reason="not available")
    def test_isolation_forest_int64(self):
        rs = np.random.RandomState(0)
        train = rs.randint(0, 10, size=(300, 4)).astype(np.int64)
        test = np.vstack([train[:50], rs.randint(30, 60, size=(10, 4))]).astype(
            np.int64
        )
        model = IsolationForest(n_estimators=20, random_state=0).fit(train)
        model_onnx = to_onnx(
            model,
            train[:1],
            target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML},
        )
        self._check_outputs(model, model_onnx, test)

    @unittest.skipIf(IsolationForest is None, reason="old scikit-learn")
    @unittest.skipIf(
        pv.Version(sklv2) < pv.Version("0.22.0"), reason="tree structure is different."
    )
    @unittest.skipIf(TARGET_OPSET < 12, reason="not available")
    def test_isolation_forest_rnd(self):
        isol = IsolationForest(n_estimators=2, random_state=0)
        rs = np.random.RandomState(0)
        data = rs.randn(100, 4).astype(np.float32)
        data[-1, 2:] = 99.0
        data[-2, :2] = -99.0
        model = isol.fit(data)
        model_onnx = to_onnx(
            model, data, target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML}
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(data, model, model_onnx, basename="IsolationForestRnd")


if __name__ == "__main__":
    logging.getLogger("skl2onnx").setLevel(logging.ERROR)
    unittest.main()
