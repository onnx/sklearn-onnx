# SPDX-License-Identifier: Apache-2.0

"""
Test scikit-learn's LocalOutlierFactor.
"""
import unittest
from distutils.version import StrictVersion
import numpy as np
from numpy.testing import assert_almost_equal
from onnxruntime import __version__ as ort_version
from onnxruntime import InferenceSession
try:
    from onnxruntime.capi.onnxruntime_pybind11_state import InvalidGraph
except ImportError:
    InvalidGraph = RuntimeError
try:
    from sklearn.neighbors import LocalOutlierFactor
except ImportError:
    LocalOutlierFactor = None
from skl2onnx import to_onnx
from test_utils import TARGET_OPSET
try:
    from onnxruntime.capi.onnxruntime_pybind11_state import NotImplemented
except ImportError:
    NotImplemented = RuntimeError


ort_version = ".".join(ort_version.split('.')[:2])


class TestSklearnLocalOutlierForest(unittest.TestCase):

    @unittest.skipIf(LocalOutlierFactor is None, reason="old scikit-learn")
    def test_local_outlier_factor(self):
        lof = LocalOutlierFactor(n_neighbors=2, novelty=True)
        data = np.array([[-1.1, -1.2], [0.3, 0.2],
                         [0.5, 0.4], [100., 99.]], dtype=np.float32)
        model = lof.fit(data)
        model_onnx = to_onnx(model, data, target_opset=TARGET_OPSET)
        self.assertNotIn('CDist', str(model_onnx))

        data = data.copy()
        data[:, 0] += 0.1

        sess = InferenceSession(model_onnx.SerializeToString())
        names = [o.name for o in sess.get_outputs()]
        self.assertEqual(names, ['label', 'scores'])
        got = sess.run(None, {'X': data})
        self.assertEqual(len(got), 2)
        expected_label = lof.predict(data)
        expected_decif = lof.decision_function(data)
        assert_almost_equal(expected_label, got[0].ravel())
        assert_almost_equal(expected_decif, got[1].ravel())

    @unittest.skipIf(LocalOutlierFactor is None, reason="old scikit-learn")
    @unittest.skipIf(StrictVersion(ort_version) < StrictVersion("1.5.0"),
                     reason="CDist")
    def test_local_outlier_factor_cdist(self):
        lof = LocalOutlierFactor(n_neighbors=2, novelty=True)
        data = np.array([[-1.1, -1.2], [0.3, 0.2],
                         [0.5, 0.4], [100., 99.]], dtype=np.float32)
        model = lof.fit(data)
        model_onnx = to_onnx(model, data, target_opset=TARGET_OPSET,
                             options={'optim': 'cdist'})
        self.assertIn('CDist', str(model_onnx))

        data = data.copy()
        data[:, 0] += 0.1

        sess = InferenceSession(model_onnx.SerializeToString())
        names = [o.name for o in sess.get_outputs()]
        self.assertEqual(names, ['label', 'scores'])
        got = sess.run(None, {'X': data})
        self.assertEqual(len(got), 2)
        expected_label = lof.predict(data)
        expected_decif = lof.decision_function(data)
        assert_almost_equal(expected_label, got[0].ravel())
        assert_almost_equal(expected_decif, got[1].ravel())

    @unittest.skipIf(LocalOutlierFactor is None, reason="old scikit-learn")
    @unittest.skipIf(StrictVersion(ort_version) < StrictVersion("1.5.0"),
                     reason="CDist")
    def test_local_outlier_factor_p3(self):
        lof = LocalOutlierFactor(n_neighbors=2, novelty=True, p=3)
        data = np.array([[-1.1, -1.2], [0.3, 0.2],
                         [0.5, 0.4], [100., 99.]], dtype=np.float32)
        model = lof.fit(data)
        model_onnx = to_onnx(model, data, target_opset=TARGET_OPSET)
        self.assertNotIn('CDist', str(model_onnx))

        data = data.copy()
        data[:, 0] += 0.1

        sess = InferenceSession(model_onnx.SerializeToString())
        names = [o.name for o in sess.get_outputs()]
        self.assertEqual(names, ['label', 'scores'])
        got = sess.run(None, {'X': data})
        self.assertEqual(len(got), 2)
        expected_label = lof.predict(data)
        expected_decif = lof.decision_function(data)
        assert_almost_equal(expected_label, got[0].ravel())
        assert_almost_equal(expected_decif, got[1].ravel(), decimal=5)

    @unittest.skipIf(LocalOutlierFactor is None, reason="old scikit-learn")
    @unittest.skipIf(StrictVersion(ort_version) < StrictVersion("1.5.0"),
                     reason="CDist")
    def test_local_outlier_factor_cdist_p3(self):
        lof = LocalOutlierFactor(n_neighbors=2, novelty=True, p=3)
        data = np.array([[-1.1, -1.2], [0.3, 0.2],
                         [0.5, 0.4], [100., 99.]], dtype=np.float32)
        model = lof.fit(data)
        model_onnx = to_onnx(model, data, target_opset=TARGET_OPSET,
                             options={'optim': 'cdist'})
        self.assertIn('CDist', str(model_onnx))

        data = data.copy()
        data[:, 0] += 0.1

        try:
            sess = InferenceSession(model_onnx.SerializeToString())
        except InvalidGraph as e:
            if "Unrecognized attribute: p for operator CDist" in str(e):
                return
            raise e

        names = [o.name for o in sess.get_outputs()]
        self.assertEqual(names, ['label', 'scores'])
        got = sess.run(None, {'X': data})
        self.assertEqual(len(got), 2)
        expected_label = lof.predict(data)
        expected_decif = lof.decision_function(data)
        assert_almost_equal(expected_label, got[0].ravel())
        assert_almost_equal(expected_decif, got[1].ravel())

    @unittest.skipIf(LocalOutlierFactor is None, reason="old scikit-learn")
    def test_local_outlier_factor_metric(self):
        for metric in ['cityblock', 'euclidean', 'manhattan', 'sqeuclidean']:
            with self.subTest(metric=metric):
                lof = LocalOutlierFactor(n_neighbors=2, novelty=True,
                                         metric=metric)
                data = np.array([[-1.1, -1.2], [0.3, 0.2],
                                 [0.5, 0.4], [100., 99.]], dtype=np.float32)
                model = lof.fit(data)
                model_onnx = to_onnx(model, data, target_opset=TARGET_OPSET)

                data = data.copy()
                data[:, 0] += 0.1

                sess = InferenceSession(model_onnx.SerializeToString())
                names = [o.name for o in sess.get_outputs()]
                self.assertEqual(names, ['label', 'scores'])
                got = sess.run(None, {'X': data})
                self.assertEqual(len(got), 2)
                expected_label = lof.predict(data)
                expected_decif = lof.decision_function(data)
                assert_almost_equal(expected_label, got[0].ravel())
                assert_almost_equal(expected_decif, got[1].ravel(), decimal=4)

    @unittest.skipIf(LocalOutlierFactor is None, reason="old scikit-learn")
    @unittest.skipIf(StrictVersion(ort_version) < StrictVersion("1.5.0"),
                     reason="CDist")
    def test_local_outlier_factor_metric_cdist(self):
        for metric in ['euclidean', 'sqeuclidean']:
            with self.subTest(metric=metric):
                lof = LocalOutlierFactor(n_neighbors=2, novelty=True,
                                         metric=metric)
                data = np.array([[-1.1, -1.2], [0.3, 0.2],
                                 [0.5, 0.4], [100., 99.]], dtype=np.float32)
                model = lof.fit(data)
                model_onnx = to_onnx(model, data, target_opset=TARGET_OPSET,
                                     options={'optim': 'cdist'})

                data = data.copy()
                data[:, 0] += 0.1

                sess = InferenceSession(model_onnx.SerializeToString())
                names = [o.name for o in sess.get_outputs()]
                self.assertEqual(names, ['label', 'scores'])
                got = sess.run(None, {'X': data})
                self.assertEqual(len(got), 2)
                expected_label = lof.predict(data)
                expected_decif = lof.decision_function(data)
                assert_almost_equal(expected_label, got[0].ravel())
                assert_almost_equal(expected_decif, got[1].ravel(), decimal=4)

    @unittest.skipIf(LocalOutlierFactor is None, reason="old scikit-learn")
    @unittest.skipIf(TARGET_OPSET < 13, reason="TopK")
    @unittest.skipIf(StrictVersion(ort_version) < StrictVersion("1.7.0"),
                     reason="TopK")
    def test_local_outlier_factor_double(self):
        lof = LocalOutlierFactor(n_neighbors=2, novelty=True)
        data = np.array([[-1.1, -1.2], [0.3, 0.2],
                         [0.5, 0.4], [100., 99.]], dtype=np.float64)
        model = lof.fit(data)
        model_onnx = to_onnx(model, data, target_opset=TARGET_OPSET)

        sess = InferenceSession(model_onnx.SerializeToString())
        names = [o.name for o in sess.get_outputs()]
        self.assertEqual(names, ['label', 'scores'])
        got = sess.run(None, {'X': data})
        self.assertEqual(len(got), 2)
        expected_label = lof.predict(data)
        expected_decif = lof.decision_function(data)
        assert_almost_equal(expected_label, got[0].ravel())
        assert_almost_equal(expected_decif, got[1].ravel())

    @unittest.skipIf(LocalOutlierFactor is None, reason="old scikit-learn")
    def test_local_outlier_factor_score_samples(self):
        lof = LocalOutlierFactor(n_neighbors=2, novelty=True)
        data = np.array([[-1.1, -1.2], [0.3, 0.2],
                         [0.5, 0.4], [100., 99.]], dtype=np.float32)
        model = lof.fit(data)
        model_onnx = to_onnx(model, data, target_opset=TARGET_OPSET,
                             options={'score_samples': True})
        sess = InferenceSession(model_onnx.SerializeToString())
        names = [o.name for o in sess.get_outputs()]
        self.assertEqual(names, ['label', 'scores', 'score_samples'])
        got = sess.run(None, {'X': data})
        self.assertEqual(len(got), 3)
        expected_label = lof.predict(data)
        expected_decif = lof.decision_function(data)
        expected_score = lof.score_samples(data)
        assert_almost_equal(expected_label, got[0].ravel())
        assert_almost_equal(expected_decif, got[1].ravel(), decimal=5)
        assert_almost_equal(expected_score, got[2].ravel(), decimal=5)

    @unittest.skipIf(LocalOutlierFactor is None, reason="old scikit-learn")
    def test_local_outlier_factor_rnd(self):
        lof = LocalOutlierFactor(n_neighbors=2, novelty=True)
        rs = np.random.RandomState(0)
        data = rs.randn(100, 4).astype(np.float32)
        data[-1, 2:] = 99.
        data[-2, :2] = -99.
        model = lof.fit(data)
        model_onnx = to_onnx(model, data, target_opset=TARGET_OPSET)

        sess = InferenceSession(model_onnx.SerializeToString())
        names = [o.name for o in sess.get_outputs()]
        self.assertEqual(names, ['label', 'scores'])
        got = sess.run(None, {'X': data})
        self.assertEqual(len(got), 2)
        expected_label = lof.predict(data)
        expected_decif = lof.decision_function(data)
        assert_almost_equal(expected_label, got[0].ravel())
        assert_almost_equal(expected_decif, got[1].ravel(), decimal=5)


if __name__ == '__main__':
    unittest.main()
