import unittest
from distutils.version import StrictVersion
import numpy as np
from sklearn.exceptions import ConvergenceWarning
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import Binarizer
from onnxruntime import InferenceSession
try:
    from onnxruntime.capi.onnxruntime_pybind11_state import Fail as OrtFail
except ImportError:
    OrtFail = RuntimeError
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn, to_onnx
from skl2onnx.common.data_types import DoubleTensorType
from onnxruntime import __version__ as ort_version
from test_utils import (
    dump_data_and_model, fit_classification_model)  # , TARGET_OPSET)

TARGET_OPSET = 12  # change when PR 551
warnings_to_skip = (DeprecationWarning, FutureWarning, ConvergenceWarning)


class TestSklearnDoubleTensorTypeTransformer(unittest.TestCase):

    def _common_transform(
            self, model_cls_set, name_root=None, debug=False):
        for model_cls in model_cls_set:
            if name_root is None:
                name = model_cls.__name__
            else:
                name = name_root
            
            model = model_cls()
            X = np.random.randn(100, 4).astype(np.float64)
            model.fit(X)
            X = np.random.randn(100, 4).astype(np.float64)
            pmethod = 'transform'
            with self.subTest(model=name):
                options = {}
                model_onnx = convert_sklearn(
                    model, "model",
                    [("input", DoubleTensorType([None, X.shape[1]]))],
                    target_opset=TARGET_OPSET,
                    options={id(model): options})
                if debug:
                    print(model_onnx)
                self.assertIn("elem_type: 11", str(model_onnx))
                methods = [pmethod]
                dump_data_and_model(
                    X.astype(np.float64), model, model_onnx,
                    methods=methods,
                    basename="Sklearn{}Double".format(name))

    @ignore_warnings(category=warnings_to_skip)
    def test_scaler_64(self):
        self._common_transform([StandardScaler])

    def _fit_model_binary_classification(self, model, data, **kwargs):
        X = data.data
        y = data.target
        mid_point = len(data.target_names) / 2
        y[y < mid_point] = 0
        y[y >= mid_point] = 1
        model.fit(X, y)
        return model, X.astype(np.float32)

    def _fit_model_multiclass_classification(self, model, data):
        X = data.data
        y = data.target
        model.fit(X, y)
        return model, X.astype(np.float32)

    def _test_score(self, model, X, tg, decimal=5, black_op=None):
        X = X.astype(np.float32)
        exp = model.score_samples(X)
        expp = model.predict_proba(X)
        onx = to_onnx(
            model, X[:1], target_opset=tg,
            options={id(model): {'score_samples': True}},
            black_op=black_op)
        try:
            sess = InferenceSession(onx.SerializeToString())
        except OrtFail as e:
            raise RuntimeError('Issue {}\n{}'.format(e, str(onx)))
        got = sess.run(None, {'X': X})
        self.assertEqual(len(got), 3)
        np.testing.assert_almost_equal(
            expp.ravel(), got[1].ravel(), decimal=decimal)
        np.testing.assert_almost_equal(
            exp.ravel(), got[2].ravel(), decimal=decimal)

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="onnxruntime misses Gemm for double")
    def test_model_gaussian_mixture_binary_classification(self):
        model, X = self._fit_model_binary_classification(
            GaussianMixture(), load_iris())
        for tg in range(min(9, TARGET_OPSET), TARGET_OPSET):
            with self.subTest(target_opset=tg):
                model_onnx = convert_sklearn(
                    model, "gaussian_mixture",
                    [("input", DoubleTensorType([None, X.shape[1]]))],
                    target_opset=tg)
                self.assertIsNotNone(model_onnx)
                dump_data_and_model(
                    X, model, model_onnx,
                    basename="SklearnBinGaussianMixtureDouble")
                self._test_score(model, X, tg)

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="onnxruntime misses Gemm for double")
    def test_model_bayesian_mixture_binary_classification(self):
        for cov in ["full", "tied", "diag", "spherical"]:
            with self.subTest(cov=cov):
                model, X = self._fit_model_binary_classification(
                    BayesianGaussianMixture(), load_iris(),
                    covariance_type=cov)
                model_onnx = convert_sklearn(
                    model, "gaussian_mixture",
                    [("input", DoubleTensorType([None, X.shape[1]]))],
                    target_opset=TARGET_OPSET)
                self.assertIsNotNone(model_onnx)
                dump_data_and_model(
                    X, model, model_onnx,
                    basename="SklearnBinBayesianGaussianMixtureDouble")
                self._test_score(model, X, TARGET_OPSET)

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="onnxruntime misses Gemm for double")
    def test_model_gaussian_mixture_multiclass(self):
        model, X = self._fit_model_multiclass_classification(
            GaussianMixture(), load_iris())
        model_onnx = convert_sklearn(
            model, "gaussian_mixture",
            [("input", DoubleTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnMclGaussianMixtureDouble")
        self._test_score(model, X, TARGET_OPSET)

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="onnxruntime misses Gemm for double")
    def test_gaussian_mixture_comp2(self):
        data = load_iris()
        X = data.data
        model = GaussianMixture(n_components=2)
        model.fit(X)
        model_onnx = convert_sklearn(
            model, "GM", [("input", DoubleTensorType([None, 4]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32)[40:60], model, model_onnx,
            basename="GaussianMixtureC2Double",
            intermediate_steps=False)
        self._test_score(model, X, TARGET_OPSET)

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="onnxruntime misses Gemm for double")
    def test_gaussian_mixture_full(self):
        data = load_iris()
        X = data.data
        model = GaussianMixture(n_components=2, covariance_type='full')
        model.fit(X)
        model_onnx = convert_sklearn(
            model, "GM", [("input", DoubleTensorType([None, 4]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32)[40:60], model, model_onnx,
            basename="GaussianMixtureC2FullDouble",
            intermediate_steps=False)
        self._test_score(model, X, TARGET_OPSET)

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="onnxruntime misses Gemm for double")
    def test_gaussian_mixture_tied(self):
        data = load_iris()
        X = data.data
        model = GaussianMixture(n_components=2, covariance_type='tied')
        model.fit(X)
        model_onnx = convert_sklearn(
            model, "GM", [("input", DoubleTensorType([None, 4]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32)[40:60],
            model, model_onnx, basename="GaussianMixtureC2TiedDouble",
            intermediate_steps=False)
        self._test_score(model, X, TARGET_OPSET)

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="onnxruntime misses Gemm for double")
    def test_gaussian_mixture_diag(self):
        data = load_iris()
        X = data.data
        model = GaussianMixture(n_components=2, covariance_type='diag')
        model.fit(X)
        model_onnx = convert_sklearn(
            model, "GM", [("input", DoubleTensorType([None, 4]))],
            target_opset=TARGET_OPSET)
        self.assertIn('ReduceLogSumExp', str(model_onnx))
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32)[40:60],
            model, model_onnx, basename="GaussianMixtureC2DiagDouble",
            intermediate_steps=False)
        self._test_score(model, X, TARGET_OPSET, decimal=4)

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="onnxruntime misses Gemm for double")
    def test_gaussian_mixture_spherical(self):
        data = load_iris()
        X = data.data
        model = GaussianMixture(n_components=2, covariance_type='spherical')
        model.fit(X)
        model_onnx = convert_sklearn(
            model, "GM", [("input", DoubleTensorType([None, 4]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X.astype(np.float32)[40:60],
            model, model_onnx, basename="GaussianMixtureC2SphericalDouble",
            intermediate_steps=False)
        self._test_score(model, X, TARGET_OPSET, decimal=4)

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="onnxruntime misses Gemm for double")
    def test_gaussian_mixture_full_black_op(self):
        data = load_iris()
        X = data.data
        model = GaussianMixture(n_components=2, covariance_type='full')
        model.fit(X)
        with self.assertRaises(RuntimeError):
            convert_sklearn(
                model, "GM", [("input", DoubleTensorType([None, 4]))],
                target_opset=TARGET_OPSET, black_op={'Add'})
        model_onnx = convert_sklearn(
            model, "GM", [("input", DoubleTensorType([None, 4]))],
            target_opset=TARGET_OPSET, black_op={'ReduceLogSumExp'})
        self.assertIsNotNone(model_onnx)
        self.assertNotIn('ReduceLogSumExp', str(model_onnx))
        dump_data_and_model(
            X.astype(np.float32)[40:60],
            model, model_onnx, basename="GaussianMixtureC2FullBLDouble",
            intermediate_steps=False)
        self._test_score(model, X, TARGET_OPSET)

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="onnxruntime misses Gemm for double")
    @unittest.skipIf(TARGET_OPSET < 11,
                     reason="OnnxEqual does not support float")
    def test_gaussian_mixture_full_black_op_noargmax(self):
        data = load_iris()
        X = data.data
        model = GaussianMixture(n_components=2, covariance_type='full')
        model.fit(X)
        with self.assertRaises(RuntimeError):
            convert_sklearn(
                model, "GM", [("input", DoubleTensorType([None, 4]))],
                target_opset=TARGET_OPSET, black_op={'Add'})
        model_onnx = convert_sklearn(
            model, "GM", [("input", DoubleTensorType([None, 4]))],
            target_opset=TARGET_OPSET,
            black_op={'ReduceLogSumExp', 'ArgMax'})
        self.assertIsNotNone(model_onnx)
        self.assertNotIn('ArgMax', str(model_onnx))
        dump_data_and_model(
            X.astype(np.float32)[40:60],
            model, model_onnx,
            basename="GaussianMixtureC2FullBLNMDouble",
            intermediate_steps=False)
        self._test_score(model, X, TARGET_OPSET)

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="onnxruntime misses Gemm for double")
    @unittest.skipIf(TARGET_OPSET < 11,
                     reason="OnnxEqual does not support float")
    def test_gaussian_mixture_full_black_op_noargmax_inf(self):
        data = load_iris()
        X = data.data
        model = GaussianMixture(n_components=10, covariance_type='full')
        model.fit(X)
        model_onnx1 = convert_sklearn(
            model, "GM", [("input", DoubleTensorType([None, 4]))],
            target_opset=TARGET_OPSET,
            options={id(model): {'score_samples': True}})
        model_onnx2 = convert_sklearn(
            model, "GM", [("input", DoubleTensorType([None, 4]))],
            target_opset=TARGET_OPSET,
            options={id(model): {'score_samples': True}},
            black_op={'ReduceLogSumExp', 'ArgMax'})
        self.assertNotIn('ArgMax', str(model_onnx2))

        sess1 = InferenceSession(model_onnx1.SerializeToString())
        res1 = sess1.run(None, {'input': (X[:5] * 1e2).astype(np.float32)})
        a1, b1, c1 = res1

        sess2 = InferenceSession(model_onnx2.SerializeToString())
        res2 = sess2.run(None, {'input': (X[:5] * 1e2).astype(np.float32)})
        a2, b2, c2 = res2

        self.assertEqual(b1.max(), b2.max())
        self.assertEqual(b1.min(), b2.min())
        self.assertLess(abs(c1.max() - c2.max()) / c2.max(), 1e-5)
        self.assertLess(abs(c1.min() - c2.min()) / c2.min(), 1e-5)

        self._test_score(
            model, X, TARGET_OPSET, black_op={'ReduceLogSumExp', 'ArgMax'},
            decimal=2)

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="onnxruntime misses Where for double")
    def test_binarizer(self):
        data = np.array([[1., -1., 2.],
                         [2., 0., 0.],
                         [0., 1., -1.]], dtype=np.float32)
        model = Binarizer(threshold=0.5)
        model_onnx = convert_sklearn(
            model, "scikit-learn binarizer",
            [("input", DoubleTensorType(data.shape))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data, model, model_onnx,
            basename="SklearnBinarizerDouble-SkipDim1")

    # DictVectorizer
    # FunctionTransformer
    # GaussianRandomProjection, IncrementalPCA, KBinsDiscretizer
    # KMeans, KNNImputer, KNeighborsTransformer,
    # LabelBinarizer, LabelEncoder, LinearDiscriminantAnalysis
    # NeighborhoodComponentsAnalysis, OneClassSVM
    # OneHotEncoder, OrdinalEncoder, OrthogonalMatchingPursuit
    # PCA, PolynomialFeatures, PowerTransformer, RFE, SelectFdr
    # SimpleImputer, TruncatedSVD, VarianceThreshold


if __name__ == "__main__":
    unittest.main()
