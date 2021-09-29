# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-linear converter.
"""
import unittest
from distutils.version import StrictVersion
import numpy
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession
from sklearn.datasets import load_iris
from sklearn.svm import SVC, SVR, NuSVC, NuSVR, OneClassSVM, LinearSVC
try:
    from skl2onnx.common._apply_operation import apply_less
except ImportError:
    # onnxconverter-common is too old
    apply_less = None
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    BooleanTensorType,
    FloatTensorType,
    Int64TensorType,
)
from skl2onnx.operator_converters.ada_boost import _scikit_learn_before_022
import onnx
from onnxruntime import __version__ as ort_version
from test_utils import (
    dump_data_and_model, fit_regression_model, TARGET_OPSET)


ort_version = ort_version.split('+')[0]


class TestSklearnSVM(unittest.TestCase):

    def _fit_binary_classification(self, model):
        iris = load_iris()
        X = iris.data[:, :3]
        y = iris.target
        y[y == 2] = 1
        model.fit(X, y)
        return model, X[:5].astype(numpy.float32)

    def _fit_one_class_svm(self, model):
        iris = load_iris()
        X = iris.data[:, :3]
        model.fit(X)
        return model, X[10:15].astype(numpy.float32)

    def _fit_multi_classification(self, model, nbclass=4):
        iris = load_iris()
        X = iris.data[:, :3]
        y = iris.target
        if nbclass == 4:
            y[-10:] = 3
        model.fit(X, y)
        X = numpy.vstack([X[:2], X[-3:]])
        return model, X.astype(numpy.float32)

    def _fit_multi_regression(self, model):
        iris = load_iris()
        X = iris.data[:, :3]
        y = numpy.vstack([iris.target, iris.target]).T
        model.fit(X, y)
        return model, X[:5].astype(numpy.float32)

    def _check_attributes(self, node, attribute_test):
        attributes = node.attribute
        attribute_map = {}
        for attribute in attributes:
            attribute_map[attribute.name] = attribute

        for k, v in attribute_test.items():
            self.assertTrue(k in attribute_map)
            if v is not None:
                attrib = attribute_map[k]
                if isinstance(v, str):
                    self.assertEqual(attrib.s, v.encode(encoding="UTF-8"))
                elif isinstance(v, int):
                    self.assertEqual(attrib.i, v)
                elif isinstance(v, float):
                    self.assertEqual(attrib.f, v)
                elif isinstance(v, list):
                    self.assertEqual(attrib.ints, v)
                else:
                    self.fail("Unknown type")

    def test_convert_svc_binary_linear_pfalse(self):
        model, X = self._fit_binary_classification(
            SVC(kernel="linear", probability=False,
                decision_function_shape='ovo'))

        model_onnx = convert_sklearn(
            model, "SVC", [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        nodes = model_onnx.graph.node
        self.assertIsNotNone(nodes)
        svc_node = nodes[0]
        self._check_attributes(
            svc_node,
            {
                "coefficients": None,
                "kernel_params": None,
                "kernel_type": "LINEAR",
                "post_transform": None,
                "rho": None,
                "support_vectors": None,
                "vectors_per_class": None,
            },
        )
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnBinSVCLinearPF-NoProbOpp")
        model_onnx = convert_sklearn(
            model, "SVC", [("input", FloatTensorType([None, X.shape[1]]))],
            options={id(model): {'zipmap': False}},
            target_opset=TARGET_OPSET)
        nodes = model_onnx.graph.node
        self.assertIsNotNone(nodes)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnBinSVCLinearPF-NoProbOpp")

    def test_convert_svc_binary_linear_ptrue(self):
        model, X = self._fit_binary_classification(
            SVC(kernel="linear", probability=True))

        model_onnx = convert_sklearn(
            model, "SVC", [("input",
                            FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        nodes = model_onnx.graph.node
        self.assertIsNotNone(nodes)
        svc_node = nodes[0]
        self._check_attributes(
            svc_node,
            {
                "coefficients": None,
                "kernel_params": None,
                "kernel_type": "LINEAR",
                "post_transform": None,
                "rho": None,
                "support_vectors": None,
                "vectors_per_class": None,
            },
        )
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnBinSVCLinearPT")

    def test_convert_svc_multi_linear_pfalse(self):
        model, X = self._fit_multi_classification(
            SVC(kernel="linear", probability=False,
                decision_function_shape="ovo"))
        model_onnx = convert_sklearn(
            model, "SVC", [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)

        nodes = model_onnx.graph.node
        self.assertIsNotNone(nodes)
        svc_node = nodes[0]
        self._check_attributes(
            svc_node, {
                "coefficients": None, "kernel_params": None,
                "kernel_type": "LINEAR", "post_transform": None,
                "rho": None, "support_vectors": None,
                "vectors_per_class": None})

        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnMclSVCLinearPF-Dec4")

    @unittest.skipIf(apply_less is None, reason="onnxconverter-common old")
    def test_convert_svc_multi_linear_pfalse_ovr(self):
        model, X = self._fit_multi_classification(
            SVC(kernel="linear", probability=False,
                decision_function_shape='ovr'))
        model_onnx = convert_sklearn(
            model, "SVC", [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnMclSVCOVR-Dec4")

    def test_convert_svc_multi_linear_ptrue(self):
        model, X = self._fit_multi_classification(
            SVC(kernel="linear", probability=True),
            nbclass=3)
        model_onnx = convert_sklearn(
            model, "SVC", [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        nodes = model_onnx.graph.node
        self.assertIsNotNone(nodes)
        svc_node = nodes[0]
        self._check_attributes(
            svc_node, {
                "coefficients": None, "kernel_params": None,
                "kernel_type": "LINEAR", "post_transform": None,
                "rho": None, "support_vectors": None,
                "vectors_per_class": None})
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnMclSVCLinearPT-Dec2")

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion("0.4.0"),
        reason="use of recent Cast operator")
    def test_convert_svr_linear(self):
        model, X = self._fit_binary_classification(SVR(kernel="linear"))
        model_onnx = convert_sklearn(
            model, "SVR", [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        nodes = model_onnx.graph.node
        self.assertIsNotNone(nodes)
        self._check_attributes(
            nodes[0],
            {
                "coefficients": None,
                "kernel_params": None,
                "kernel_type": "LINEAR",
                "post_transform": None,
                "rho": None,
                "support_vectors": None,
            },
        )
        dump_data_and_model(X, model, model_onnx,
                            basename="SklearnRegSVRLinear-Dec3")

    def test_convert_nusvc_binary_pfalse(self):
        model, X = self._fit_binary_classification(
            NuSVC(probability=False, decision_function_shape='ovo'))
        model_onnx = convert_sklearn(
            model, "SVC", [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        nodes = model_onnx.graph.node
        self.assertIsNotNone(nodes)
        svc_node = nodes[0]
        self._check_attributes(
            svc_node,
            {
                "coefficients": None,
                "kernel_params": None,
                "kernel_type": "RBF",
                "post_transform": None,
                "rho": None,
                "support_vectors": None,
                "vectors_per_class": None,
            },
        )
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnBinNuSVCPF-NoProbOpp")

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion("0.4.0"),
        reason="use of recent Cast operator")
    def test_convert_nusvc_binary_ptrue(self):
        model, X = self._fit_binary_classification(NuSVC(probability=True))
        model_onnx = convert_sklearn(
            model, "SVC", [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        nodes = model_onnx.graph.node
        self.assertIsNotNone(nodes)
        svc_node = nodes[0]
        self._check_attributes(
            svc_node,
            {
                "coefficients": None,
                "kernel_params": None,
                "kernel_type": "RBF",
                "post_transform": None,
                "rho": None,
                "support_vectors": None,
                "vectors_per_class": None,
            },
        )
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnBinNuSVCPT")

    def test_convert_nusvc_multi_pfalse(self):
        model, X = self._fit_multi_classification(
            NuSVC(probability=False, nu=0.1,
                  decision_function_shape='ovo'))
        model_onnx = convert_sklearn(
            model, "SVC", [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        nodes = model_onnx.graph.node
        self.assertIsNotNone(nodes)
        svc_node = nodes[0]
        self._check_attributes(
            svc_node,
            {
                "coefficients": None,
                "kernel_params": None,
                "kernel_type": "RBF",
                "post_transform": None,
                "rho": None,
                "support_vectors": None,
                "vectors_per_class": None,
            },
        )
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnMclNuSVCPF-Dec1")

    def test_convert_svc_multi_pfalse_4(self):
        model, X = self._fit_multi_classification(
            SVC(probability=False,
                decision_function_shape='ovo'), 4)
        model_onnx = convert_sklearn(
            model, "SVC", [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        nodes = model_onnx.graph.node
        self.assertIsNotNone(nodes)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnMcSVCPF")

    @unittest.skipIf(_scikit_learn_before_022(),
                     reason="break_ties introduced after 0.22")
    def test_convert_svc_multi_pfalse_4_break_ties(self):
        model, X = self._fit_multi_classification(
            SVC(probability=True, break_ties=True), 4)
        model_onnx = convert_sklearn(
            model, "unused", [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        nodes = model_onnx.graph.node
        self.assertIsNotNone(nodes)
        dump_data_and_model(
            X.astype(numpy.float32),
            model, model_onnx,
            basename="SklearnMcSVCPFBTF-Dec4")

    def test_convert_svc_multi_ptrue_4(self):
        model, X = self._fit_multi_classification(SVC(probability=True), 4)
        model_onnx = convert_sklearn(
            model, "SVC", [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        nodes = model_onnx.graph.node
        self.assertIsNotNone(nodes)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnMcSVCPF4-Dec4")

    def test_convert_nusvc_multi_ptrue(self):
        model, X = self._fit_multi_classification(
            NuSVC(probability=True, nu=0.1))
        model_onnx = convert_sklearn(
            model, "SVC", [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        nodes = model_onnx.graph.node
        self.assertIsNotNone(nodes)
        svc_node = nodes[0]
        self._check_attributes(
            svc_node,
            {
                "coefficients": None,
                "kernel_params": None,
                "kernel_type": "RBF",
                "post_transform": None,
                "rho": None,
                "support_vectors": None,
                "vectors_per_class": None,
            },
        )
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnMclNuSVCPT-Dec3")

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion("0.4.0"),
        reason="use of recent Cast operator")
    def test_convert_nusvr(self):
        model, X = self._fit_binary_classification(NuSVR())
        model_onnx = convert_sklearn(
            model, "SVR", [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        node = model_onnx.graph.node[0]
        self.assertIsNotNone(node)
        self._check_attributes(
            node,
            {
                "coefficients": None,
                "kernel_params": None,
                "kernel_type": "RBF",
                "post_transform": None,
                "rho": None,
                "support_vectors": None,
            },
        )
        dump_data_and_model(X, model, model_onnx,
                            basename="SklearnRegNuSVR")

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion("0.4.0"),
        reason="use of recent Cast operator")
    def test_convert_nusvr_default(self):
        model, X = self._fit_binary_classification(NuSVR())
        model_onnx = convert_sklearn(
            model, "SVR", [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnRegNuSVR2")

    def test_convert_svr_int(self):
        model, X = fit_regression_model(
            SVR(), is_int=True)
        model_onnx = convert_sklearn(
            model, "SVR",
            [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnSVRInt-Dec4")

    def test_convert_nusvr_int(self):
        model, X = fit_regression_model(
            NuSVR(), is_int=True)
        model_onnx = convert_sklearn(
            model, "NuSVR", [("input", Int64TensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnNuSVRInt-Dec4")

    def test_convert_svr_bool(self):
        model, X = fit_regression_model(
            SVR(), is_bool=True)
        model_onnx = convert_sklearn(
            model, "SVR",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnSVRBool-Dec4")

    def test_convert_nusvr_bool(self):
        model, X = fit_regression_model(
            NuSVR(), is_bool=True)
        model_onnx = convert_sklearn(
            model, "NuSVR",
            [("input", BooleanTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnNuSVRBool")

    @unittest.skipIf(
        StrictVersion(onnx.__version__) < StrictVersion("1.4.1"),
        reason="operator sign available since opset 9")
    def test_convert_oneclasssvm(self):
        model, X = self._fit_one_class_svm(OneClassSVM())
        model_onnx = convert_sklearn(
            model, "OCSVM", [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnBinOneClassSVM")

    def test_model_linear_svc_binary_class(self):
        model, X = self._fit_binary_classification(LinearSVC(max_iter=10000))
        model_onnx = convert_sklearn(
            model, "linear SVC",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X})
        label = model.predict(X)
        proba = model.decision_function(X)
        assert_almost_equal(proba, res[1].ravel(), decimal=5)
        assert_almost_equal(label, res[0])

    def test_model_linear_svc_multi_class(self):
        model, X = self._fit_multi_classification(LinearSVC(max_iter=10000))
        model_onnx = convert_sklearn(
            model, "linear SVC",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X})
        label = model.predict(X)
        proba = model.decision_function(X)
        assert_almost_equal(proba, res[1], decimal=5)
        assert_almost_equal(label, res[0])

    def test_model_svc_binary_class_false(self):
        model, X = self._fit_binary_classification(SVC(max_iter=10000))
        model_onnx = convert_sklearn(
            model, "linear SVC",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X})
        label = model.predict(X)
        proba = model.decision_function(X)
        assert_almost_equal(proba, res[1][:, 0], decimal=5)
        assert_almost_equal(label, res[0])

    @unittest.skipIf(TARGET_OPSET < 12, reason="operator Less")
    def test_model_svc_multi_class_false(self):
        model, X = self._fit_multi_classification(SVC(max_iter=10000))
        model_onnx = convert_sklearn(
            model, "linear SVC",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X})
        label = model.predict(X)
        proba = model.decision_function(X)
        assert_almost_equal(proba, res[1], decimal=5)
        assert_almost_equal(label, res[0])

    def test_model_svc_binary_class_true(self):
        model, X = self._fit_binary_classification(
            SVC(max_iter=10000, probability=True))
        model_onnx = convert_sklearn(
            model, "linear SVC",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={'zipmap': False}, target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X})
        label = model.predict(X)
        proba = model.predict_proba(X)
        assert_almost_equal(proba, res[1], decimal=5)
        assert_almost_equal(label, res[0])

    def test_model_svc_multi_class_true(self):
        model, X = self._fit_multi_classification(
            SVC(max_iter=10000, probability=True))
        model_onnx = convert_sklearn(
            model, "linear SVC",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={'zipmap': False}, target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X})
        label = model.predict(X)
        proba = model.predict_proba(X)
        assert_almost_equal(proba, res[1], decimal=5)
        assert_almost_equal(label, res[0])

    def test_model_nusvc_binary_class_false(self):
        model, X = self._fit_binary_classification(NuSVC(max_iter=10000))
        model_onnx = convert_sklearn(
            model, "linear SVC",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X})
        label = model.predict(X)
        proba = model.decision_function(X)
        assert_almost_equal(proba, res[1][:, 0], decimal=5)
        assert_almost_equal(label, res[0])

    @unittest.skipIf(TARGET_OPSET < 12, reason="operator Less")
    def test_model_nusvc_multi_class_false(self):
        model, X = self._fit_multi_classification(
            NuSVC(max_iter=10000, nu=0.1))
        model_onnx = convert_sklearn(
            model, "linear SVC",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X})
        label = model.predict(X)
        proba = model.decision_function(X)
        assert_almost_equal(proba, res[1], decimal=4)
        assert_almost_equal(label, res[0])

    def test_model_nusvc_binary_class_true(self):
        model, X = self._fit_binary_classification(
            NuSVC(max_iter=10000, probability=True))
        model_onnx = convert_sklearn(
            model, "linear SVC",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={'zipmap': False}, target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X})
        label = model.predict(X)
        proba = model.predict_proba(X)
        assert_almost_equal(proba, res[1], decimal=5)
        assert_almost_equal(label, res[0])

    def test_model_nusvc_multi_class_true(self):
        model, X = self._fit_multi_classification(
            NuSVC(max_iter=10000, probability=True, nu=0.1))
        model_onnx = convert_sklearn(
            model, "linear SVC",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={'zipmap': False}, target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X})
        label = model.predict(X)
        proba = model.predict_proba(X)
        assert_almost_equal(proba, res[1], decimal=3)
        assert_almost_equal(label, res[0])


if __name__ == "__main__":
    unittest.main()
