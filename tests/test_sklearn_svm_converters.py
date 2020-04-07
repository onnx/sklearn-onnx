"""
Tests scikit-linear converter.
"""
import unittest
from distutils.version import StrictVersion
import numpy
from sklearn.datasets import load_iris
from sklearn.svm import SVC, SVR, NuSVC, NuSVR, OneClassSVM
from sklearn import __version__ as sk__version__
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from skl2onnx.operator_converters.support_vector_machines import (
    convert_sklearn_svm)
from skl2onnx.shape_calculators.support_vector_machines import (
    calculate_sklearn_svm_output_shapes
)
import onnx
from onnxruntime import __version__ as ort_version
from test_utils import dump_data_and_model, fit_regression_model


class SVC_raw(SVC):

    def decision_function(self, X):
        return self._decision_function(X)

    def predict(self, X):
        p = self._dense_predict(X.astype(numpy.float64))
        return p


class NuSVC_raw(NuSVC):

    def decision_function(self, X):
        return self._decision_function(X)

    def predict(self, X):
        p = self._dense_predict(X.astype(numpy.float64))
        return p


class TestSklearnSVM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        update_registered_converter(
            SVC_raw,
            "SVC_raw",
            calculate_sklearn_svm_output_shapes,
            convert_sklearn_svm)
        update_registered_converter(
            NuSVC_raw,
            "NuSVC_raw",
            calculate_sklearn_svm_output_shapes,
            convert_sklearn_svm)

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
        return model, X[:5].astype(numpy.float32)

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
            SVC(kernel="linear", probability=False))
        model_onnx = convert_sklearn(
            model, "SVC", [("input", FloatTensorType([None, X.shape[1]]))])
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
            X,
            model,
            model_onnx,
            basename="SklearnBinSVCLinearPF-NoProbOpp",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " < StrictVersion('0.5.0')"
        )

    def test_convert_svc_binary_linear_ptrue(self):
        model, X = self._fit_binary_classification(
            SVC(kernel="linear", probability=True))
        model_onnx = convert_sklearn(
            model, "SVC", [("input",
                            FloatTensorType([None, X.shape[1]]))])
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
            X,
            model,
            model_onnx,
            basename="SklearnBinSVCLinearPT",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.4.0')"
        )

    def test_convert_svc_multi_linear_pfalse(self):
        model, X = self._fit_multi_classification(
            SVC_raw(kernel="linear", probability=False))
        model_onnx = convert_sklearn(
            model, "SVC", [("input", FloatTensorType([None, X.shape[1]]))])
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
            X,
            model,
            model_onnx,
            basename="SklearnMclSVCLinearPF-Dec4",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " < StrictVersion('0.5.0')"
        )

    def test_convert_svc_multi_linear_ptrue(self):
        model, X = self._fit_multi_classification(
            SVC_raw(kernel="linear", probability=False))
        model_onnx = convert_sklearn(
            model, "SVC", [("input", FloatTensorType([None, X.shape[1]]))])
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
            X,
            model,
            model_onnx,
            basename="SklearnMclSVCLinearPT-Dec4",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.4.0')"
        )

    def test_convert_svr_linear(self):
        model, X = self._fit_binary_classification(SVR(kernel="linear"))
        model_onnx = convert_sklearn(
            model, "SVR", [("input", FloatTensorType([None, X.shape[1]]))])
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
        dump_data_and_model(X,
                            model,
                            model_onnx,
                            basename="SklearnRegSVRLinear-Dec3")

    def test_convert_nusvc_binary_pfalse(self):
        model, X = self._fit_binary_classification(NuSVC(probability=False))
        model_onnx = convert_sklearn(
            model, "SVC", [("input", FloatTensorType([None, X.shape[1]]))])
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
            X,
            model,
            model_onnx,
            basename="SklearnBinNuSVCPF-NoProbOpp",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " < StrictVersion('0.5.0')"
        )

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion("0.4.0"),
        reason="use of recent Cast operator")
    def test_convert_nusvc_binary_ptrue(self):
        model, X = self._fit_binary_classification(NuSVC(probability=True))
        model_onnx = convert_sklearn(
            model, "SVC", [("input", FloatTensorType([None, X.shape[1]]))])
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
            X,
            model,
            model_onnx,
            basename="SklearnBinNuSVCPT",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.4.0')"
        )

    def test_convert_nusvc_multi_pfalse(self):
        model, X = self._fit_multi_classification(
            NuSVC_raw(probability=False, nu=0.1))
        model_onnx = convert_sklearn(
            model, "SVC", [("input", FloatTensorType([None, X.shape[1]]))])
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
            X,
            model,
            model_onnx,
            basename="SklearnMclNuSVCPF-Dec2",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " < StrictVersion('0.5.0')"
        )

    def test_convert_svc_multi_pfalse_4(self):
        model, X = self._fit_multi_classification(
            SVC_raw(probability=False), 4)
        model_onnx = convert_sklearn(
            model, "SVC", [("input", FloatTensorType([None, X.shape[1]]))])
        nodes = model_onnx.graph.node
        self.assertIsNotNone(nodes)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnMcSVCPF",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " < StrictVersion('0.5.0')"
        )

    @unittest.skipIf(
        StrictVersion(
            '.'.join(sk__version__.split('.')[:2]))
                < StrictVersion("0.22.0"),
        reason="break_ties introduced after 0.22.0")
    def test_convert_svc_multi_pfalse_4_break_ties(self):
        model, X = self._fit_multi_classification(
            SVC_raw(probability=False, break_ties=True), 4)
        model_onnx = convert_sklearn(
            model, "SVC_raw", [("input", FloatTensorType([None, X.shape[1]]))])
        nodes = model_onnx.graph.node
        self.assertIsNotNone(nodes)
        dump_data_and_model(
            X.astype(numpy.float32),
            model,
            model_onnx,
            basename="SklearnMcSVCPFBTF",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " < StrictVersion('0.5.0')"
        )

    def test_convert_svc_multi_ptrue_4(self):
        model, X = self._fit_multi_classification(SVC(probability=True), 4)
        model_onnx = convert_sklearn(
            model, "SVC", [("input", FloatTensorType([None, X.shape[1]]))])
        nodes = model_onnx.graph.node
        self.assertIsNotNone(nodes)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnMcSVCPF4",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.4.0')"
        )

    def test_convert_nusvc_multi_ptrue(self):
        model, X = self._fit_multi_classification(
            NuSVC(probability=True, nu=0.1))
        model_onnx = convert_sklearn(
            model, "SVC", [("input", FloatTensorType([None, X.shape[1]]))])
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
            X,
            model,
            model_onnx,
            basename="SklearnMclNuSVCPT",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.4.0')"
        )

    def test_convert_nusvr(self):
        model, X = self._fit_binary_classification(NuSVR())
        model_onnx = convert_sklearn(
            model, "SVR", [("input", FloatTensorType([None, X.shape[1]]))])
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

    def test_convert_nusvr_default(self):
        model, X = self._fit_binary_classification(NuSVR())
        model_onnx = convert_sklearn(
            model, "SVR", [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnRegNuSVR2")

    def test_convert_svr_int(self):
        model, X = fit_regression_model(
            SVR(), is_int=True)
        model_onnx = convert_sklearn(
            model,
            "SVR",
            [("input", Int64TensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnSVRInt-Dec4",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')"
        )

    def test_convert_nusvr_int(self):
        model, X = fit_regression_model(
            NuSVR(), is_int=True)
        model_onnx = convert_sklearn(
            model,
            "NuSVR",
            [("input", Int64TensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnNuSVRInt-Dec4",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')"
        )

    @unittest.skipIf(
        StrictVersion(onnx.__version__) < StrictVersion("1.4.1"),
        reason="operator sign available since opset 9")
    def test_convert_oneclasssvm(self):
        model, X = self._fit_one_class_svm(OneClassSVM())
        model_onnx = convert_sklearn(
            model, "OCSVM", [("input", FloatTensorType([None, X.shape[1]]))])
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnBinOneClassSVM",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " < StrictVersion('0.5.0')"
        )


if __name__ == "__main__":
    unittest.main()
