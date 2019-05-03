"""
Tests scikit-linear converter.
"""
import unittest
import numpy
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC, SVR, NuSVC, NuSVR
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
from test_utils import dump_data_and_model


class TestSklearnSVM(unittest.TestCase):
    def _fit_binary_classification(self, model):
        iris = load_iris()
        X = iris.data[:, :3]
        y = iris.target
        y[y == 2] = 1
        model.fit(X, y)
        return model, X[:5].astype(numpy.float32)

    def _fit_multi_classification(self, model):
        iris = load_iris()
        X = iris.data[:, :3]
        y = iris.target
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

    def test_convert_svmc_linear_binary(self):
        model, X = self._fit_binary_classification(
            SVC(kernel="linear", probability=False))
        model_onnx = convert_sklearn(
            model, "SVC", [("input", FloatTensorType([1, X.shape[1]]))])
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
            basename="SklearnBinSVCLinearPF",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.4')",
        )

    def test_convert_svmc_linear_multi(self):
        model, X = self._fit_multi_classification(
            SVC(kernel="linear", probability=False))
        model_onnx = convert_sklearn(
            model, "SVC", [("input", FloatTensorType([1, X.shape[1]]))])
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
            basename="SklearnMclSVCLinearPF",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.4')",
        )

    def test_convert_svmr_linear_binary(self):
        model, X = self._fit_binary_classification(SVR(kernel="linear"))
        model_onnx = convert_sklearn(
            model, "SVR", [("input", FloatTensorType([1, X.shape[1]]))])
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

    def test_convert_nusvmc_binary(self):
        model, X = self._fit_binary_classification(NuSVC(probability=False))
        model_onnx = convert_sklearn(
            model, "SVC", [("input", FloatTensorType([1, X.shape[1]]))])
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
            basename="SklearnBinNuSVCPF",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.4')",
        )

    def test_convert_nusvmc_multi(self):
        model, X = self._fit_multi_classification(NuSVC(probability=False))
        model_onnx = convert_sklearn(
            model, "SVC", [("input", FloatTensorType([1, X.shape[1]]))])
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
            basename="SklearnMclNuSVCPF",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.4')",
        )

    def test_convert_nusvmr_binary(self):
        model, X = self._fit_binary_classification(NuSVR())
        model_onnx = convert_sklearn(
            model, "SVR", [("input", FloatTensorType([1, X.shape[1]]))])
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
        dump_data_and_model(X, model, model_onnx, basename="SklearnRegNuSVR")

    def test_registration_convert_nusvr_model(self):
        model, X = self._fit_binary_classification(NuSVR())
        model_onnx = convert_sklearn(
            model, "SVR", [("input", FloatTensorType([1, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(X, model, model_onnx, basename="SklearnRegNuSVR2")

    def test_registration_convert_nusvc_model(self):
        model, X = self._fit_multi_classification(NuSVC(probability=True))
        model_onnx = convert_sklearn(
            model, "SVC", [("input", FloatTensorType([1, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnMclNuSVCPT",
            # Operator cast-1 is not implemented in onnxruntime
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2')",
        )

    def test_registration_convert_svc_model(self):
        model, X = self._fit_binary_classification(
            SVC(kernel="linear", probability=True))
        model_onnx = convert_sklearn(
            model, "SVC", [("input", FloatTensorType([1, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnBinNuSVCPT",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2')",
        )

    def testlinear_svc_iris(self):        
        data = load_iris()
        X = data.data
        y = data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42)

        clf = SGDClassifier().fit(X_train, y_train)
        model = CalibratedClassifierCV(clf, cv='prefit').fit(
            X_train, y_train)

        model_onnx = convert_sklearn(model, 'cccv',
                                     [('input',
                                       FloatTensorType(X_test.shape))])
        from onnxruntime import InferenceSession
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X_test.astype(numpy.float32)})
        numpy.mean(numpy.isclose(list(map(lambda x: [x[0], x[1],
                                                     x[2]], res[1])),
                                 model.predict_proba(X_test)))


if __name__ == "__main__":
    unittest.main()
