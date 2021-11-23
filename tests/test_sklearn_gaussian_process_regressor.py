# SPDX-License-Identifier: Apache-2.0


import unittest
import inspect
import warnings
from io import StringIO
from distutils.version import StrictVersion
import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris, make_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Sum, DotProduct, ExpSineSquared, RationalQuadratic,
    RBF, ConstantKernel as C, PairwiseKernel)
from sklearn.model_selection import train_test_split
try:
    # scikit-learn >= 0.22
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    # scikit-learn < 0.22
    from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType
from skl2onnx import to_onnx
from skl2onnx.proto import get_latest_tested_opset_version
from skl2onnx.operator_converters.gaussian_process import (
    convert_kernel, convert_kernel_diag
)
from onnxruntime import InferenceSession, SessionOptions
try:
    from onnxruntime import GraphOptimizationLevel
except ImportError:
    GraphOptimizationLevel = None
try:
    from onnxruntime.capi.onnxruntime_pybind11_state import NotImplemented
except ImportError:
    NotImplemented = RuntimeError
from onnxruntime import __version__ as ort_version
from test_utils import dump_data_and_model, fit_regression_model, TARGET_OPSET

_TARGET_OPSET_ = min(get_latest_tested_opset_version(), TARGET_OPSET)
ort_version = ".".join(ort_version.split('.')[:2])


Xtrain_ = pd.read_csv(StringIO("""
1.000000000000000000e+02,1.158972369426435591e+02,5.667579938823991137e-01,2.264397682069040421e-02,1.182166076334919581e-02,2.600819340784729095e-01
1.000000000000000000e+02,8.493978168996618194e+01,2.775702708579337874e-01,1.887456201351307358e-02,2.912599235354124821e-02,2.327206144705836199e-01
1.000000000000000000e+02,8.395765637241281354e+01,7.760226193410907358e-01,2.139558949508506974e-02,1.944769253403489523e-02,5.462612465817335838e-01
1.000000000000000000e+02,1.251224039142802411e+02,1.085922727328213266e+00,1.650449428041057126e-02,2.006508371199252141e-02,3.925044939686896939e-01
1.000000000000000000e+02,7.292655293041464404e+01,1.310113459857209950e+00,2.422656953481223258e-02,3.328909433367271964e-02,4.321979372794531593e-01
1.000000000000000000e+02,1.002649729946309094e+02,1.105327461462607630e+00,2.148827969317553335e-02,3.148001380372193736e-02,1.684894130082370545e-01
1.000000000000000000e+02,9.628657457451673451e+01,3.460979367851939603e-01,1.538570748635538499e-02,3.597376501128631693e-02,5.345963757636325031e-01
1.000000000000000000e+02,8.121250906502669409e+01,1.865077048426986073e+00,2.182149790268794742e-02,4.300530595437276893e-02,5.083327963416256479e-01
1.000000000000000000e+02,8.612638714481262525e+01,2.717895097207565502e-01,2.029318789405683970e-02,2.387016690377936207e-02,1.889736980423707968e-01
1.000000000000000000e+02,7.377491009582655579e+01,7.210994150180145557e-01,2.239484250704669444e-02,1.642684033674572316e-02,4.341188586319142395e-01
""".strip("\n\r ")), header=None).values

Xtest_ = pd.read_csv(StringIO("""
1.000000000000000000e+02,1.061277971307766705e+02,1.472195004809226493e+00,2.307125069497626552e-02,4.539948095743629591e-02,2.855191098141335870e-01
1.000000000000000000e+02,9.417031896832908444e+01,1.249743892709246573e+00,2.370416174339620707e-02,2.613847280316268853e-02,5.097165413593484073e-01
1.000000000000000000e+02,9.305231488674536422e+01,1.795726729335217264e+00,2.473274733802270642e-02,1.349765645107412620e-02,9.410288840541443378e-02
1.000000000000000000e+02,7.411264142156210255e+01,1.747723020195752319e+00,1.559695663417645997e-02,4.230394035515055301e-02,2.225492746314280956e-01
1.000000000000000000e+02,9.326006195761877393e+01,1.738860294343326229e+00,2.280160135767652502e-02,4.883335335161764074e-02,2.806808409247734115e-01
1.000000000000000000e+02,8.341529291866362428e+01,5.119682123742423929e-01,2.488795768635816003e-02,4.887573336092913834e-02,1.673462179673477768e-01
1.000000000000000000e+02,1.182436477919874562e+02,1.733516391831658954e+00,1.533520930349476820e-02,3.131213519485807895e-02,1.955345358785769427e-01
1.000000000000000000e+02,1.228982583299257101e+02,1.115599996405831629e+00,1.929354155079938959e-02,3.056996308544096715e-03,1.197052763998271013e-01
1.000000000000000000e+02,1.160303269386108838e+02,1.018627021014927303e+00,2.248784981616459844e-02,2.688111547114307651e-02,3.326105131778724355e-01
1.000000000000000000e+02,1.163414374640396005e+02,6.644299545804077667e-01,1.508088417713602906e-02,4.451836657613789106e-02,3.245643044204808425e-01
""".strip("\n\r ")), header=None).values

Ytrain_ = pd.read_csv(StringIO("""
1.810324564191880370e+01
4.686462914930641377e-01
1.032271142638131778e+01
3.308144139528823047e+01
6.525165063871320115e+00
7.501105337335136625e+00
1.047000553596901895e+01
1.652864171243088975e+01
2.491797751537555006e-01
3.413210402096089169e+00
""".strip("\n\r ")), header=None).values

Ytest_ = pd.read_csv(StringIO("""
1.836586066727948463e+01
1.848708258852349573e+01
1.641115566770171341e+00
2.555927439688699288e+00
1.216079754835943660e+01
3.545972261849191787e-01
2.385075724064493130e+01
2.289825832992571009e+01
2.353204496952379898e+01
2.237280571788585348e+01
""".strip("\n\r ")), header=None).values


THRESHOLD = "0.4.0"
THRESHOLD2 = "0.5.0"


class TestSklearnGaussianProcessRegressor(unittest.TestCase):

    def remove_dim1(self, arr):
        new_shape = tuple(v for v in arr.shape if v != 1)
        if new_shape != arr.shape:
            arr = arr.reshape(new_shape)
        return arr

    def check_outputs(self, model, model_onnx, Xtest,
                      predict_attributes, decimal=5,
                      skip_if_float32=False, disable_optimisation=True):
        if "TransposeScaleMatMul" in str(model_onnx):
            raise RuntimeError("This node must not be added.")
        if predict_attributes is None:
            predict_attributes = {}
        exp = model.predict(Xtest, **predict_attributes)
        if disable_optimisation and GraphOptimizationLevel is not None:
            opts = SessionOptions()
            opts.graph_optimization_level = (
                GraphOptimizationLevel.ORT_DISABLE_ALL)
            sess = InferenceSession(
                model_onnx.SerializeToString(), sess_options=opts)
        else:
            sess = InferenceSession(model_onnx.SerializeToString())
        got = sess.run(None, {'X': Xtest})
        if isinstance(exp, tuple):
            if len(exp) != len(got):
                raise AssertionError("Mismatched number of outputs.")
            for i, (e, g) in enumerate(zip(exp, got)):
                if skip_if_float32 and g.dtype == np.float32:
                    continue
                try:
                    assert_almost_equal(self.remove_dim1(e),
                                        self.remove_dim1(g),
                                        decimal=decimal)
                except AssertionError as e:  # noqa
                    raise AssertionError(
                        "Mismatch for output {} and attributes {}"
                        ".".format(i, predict_attributes)) from e
        else:
            if skip_if_float32 and Xtest.dtype == np.float32:
                return
            assert_almost_equal(np.squeeze(exp),
                                np.squeeze(got), decimal=decimal)

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_kernel_constant1(self):
        ker = C(5.)
        onx = convert_kernel(ker, 'X', output_names=['Y'], dtype=np.float32,
                             op_version=_TARGET_OPSET_)
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType([None, None]))],
            target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'X': Xtest_.astype(np.float32)})[0]
        m1 = res
        m2 = ker(Xtest_)
        assert_almost_equal(m1, m2, decimal=5)

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_kernel_cosine_float(self):
        ker = PairwiseKernel(metric='cosine')

        # X, X
        onx = convert_kernel(ker, 'X', output_names=['Y'], dtype=np.float32,
                             op_version=_TARGET_OPSET_)
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType([None, None]))],
            target_opset=TARGET_OPSET)

        x = np.random.randn(4, 3)
        x[0, 0] = x[1, 1] = x[2, 2] = 10.
        x[3, 2] = 5.

        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'X': x.astype(np.float32)})[0]
        m1 = res
        m2 = ker(x)
        assert_almost_equal(m1, m2, decimal=5)

        # X, x
        onx = convert_kernel(ker, 'X', x_train=x,
                             output_names=['Y'], dtype=np.float32,
                             op_version=_TARGET_OPSET_)
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType([None, None]))],
            target_opset=TARGET_OPSET)

        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'X': x.astype(np.float32)})[0]
        m1 = res
        m2 = ker(x)
        assert_almost_equal(m1, m2, decimal=5)

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_kernel_cosine_double(self):
        ker = PairwiseKernel(metric='cosine')
        onx = convert_kernel(ker, 'X', output_names=['Y'], dtype=np.float64,
                             op_version=_TARGET_OPSET_)
        model_onnx = onx.to_onnx(
            inputs=[('X', DoubleTensorType([None, None]))],
            target_opset=TARGET_OPSET)

        x = np.random.randn(4, 3)
        x[0, 0] = x[1, 1] = x[2, 2] = 10.
        x[3, 2] = 5.

        try:
            sess = InferenceSession(model_onnx.SerializeToString())
        except NotImplemented:
            # Failed to find kernel for FusedMatMul(1).
            return
        res = sess.run(None, {'X': x.astype(np.float64)})[0]
        m1 = res
        m2 = ker(x)
        assert_almost_equal(m1, m2, decimal=5)

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_kernel_rbf1(self):
        ker = RBF(length_scale=1, length_scale_bounds=(1e-3, 1e3))
        onx = convert_kernel(ker, 'X', output_names=['Y'], dtype=np.float32,
                             op_version=_TARGET_OPSET_)
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType([None, None]))])
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'X': Xtest_.astype(np.float32)})[0]
        m1 = res
        m2 = ker(Xtest_)
        assert_almost_equal(m1, m2, decimal=5)

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_kernel_rbf10(self):
        ker = RBF(length_scale=10, length_scale_bounds=(1e-3, 1e3))
        onx = convert_kernel(ker, 'X', output_names=['Y'], dtype=np.float32,
                             op_version=_TARGET_OPSET_)
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType([None, None]))])
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'X': Xtest_.astype(np.float32)})[0]
        m1 = res
        m2 = ker(Xtest_)
        assert_almost_equal(m1, m2, decimal=5)

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_kernel_rbf2(self):
        ker = RBF(length_scale=1, length_scale_bounds="fixed")
        onx = convert_kernel(ker, 'X', output_names=['Y'], dtype=np.float32,
                             op_version=_TARGET_OPSET_)
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType([None, None]))])
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'X': Xtest_.astype(np.float32)})[0]
        m1 = res
        m2 = ker(Xtest_)
        assert_almost_equal(m1, m2, decimal=5)

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_kernel_rbf_mul(self):
        ker = (C(1.0, constant_value_bounds="fixed") *
               RBF(1.0, length_scale_bounds="fixed"))
        onx = convert_kernel(ker, 'X', output_names=['Y'], dtype=np.float32,
                             op_version=_TARGET_OPSET_)
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType([None, None]))],
            target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'X': Xtest_.astype(np.float32)})[0]
        m1 = res
        m2 = ker(Xtest_)
        assert_almost_equal(m1, m2, decimal=5)

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_kernel_ker1_def(self):
        ker = (C(1.0, (1e-3, 1e3)) *
               RBF(length_scale=10, length_scale_bounds=(1e-3, 1e3)))
        onx = convert_kernel(ker, 'X', output_names=['Y'], dtype=np.float32,
                             op_version=_TARGET_OPSET_)
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType([None, None]))],
            target_opset=TARGET_OPSET)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'X': Xtest_.astype(np.float32)})[0]
        m1 = res
        m2 = ker(Xtest_)
        assert_almost_equal(m1, m2, decimal=5)

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_kernel_ker12_def(self):
        ker = (Sum(C(0.1, (1e-3, 1e3)), C(0.1, (1e-3, 1e3)) *
               RBF(length_scale=1, length_scale_bounds=(1e-3, 1e3))))
        onx = convert_kernel(ker, 'X', output_names=['Y'], dtype=np.float32,
                             op_version=_TARGET_OPSET_)
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType([None, None]))],
            target_opset=_TARGET_OPSET_)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'X': Xtest_.astype(np.float32)})[0]
        m1 = res
        m2 = ker(Xtest_)
        assert_almost_equal(m1, m2, decimal=5)

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_kernel_ker2_def(self):
        ker = Sum(
            C(0.1, (1e-3, 1e3)) * RBF(length_scale=10,
                                      length_scale_bounds=(1e-3, 1e3)),
            C(0.1, (1e-3, 1e3)) * RBF(length_scale=1,
                                      length_scale_bounds=(1e-3, 1e3))
        )
        onx = convert_kernel(ker, 'X', output_names=['Y'], dtype=np.float32,
                             op_version=_TARGET_OPSET_)
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType([None, None]))],
            target_opset=_TARGET_OPSET_)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'X': Xtest_.astype(np.float32)})[0]
        m1 = res
        m2 = ker(Xtest_)
        assert_almost_equal(m1, m2, decimal=0)

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_kernel_ker2_dotproduct(self):
        ker = DotProduct(sigma_0=2.)
        onx = convert_kernel(ker, 'X', output_names=['Y'], dtype=np.float32,
                             op_version=_TARGET_OPSET_)
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType())],
            outputs=[('Y', FloatTensorType())],
            target_opset=_TARGET_OPSET_)
        sess = InferenceSession(model_onnx.SerializeToString())

        x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        res = sess.run(None, {'X': x})
        m1 = res[0]
        m2 = ker(x)
        assert_almost_equal(m1, m2, decimal=5)

        res = sess.run(None, {'X': Xtest_.astype(np.float32)})
        m1 = res[0]
        m2 = ker(Xtest_)
        assert_almost_equal(m1, m2, decimal=2)

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_kernel_ker2_exp_sine_squared(self):
        ker = ExpSineSquared()
        onx = convert_kernel(ker, 'X', output_names=['Y'], dtype=np.float32,
                             op_version=_TARGET_OPSET_)
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType([None, None]))],
            target_opset=_TARGET_OPSET_)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'X': Xtest_.astype(np.float32)})[0]
        m1 = res
        m2 = ker(Xtest_)
        assert_almost_equal(m1, m2, decimal=4)

        onx = convert_kernel(ker, 'X', output_names=['Z'],
                             x_train=(Xtest_ * 2).astype(np.float32),
                             dtype=np.float32, op_version=_TARGET_OPSET_)
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType([None, None]))],
            target_opset=_TARGET_OPSET_)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'X': Xtest_.astype(np.float32)})[0]
        m1 = res
        m2 = ker(Xtest_, Xtest_ * 2)
        assert_almost_equal(m1, m2, decimal=4)

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_kernel_exp_sine_squared_diag(self):
        ker = ExpSineSquared()
        onx = convert_kernel_diag(
            ker, 'X', output_names=['Y'], dtype=np.float32,
            op_version=_TARGET_OPSET_)
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType([None, None]))],
            target_opset=_TARGET_OPSET_)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'X': Xtest_.astype(np.float32)})[0]
        m1 = res
        m2 = ker.diag(Xtest_)
        assert_almost_equal(m1, m2, decimal=4)

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_kernel_rational_quadratic_diag(self):
        ker = RationalQuadratic()
        onx = convert_kernel_diag(
            ker, 'X', output_names=['Y'], dtype=np.float32,
            op_version=_TARGET_OPSET_)
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType([None, None]))],
            target_opset=_TARGET_OPSET_)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'X': Xtest_.astype(np.float32)})[0]
        m1 = res
        m2 = ker.diag(Xtest_)
        assert_almost_equal(m1, m2, decimal=4)

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_kernel_dot_product_diag(self):
        ker = DotProduct()
        onx = convert_kernel_diag(
            ker, 'X', output_names=['Y'], dtype=np.float32,
            op_version=_TARGET_OPSET_)
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType([None, None]))],
            target_opset=_TARGET_OPSET_)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'X': Xtest_.astype(np.float32)})[0]
        m1 = res
        m2 = ker.diag(Xtest_)
        assert_almost_equal(m1 / 1000, m2 / 1000, decimal=5)

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_kernel_dot_product(self):
        ker = DotProduct()
        onx = convert_kernel(ker, 'X', output_names=['Y'], dtype=np.float32,
                             op_version=_TARGET_OPSET_)
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType([None, None]))],
            target_opset=_TARGET_OPSET_)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'X': Xtest_.astype(np.float32)})[0]
        m1 = res
        m2 = ker(Xtest_)
        assert_almost_equal(m1 / 1000, m2 / 1000, decimal=5)

        onx = convert_kernel(ker, 'X', output_names=['Z'],
                             x_train=(Xtest_ * 2).astype(np.float32),
                             dtype=np.float32, op_version=_TARGET_OPSET_)
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType([None, None]))],
            target_opset=_TARGET_OPSET_)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'X': Xtest_.astype(np.float32)})[0]
        m1 = res
        m2 = ker(Xtest_, Xtest_ * 2)
        assert_almost_equal(m1 / 1000, m2 / 1000, decimal=5)

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_kernel_rational_quadratic(self):
        ker = RationalQuadratic()
        onx = convert_kernel(ker, 'X', output_names=['Y'], dtype=np.float32,
                             op_version=_TARGET_OPSET_)
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType([None, None]))],
            target_opset=_TARGET_OPSET_)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'X': Xtest_.astype(np.float32)})[0]
        m1 = res
        m2 = ker(Xtest_)
        assert_almost_equal(m1, m2, decimal=5)

        onx = convert_kernel(ker, 'X', output_names=['Z'],
                             x_train=(Xtest_ * 2).astype(np.float32),
                             dtype=np.float32, op_version=_TARGET_OPSET_)
        model_onnx = onx.to_onnx(
            inputs=[('X', FloatTensorType([None, None]))])
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'X': Xtest_.astype(np.float32)})[0]
        m1 = res
        m2 = ker(Xtest_, Xtest_ * 2)
        assert_almost_equal(m1, m2, decimal=3)

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_gpr_rbf_unfitted(self):

        se = (C(1.0, (1e-3, 1e3)) *
              RBF(length_scale=10, length_scale_bounds=(1e-3, 1e3)))
        kernel = (Sum(se, C(0.1, (1e-3, 1e3)) *
                  RBF(length_scale=1, length_scale_bounds=(1e-3, 1e3))))

        gp = GaussianProcessRegressor(alpha=1e-5, kernel=kernel,
                                      n_restarts_optimizer=25,
                                      normalize_y=True)

        # return_cov=False, return_std=False
        model_onnx = to_onnx(
            gp, initial_types=[('X', FloatTensorType([]))],
            target_opset=_TARGET_OPSET_)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(Xtest_.astype(np.float32), gp, model_onnx,
                            verbose=False,
                            basename="SklearnGaussianProcessRBFUnfitted")

        # return_cov=True, return_std=True
        options = {GaussianProcessRegressor: {"return_std": True,
                                              "return_cov": True}}
        try:
            to_onnx(gp, Xtrain_.astype(np.float32), options=options,
                    target_opset=TARGET_OPSET)
        except RuntimeError as e:
            assert "Not returning standard deviation" in str(e)

        # return_std=True
        options = {GaussianProcessRegressor: {"return_std": True}}
        model_onnx = to_onnx(
            gp, options=options,
            initial_types=[('X', FloatTensorType([None, None]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        self.check_outputs(gp, model_onnx, Xtest_.astype(np.float32),
                           predict_attributes=options[
            GaussianProcessRegressor])

        # return_cov=True
        options = {GaussianProcessRegressor: {"return_cov": True}}
        # model_onnx = to_onnx(gp, Xtrain_.astype(np.float32), options=options)
        model_onnx = to_onnx(
            gp, options=options,
            initial_types=[('X', FloatTensorType([None, None]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        self.check_outputs(gp, model_onnx, Xtest_.astype(np.float32),
                           predict_attributes=options[
            GaussianProcessRegressor])

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="shape_inference fails")
    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_gpr_rbf_fitted_true(self):

        gp = GaussianProcessRegressor(alpha=1e-5,
                                      n_restarts_optimizer=25,
                                      normalize_y=True)
        gp, X = fit_regression_model(gp)

        # return_cov=False, return_std=False
        model_onnx = to_onnx(
            gp, initial_types=[('X', DoubleTensorType([None, None]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.float64), gp, model_onnx,
                            verbose=False,
                            basename="SklearnGaussianProcessRBFTDouble")

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="shape_inference fails")
    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_gpr_cosine_fitted_true_float(self):
        gp = GaussianProcessRegressor(alpha=1e-5,
                                      n_restarts_optimizer=25,
                                      normalize_y=False,
                                      kernel=PairwiseKernel(metric='cosine'))
        gp, X = fit_regression_model(
            gp, n_features=2, n_samples=20, factor=0.01)

        # return_cov=False, return_std=False
        model_onnx = to_onnx(
            gp, initial_types=[('X', FloatTensorType([None, None]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.float32), gp, model_onnx,
                            verbose=False,
                            basename="SklearnGaussianProcessCosineFloat-Dec2")

    @unittest.skipIf(
        StrictVersion(ort_version) < StrictVersion("1.6.0"),
        reason="shape_inference fails")
    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_gpr_cosine_fitted_true_double(self):
        gp = GaussianProcessRegressor(alpha=1e-5,
                                      n_restarts_optimizer=25,
                                      normalize_y=False,
                                      kernel=PairwiseKernel(metric='cosine'))
        gp, X = fit_regression_model(
            gp, n_features=2, n_samples=20, factor=0.01)

        # return_cov=False, return_std=False
        model_onnx = to_onnx(
            gp, initial_types=[('X', DoubleTensorType([None, None]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(np.float64), gp, model_onnx,
                            verbose=False,
                            basename="SklearnGaussianProcessCosineDouble")

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_gpr_rbf_fitted_false(self):

        gp = GaussianProcessRegressor(alpha=1e-5,
                                      n_restarts_optimizer=25,
                                      normalize_y=False)
        gp.fit(Xtrain_, Ytrain_)

        # return_cov=False, return_std=False
        model_onnx = to_onnx(
            gp, initial_types=[('X', FloatTensorType([None, None]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(Xtest_.astype(np.float32), gp, model_onnx,
                            verbose=False,
                            basename="SklearnGaussianProcessRBF-Dec4")

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_gpr_rbf_fitted_return_std_true(self):
        gp = GaussianProcessRegressor(alpha=1e-5,
                                      n_restarts_optimizer=25,
                                      normalize_y=True)
        gp.fit(Xtrain_, Ytrain_)

        # return_cov=False, return_std=False
        options = {GaussianProcessRegressor: {"return_std": True}}
        try:
            to_onnx(
                gp, initial_types=[('X', FloatTensorType([None, None]))],
                options=options, target_opset=TARGET_OPSET)
        except RuntimeError as e:
            assert "The method *predict* must be called" in str(e)
        gp.predict(Xtrain_, return_std=True)
        model_onnx = to_onnx(
            gp, initial_types=[('X', FloatTensorType([None, None]))],
            options=options, target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        self.check_outputs(gp, model_onnx, Xtest_.astype(np.float32),
                           predict_attributes=options[
            GaussianProcessRegressor],
            decimal=4, disable_optimisation=True)
        dump_data_and_model(Xtest_.astype(np.float32), gp, model_onnx,
                            verbose=False,
                            basename="SklearnGaussianProcessRBFStd-Out0",
                            disable_optimisation=True)

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @unittest.skipIf(
        TARGET_OPSET >= 12, reason="TARGET_OPSET < 12")
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_gpr_rbf_fitted_return_std_exp_sine_squared_true(self):
        state = np.random.RandomState(0)
        X = 15 * state.rand(100, 2)
        y = np.sin(X[:, 0] - X[:, 1]).ravel()
        y += 0.5 * (0.5 - state.rand(X.shape[0]))
        y /= 10
        X_train, X_test, y_train, _ = train_test_split(X, y)
        gp = GaussianProcessRegressor(
            kernel=ExpSineSquared(periodicity_bounds=(1e-10, 1e10)),
            alpha=1e-7, n_restarts_optimizer=25, normalize_y=True,
            random_state=1)
        try:
            gp.fit(X_train, y_train)
        except (AttributeError, TypeError):
            # unstable bug in scikit-learn, fixed in 0.24
            return

        # return_cov=False, return_std=False
        options = {GaussianProcessRegressor: {"return_std": True}}
        gp.predict(X_train, return_std=True)
        model_onnx = to_onnx(
            gp, initial_types=[('X', DoubleTensorType([None, None]))],
            options=options, target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test.astype(np.float64), gp, model_onnx,
            verbose=False,
            basename="SklearnGaussianProcessExpSineSquaredStdT-Out0-Dec2",
            disable_optimisation=True)
        self.check_outputs(gp, model_onnx, X_test.astype(np.float64),
                           predict_attributes=options[
            GaussianProcessRegressor],
            decimal=4, disable_optimisation=True)

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_gpr_rbf_fitted_return_std_exp_sine_squared_false(self):
        X = 15 * np.random.rand(100, 2)
        y = np.sin(X[:, 0] - X[:, 1]).ravel()
        y += 0.1 * (0.5 - np.random.rand(X.shape[0]))
        X_train, X_test, y_train, _ = train_test_split(X, y)
        gp = GaussianProcessRegressor(
            kernel=ExpSineSquared(periodicity_bounds=(1e-10, 1e10)),
            alpha=1e-7, n_restarts_optimizer=20, normalize_y=False,
            random_state=0)
        try:
            gp.fit(X_train, y_train)
        except (AttributeError, TypeError):
            # unstable bug in scikit-learn, fixed in 0.24
            return

        # return_cov=False, return_std=False
        options = {GaussianProcessRegressor: {"return_std": True}}
        gp.predict(X_train, return_std=True)
        model_onnx = to_onnx(
            gp, initial_types=[('X', DoubleTensorType([None, None]))],
            options=options, target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test.astype(np.float64), gp, model_onnx,
            verbose=False,
            basename="SklearnGaussianProcessExpSineSquaredStdF-Out0-Dec3")
        self.check_outputs(gp, model_onnx, X_test.astype(np.float64),
                           predict_attributes=options[
            GaussianProcessRegressor],
            decimal=3)

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_gpr_rbf_fitted_return_std_exp_sine_squared_double_true(self):

        gp = GaussianProcessRegressor(kernel=ExpSineSquared(),
                                      alpha=1e-7,
                                      n_restarts_optimizer=15,
                                      normalize_y=True)
        try:
            gp.fit(Xtrain_, Ytrain_)
        except (AttributeError, TypeError) as e:
            # unstable issue fixed with scikit-learn>=0.24
            warnings.warn(
                "Training did not converge but fails at raising "
                "a warning: %r." % e)
            return

        # return_cov=False, return_std=False
        options = {GaussianProcessRegressor: {"return_std": True}}
        gp.predict(Xtrain_, return_std=True)
        model_onnx = to_onnx(
            gp, initial_types=[('X', DoubleTensorType([None, None]))],
            options=options, target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            Xtest_.astype(np.float64), gp, model_onnx,
            verbose=False,
            basename="SklearnGaussianProcessExpSineSquaredStdDouble-Out0-Dec3",
            disable_optimisation=True)
        self.check_outputs(gp, model_onnx, Xtest_.astype(np.float64),
                           predict_attributes=options[
            GaussianProcessRegressor],
            decimal=3, disable_optimisation=True)

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @unittest.skipIf(
        TARGET_OPSET >= 12, reason="TARGET_OPSET < 12")
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_gpr_rbf_fitted_return_std_dot_product_true(self):
        X = 15 * np.random.rand(100, 2)
        y = np.sin(X[:, 0] - X[:, 1]).ravel()
        y += 0.5 * (0.5 - np.random.rand(X.shape[0]))
        X_train, X_test, y_train, _ = train_test_split(X, y)
        gp = GaussianProcessRegressor(kernel=DotProduct(),
                                      alpha=1e-2,
                                      n_restarts_optimizer=25,
                                      normalize_y=True,
                                      random_state=0)
        try:
            gp.fit(X_train, y_train)
        except (AttributeError, TypeError):
            # unstable bug in scikit-learn, fixed in 0.24
            return

        gp.predict(X_train, return_std=True)

        # return_cov=False, return_std=False
        options = {GaussianProcessRegressor: {"return_std": True}}
        model_onnx = to_onnx(
            gp, initial_types=[('X', DoubleTensorType([None, None]))],
            options=options, target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test.astype(np.float64), gp, model_onnx,
            basename="SklearnGaussianProcessDotProductStdDouble-Out0-Dec3",
            disable_optimisation=True)
        self.check_outputs(gp, model_onnx, X_test.astype(np.float64),
                           predict_attributes=options[
            GaussianProcessRegressor],
            decimal=3, disable_optimisation=True)

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @unittest.skipIf(
        TARGET_OPSET >= 12, reason="TARGET_OPSET < 12")
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_gpr_rbf_fitted_return_std_rational_quadratic_true(self):

        X, y = make_regression(n_features=2, n_informative=2, random_state=2)
        X_train, X_test, y_train, _ = train_test_split(X, y)
        gp = GaussianProcessRegressor(kernel=RationalQuadratic(),
                                      alpha=1e-3,
                                      n_restarts_optimizer=25,
                                      normalize_y=True)
        try:
            gp.fit(X_train, y_train)
        except (AttributeError, TypeError):
            # unstable bug fixed in scikit-learn 0.24
            return
        gp.predict(X_train, return_std=True)

        # return_cov=False, return_std=False
        options = {GaussianProcessRegressor: {"return_std": True}}
        model_onnx = to_onnx(
            gp, initial_types=[('X', DoubleTensorType([None, None]))],
            options=options, target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test.astype(np.float64), gp, model_onnx,
            basename="SklearnGaussianProcessRationalQuadraticStdDouble-Out0",
            disable_optimisation=True)
        self.check_outputs(gp, model_onnx, X_test.astype(np.float64),
                           predict_attributes=options[
            GaussianProcessRegressor],
            disable_optimisation=True)

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_gpr_fitted_shapes(self):
        data = load_iris()
        X = data.data.astype(np.float32)
        y = data.target.astype(np.float32)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        gp = GaussianProcessRegressor()
        gp.fit(X_train, y_train)

        model_onnx = to_onnx(
            gp, initial_types=[('X', FloatTensorType([None, None]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        self.check_outputs(gp, model_onnx, X_test, {}, skip_if_float32=True)

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_gpr_fitted_partial_float64(self):
        data = load_iris()
        X = data.data
        y = data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        gp = GaussianProcessRegressor(kernel=DotProduct(), alpha=10.)
        gp.fit(X_train, y_train)

        model_onnx = to_onnx(
            gp, initial_types=[('X', FloatTensorType([None, None]))],
            target_opset=_TARGET_OPSET_)
        self.assertTrue(model_onnx is not None)
        try:
            self.check_outputs(gp, model_onnx, X_test.astype(np.float32), {})
        except AssertionError as e:
            assert "Max relative difference:" in str(e)

        model_onnx = to_onnx(
            gp, initial_types=[('X', DoubleTensorType([None, None]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        self.check_outputs(gp, model_onnx, X_test, {})

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD2),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_gpr_fitted_partial_float64_operator_cdist_rbf(self):
        data = load_iris()
        X = data.data
        y = data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        gp = GaussianProcessRegressor(kernel=RBF(), alpha=10.)
        gp.fit(X_train, y_train)

        try:
            to_onnx(
                gp, initial_types=[('X', FloatTensorType([None, None]))],
                options={GaussianProcessRegressor: {'optim': 'CDIST'}},
                target_opset=TARGET_OPSET)
            raise AssertionError("CDIST is not implemented")
        except ValueError:
            pass

        model_onnx = to_onnx(
            gp, initial_types=[('X', FloatTensorType([None, None]))],
            options={GaussianProcessRegressor: {'optim': 'cdist'}},
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        name_save = inspect.currentframe().f_code.co_name + '.onnx'
        with open(name_save, 'wb') as f:
            f.write(model_onnx.SerializeToString())
        try:
            self.check_outputs(gp, model_onnx, X_test.astype(np.float32), {})
        except RuntimeError as e:
            if "CDist is not a registered" in str(e):
                return
        except AssertionError as e:
            assert "Max relative difference:" in str(e)

        model_onnx = to_onnx(
            gp, initial_types=[('X', DoubleTensorType([None, None]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        self.check_outputs(gp, model_onnx, X_test, {})

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD2),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_gpr_fitted_partial_float64_operator_cdist_sine(self):
        data = load_iris()
        X = data.data[:, :2]
        y = data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        gp = GaussianProcessRegressor(kernel=ExpSineSquared(), alpha=100.)
        gp.fit(X_train, y_train)

        try:
            to_onnx(
                gp, initial_types=[('X', FloatTensorType([None, None]))],
                options={GaussianProcessRegressor: {'optim': 'CDIST'}},
                target_opset=TARGET_OPSET)
            raise AssertionError("CDIST is not implemented")
        except ValueError:
            pass

        model_onnx = to_onnx(
            gp, initial_types=[('X', FloatTensorType([None, None]))],
            options={GaussianProcessRegressor: {'optim': 'cdist'}},
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        name_save = inspect.currentframe().f_code.co_name + '.onnx'
        with open(name_save, 'wb') as f:
            f.write(model_onnx.SerializeToString())
        try:
            self.check_outputs(gp, model_onnx, X_test.astype(np.float32), {})
        except RuntimeError as e:
            if "CDist is not a registered" in str(e):
                return
        except AssertionError as e:
            assert "Max relative difference:" in str(e)

        model_onnx = to_onnx(
            gp, initial_types=[('X', DoubleTensorType([None, None]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        self.check_outputs(gp, model_onnx, X_test, {})

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion(THRESHOLD2),
        reason="onnxruntime %s" % THRESHOLD)
    @ignore_warnings(category=(DeprecationWarning, ConvergenceWarning))
    def test_gpr_fitted_partial_float64_operator_cdist_quad(self):
        data = load_iris()
        X = data.data
        y = data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        gp = GaussianProcessRegressor(kernel=RationalQuadratic(), alpha=100.)
        gp.fit(X_train, y_train)

        try:
            to_onnx(
                gp, initial_types=[('X', FloatTensorType([None, None]))],
                options={GaussianProcessRegressor: {'optim': 'CDIST'}},
                target_opset=TARGET_OPSET)
            raise AssertionError("CDIST is not implemented")
        except ValueError:
            pass

        model_onnx = to_onnx(
            gp, initial_types=[('X', FloatTensorType([None, None]))],
            options={GaussianProcessRegressor: {'optim': 'cdist'}},
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        name_save = inspect.currentframe().f_code.co_name + '.onnx'
        with open(name_save, 'wb') as f:
            f.write(model_onnx.SerializeToString())
        try:
            self.check_outputs(gp, model_onnx, X_test.astype(np.float32), {})
        except RuntimeError as e:
            if "CDist is not a registered" in str(e):
                return
        except AssertionError as e:
            assert "Max relative difference:" in str(e)

        model_onnx = to_onnx(
            gp, initial_types=[('X', DoubleTensorType([None, None]))],
            target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        self.check_outputs(gp, model_onnx, X_test, {})

    def test_issue_789(self):
        n_samples, n_features = 10000, 10
        X, y = make_regression(n_samples, n_features)
        tx1, vx1, ty1, vy1 = train_test_split(X, y)
        model = GaussianProcessRegressor()
        pipe = make_pipeline(MinMaxScaler(feature_range=(-1, 1)), model)
        pipe.fit(tx1, ty1)
        initial_type = [('data_in', DoubleTensorType([None, X.shape[1]]))]
        onx = to_onnx(pipe, initial_types=initial_type,
                      target_opset=_TARGET_OPSET_)
        sess = InferenceSession(onx.SerializeToString())
        pred = sess.run(None, {'data_in': vx1.astype(np.float64)})
        assert_almost_equal(pipe.predict(vx1.astype(np.float64)).ravel(),
                            pred[0].ravel())

    def test_issue_789_cdist(self):
        n_samples, n_features = 10000, 10
        X, y = make_regression(n_samples, n_features)
        tx1, vx1, ty1, vy1 = train_test_split(X, y)
        model = GaussianProcessRegressor()
        pipe = make_pipeline(MinMaxScaler(feature_range=(-1, 1)), model)
        pipe.fit(tx1, ty1)
        initial_type = [('data_in', DoubleTensorType([None, X.shape[1]]))]
        onx = to_onnx(pipe, initial_types=initial_type,
                      target_opset=_TARGET_OPSET_,
                      options={GaussianProcessRegressor: {'optim': 'cdist'}})
        self.assertIn('op_type: "CDist"', str(onx))
        sess = InferenceSession(onx.SerializeToString())
        pred = sess.run(None, {'data_in': vx1.astype(np.float64)})
        assert_almost_equal(pipe.predict(vx1.astype(np.float64)).ravel(),
                            pred[0].ravel())


if __name__ == "__main__":
    # import logging
    # log = logging.getLogger('skl2onnx')
    # log.setLevel(logging.DEBUG)
    # logging.basicConfig(level=logging.DEBUG)
    # TestSklearnGaussianProcessRegressor().test_issue_789()
    unittest.main()
