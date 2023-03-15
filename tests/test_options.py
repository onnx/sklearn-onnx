# SPDX-License-Identifier: Apache-2.0

"""
Tests topology.
"""
import unittest
import numpy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import datasets
from skl2onnx import to_onnx, update_registered_converter
from skl2onnx.algebra.onnx_ops import OnnxIdentity, OnnxAdd
from test_utils import TARGET_OPSET


class DummyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        TransformerMixin.__init__(self)
        BaseEstimator.__init__(self)

    def fit(self, X, y, sample_weight=None):
        return self

    def transform(self, X):
        return X


def dummy_shape_calculator(operator):
    op_input = operator.inputs[0]
    operator.outputs[0].type.shape = op_input.type.shape


def dummy_converter(scope, operator, container):
    X = operator.inputs[0]
    out = operator.outputs
    opv = container.target_opset

    options = container.get_options(operator.raw_operator)
    if len(options) == 0:
        cst = numpy.array([57777], dtype=numpy.float32)
    elif len(options) == 1:
        opts = list(options.items())
        if opts[0][0] == 'opt1':
            if opts[0][1] is None:
                cst = numpy.array([57789], dtype=numpy.float32)
            elif opts[0][1]:
                cst = numpy.array([57778], dtype=numpy.float32)
            elif not opts[0][1]:
                cst = numpy.array([57779], dtype=numpy.float32)
            else:
                raise AssertionError("Issue with %r." % options)
        elif opts[0][0] == 'opt3':
            if opts[0][1] is None:
                cst = numpy.array([51789], dtype=numpy.float32)
            elif opts[0][1] == 'r':
                cst = numpy.array([56779], dtype=numpy.float32)
            elif opts[0][1] == 't':
                cst = numpy.array([58779], dtype=numpy.float32)
            else:
                raise AssertionError("Issue with %r." % options)
        elif opts[0][0] == 'opt2':
            if opts[0][1] is None:
                cst = numpy.array([44444], dtype=numpy.float32)
            elif isinstance(opts[0][1], int):
                cst = numpy.array([opts[0][1]], dtype=numpy.float32)
            else:
                raise AssertionError("Issue with %r." % options)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    id1 = OnnxIdentity(X, op_version=opv)
    op = OnnxAdd(id1, cst, op_version=opv)
    id2 = OnnxIdentity(op, output_names=out[:1],
                       op_version=opv)
    id2.add_to(scope, container)


class TestOptions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        update_registered_converter(
            DummyTransformer, "IdentityTransformer",
            dummy_shape_calculator, dummy_converter,
            options={'opt1': [False, True], 'opt2': None,
                     'opt3': ('r', 't'), 'opt4': -1})

    def check_in(self, value, onx):
        if str(value) not in str(onx):
            raise AssertionError(
                "Unable to find %r in\n%s" % (str(value), str(onx)))

    def test_no_options(self):
        digits = datasets.load_digits(n_class=6)
        Xd = digits.data[:20].astype(numpy.float32)
        yd = digits.target[:20]
        idtr = DummyTransformer().fit(Xd, yd)
        model_onnx = to_onnx(idtr, Xd, target_opset=TARGET_OPSET)
        self.check_in('57777', model_onnx)

    def test_options_list_true(self):
        digits = datasets.load_digits(n_class=6)
        Xd = digits.data[:20].astype(numpy.float32)
        yd = digits.target[:20]
        idtr = DummyTransformer().fit(Xd, yd)
        model_onnx = to_onnx(idtr, Xd, target_opset=TARGET_OPSET,
                             options={'opt1': True})
        self.check_in('57778', model_onnx)

    def test_options_list_false(self):
        digits = datasets.load_digits(n_class=6)
        Xd = digits.data[:20].astype(numpy.float32)
        yd = digits.target[:20]
        idtr = DummyTransformer().fit(Xd, yd)
        model_onnx = to_onnx(idtr, Xd, target_opset=TARGET_OPSET,
                             options={'opt1': False})
        self.check_in('57779', model_onnx)

    def test_options_list_outside_none(self):
        digits = datasets.load_digits(n_class=6)
        Xd = digits.data[:20].astype(numpy.float32)
        yd = digits.target[:20]
        idtr = DummyTransformer().fit(Xd, yd)
        model_onnx = to_onnx(idtr, Xd, target_opset=TARGET_OPSET,
                             options={'opt1': None})
        self.check_in('57789', model_onnx)

    def test_options_list_outside(self):
        digits = datasets.load_digits(n_class=6)
        Xd = digits.data[:20].astype(numpy.float32)
        yd = digits.target[:20]
        idtr = DummyTransformer().fit(Xd, yd)
        with self.assertRaises(ValueError):
            # value not allowed
            to_onnx(idtr, Xd, target_opset=TARGET_OPSET,
                    options={'opt1': 'OUT'})

    def test_options_integer(self):
        digits = datasets.load_digits(n_class=6)
        Xd = digits.data[:20].astype(numpy.float32)
        yd = digits.target[:20]
        idtr = DummyTransformer().fit(Xd, yd)
        with self.assertRaises(TypeError):
            # integer not allowed
            to_onnx(idtr, Xd, target_opset=TARGET_OPSET,
                    options={'opt4': 44444})

    def test_options_tuple1(self):
        digits = datasets.load_digits(n_class=6)
        Xd = digits.data[:20].astype(numpy.float32)
        yd = digits.target[:20]
        idtr = DummyTransformer().fit(Xd, yd)
        model_onnx = to_onnx(idtr, Xd, target_opset=TARGET_OPSET,
                             options={'opt3': 't'})
        self.check_in('58779', model_onnx)

    def test_options_tuple2(self):
        digits = datasets.load_digits(n_class=6)
        Xd = digits.data[:20].astype(numpy.float32)
        yd = digits.target[:20]
        idtr = DummyTransformer().fit(Xd, yd)
        model_onnx = to_onnx(idtr, Xd, target_opset=TARGET_OPSET,
                             options={'opt3': 'r'})
        self.check_in('56779', model_onnx)

    def test_options_tuple_none(self):
        digits = datasets.load_digits(n_class=6)
        Xd = digits.data[:20].astype(numpy.float32)
        yd = digits.target[:20]
        idtr = DummyTransformer().fit(Xd, yd)
        model_onnx = to_onnx(idtr, Xd, target_opset=TARGET_OPSET,
                             options={'opt3': None})
        self.check_in('51789', model_onnx)

    def test_options_tuple_out(self):
        digits = datasets.load_digits(n_class=6)
        Xd = digits.data[:20].astype(numpy.float32)
        yd = digits.target[:20]
        idtr = DummyTransformer().fit(Xd, yd)
        with self.assertRaises(ValueError):
            # value not allowed
            to_onnx(idtr, Xd, target_opset=TARGET_OPSET,
                    options={'opt3': 'G'})

    def test_options_none(self):
        digits = datasets.load_digits(n_class=6)
        Xd = digits.data[:20].astype(numpy.float32)
        yd = digits.target[:20]
        idtr = DummyTransformer().fit(Xd, yd)
        model_onnx = to_onnx(idtr, Xd, target_opset=TARGET_OPSET,
                             options={'opt2': None})
        self.check_in('44444', model_onnx)

    def test_options_num(self):
        digits = datasets.load_digits(n_class=6)
        Xd = digits.data[:20].astype(numpy.float32)
        yd = digits.target[:20]
        idtr = DummyTransformer().fit(Xd, yd)
        model_onnx = to_onnx(idtr, Xd, target_opset=TARGET_OPSET,
                             options={'opt2': 33333})
        self.check_in('33333', model_onnx)


if __name__ == "__main__":
    unittest.main()
