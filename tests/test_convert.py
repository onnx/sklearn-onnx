# SPDX-License-Identifier: Apache-2.0


import unittest
import warnings
import numpy
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.cluster import KMeans
try:
    from sklearn.preprocessing import KBinsDiscretizer
except ImportError:
    KBinsDiscretizer = None
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.datasets import load_iris
from skl2onnx import to_onnx, convert_sklearn
from test_utils import TARGET_OPSET


def get_domain_opset(onx):
    domains = onx.opset_import
    res = [{'domain': dom.domain, 'version': dom.version}
           for dom in domains]
    return {d['domain']: d['version'] for d in res}


class TestConvert(unittest.TestCase):

    def test_target_opset(self):
        data = load_iris()
        X = data.data
        model = KMeans(n_clusters=3)
        model.fit(X)
        for i in range(1, TARGET_OPSET+1):
            model_onnx = to_onnx(model, X[:1].astype(numpy.float32),
                                 target_opset=i)
            dom = get_domain_opset(model_onnx)
            self.assertEqual(len(dom), 1)
            assert dom[''] <= i

    def test_target_opset_dict(self):
        data = load_iris()
        X = data.data
        model = KMeans(n_clusters=3)
        model.fit(X)
        for i in range(1, TARGET_OPSET+1):
            for j in (1, 2):
                tops = {'': i, 'ai.onnx.ml': j}
                model_onnx = to_onnx(model, X[:1].astype(numpy.float32),
                                     target_opset=tops)
                dom = get_domain_opset(model_onnx)
                self.assertEqual(len(dom), 1)
                assert dom[''] <= i

    @unittest.skipIf(KBinsDiscretizer is None, "skl too old")
    def test_target_opset_dict_kbins(self):
        data = load_iris()
        X = data.data
        model = KBinsDiscretizer(encode="ordinal")
        model.fit(X)
        for i in range(9, TARGET_OPSET+1):
            for j in (1, 2):
                tops = {'': i, 'ai.onnx.ml': j}
                model_onnx = to_onnx(model, X[:1].astype(numpy.float32),
                                     target_opset=tops)
                dom = get_domain_opset(model_onnx)
                if dom != {'ai.onnx.ml': 1, '': i}:
                    assert dom[''] <= i
                    assert dom['ai.onnx.ml'] == 1
                    continue
                self.assertEqual(dom, {'ai.onnx.ml': 1, '': i})

    def test_regressor(self):
        data = load_iris()
        X = data.data
        y = data.target
        model = GaussianProcessRegressor()
        model.fit(X, y)
        for i in range(9, TARGET_OPSET+1):
            for j in (1, 2):
                tops = {'': i, 'ai.onnx.ml': j}
                model_onnx = to_onnx(model, X[:1].astype(numpy.float32),
                                     target_opset=tops)
                dom = get_domain_opset(model_onnx)
                self.assertEqual(len(dom), 1)
                self.assertIn(dom[''], (i, i-1))

    def test_onehot(self):
        try:
            model = OneHotEncoder(categories='auto')
        except TypeError:
            # parameter categories added in 0.20
            return
        data = numpy.array([[1, 2, 3], [4, 3, 0], [0, 1, 4], [0, 5, 6]],
                           dtype=numpy.int64)
        model.fit(data)
        for i in range(9, TARGET_OPSET+1):
            for j in (1, 2):
                tops = {'': i, 'ai.onnx.ml': j}
                model_onnx = to_onnx(model, data[:1],
                                     target_opset=tops)
                dom = get_domain_opset(model_onnx)
                self.assertEqual(len(dom), 2)
                self.assertIn(dom[''], list(range(9, TARGET_OPSET+1)))
                self.assertEqual(dom['ai.onnx.ml'], 1)

    def test_label_encoder(self):
        model = LabelEncoder()
        data = numpy.array([1.2, 3.4, 5.4, 1.2],
                           dtype=numpy.float32)
        model.fit(data)
        for i in range(9, TARGET_OPSET+1):
            for j in (1, 2):
                tops = {'': i, 'ai.onnx.ml': j}
                try:
                    model_onnx = to_onnx(model, data[:1],
                                         target_opset=tops)
                except RuntimeError as e:
                    if j == 1:
                        # expected
                        continue
                    raise e
                if j == 1:
                    raise AssertionError("It should fail for opset.ml == 1")
                dom = get_domain_opset(model_onnx)
                self.assertEqual(len(dom), 2)
                self.assertEqual(dom['ai.onnx.ml'], 2)

    def test_warnings(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                convert_sklearn(None, initial_types=[], dtype=numpy.float64)
            except IndexError:
                pass
            assert len(w) == 1
            assert "Parameter dtype is no longer supported." in str(w[0])

    @unittest.skipIf(KBinsDiscretizer is None, "skl too old")
    def test_name(self):
        data = load_iris()
        X = data.data
        model = KBinsDiscretizer(encode="ordinal")
        model.fit(X)

        with self.assertRaises(TypeError):
            to_onnx(model, X[:1].astype(numpy.float32),
                    target_opset=TARGET_OPSET, naming=(2, 3))

        model_onnx = to_onnx(model, X[:1].astype(numpy.float32),
                             target_opset=TARGET_OPSET, naming='KBINS')
        inputs = set(i.name for i in model_onnx.graph.input)
        outputs = set(o.name for o in model_onnx.graph.output)
        for node in model_onnx.graph.node:
            for i in node.input:
                if i not in inputs and not i.startswith('KBINS'):
                    raise AssertionError("Wrong %r." % i)
            for o in node.output:
                if o not in outputs and not o.startswith('KBINS'):
                    raise AssertionError("Wrong %r." % o)

        model_onnx = to_onnx(model, X[:1].astype(numpy.float32),
                             target_opset=TARGET_OPSET,
                             naming=lambda n, ns: 'FBINS' + n)
        inputs = set(i.name for i in model_onnx.graph.input)
        outputs = set(o.name for o in model_onnx.graph.output)
        for node in model_onnx.graph.node:
            for i in node.input:
                if i not in inputs and not i.startswith('FBINS'):
                    raise AssertionError("Wrong %r." % i)
            for o in node.output:
                if o not in outputs and not o.startswith('FBINS'):
                    raise AssertionError("Wrong %r." % o)
        self.assertEqual(inputs, {'X'})
        self.assertEqual(outputs, {'variable'})


if __name__ == "__main__":
    unittest.main()
