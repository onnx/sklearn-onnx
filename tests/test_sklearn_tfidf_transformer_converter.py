# SPDX-License-Identifier: Apache-2.0

# coding: utf-8
"""
Tests scikit-learn's TfidfTransformer converter.
"""
import unittest
import numpy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from test_utils import dump_data_and_model, TARGET_OPSET


class TestSklearnTfidfTransformerConverter(unittest.TestCase):

    def test_model_tfidf_transform(self):
        corpus = numpy.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
            "Troisième document en français",
        ]).reshape((5, 1))
        data = (CountVectorizer(ngram_range=(1, 1)).fit_transform(
            corpus.ravel()).todense())
        data = data.astype(numpy.float32)

        for sublinear_tf in (False, True):
            if sublinear_tf:
                # scikit-learn applies a log on a matrix
                # but only on strictly positive coefficients
                break
            for norm in (None, "l1", "l2"):
                for smooth_idf in (False, True):
                    for use_idf in (False, True):
                        model = TfidfTransformer(
                            norm=norm,
                            use_idf=use_idf,
                            smooth_idf=smooth_idf,
                            sublinear_tf=sublinear_tf,
                        )
                        model.fit(data)
                        model_onnx = convert_sklearn(
                            model,
                            "TfidfTransformer",
                            [("input",
                              FloatTensorType([None, data.shape[1]]))],
                            target_opset=TARGET_OPSET
                        )
                        self.assertTrue(model_onnx is not None)
                        suffix = norm.upper() if norm else ""
                        suffix += "Sub" if sublinear_tf else ""
                        suffix += "Idf" if use_idf else ""
                        suffix += "Smooth" if smooth_idf else ""
                        dump_data_and_model(
                            data,
                            model,
                            model_onnx,
                            basename="SklearnTfidfTransform" + suffix,
                            # Operator mul is not implemented in onnxruntime
                            allow_failure="StrictVersion(onnx.__version__)"
                                          " < StrictVersion('1.2')",
                        )


if __name__ == "__main__":
    unittest.main()
