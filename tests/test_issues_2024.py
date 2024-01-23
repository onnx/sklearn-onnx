# SPDX-License-Identifier: Apache-2.0
import unittest
import packaging.version as pv
from onnxruntime import __version__ as ort_version


class TestInvestigate(unittest.TestCase):
    def test_issue_1053(self):
        from sklearn.datasets import load_iris
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
        import onnxruntime as rt
        from skl2onnx.common.data_types import FloatTensorType
        from skl2onnx import convert_sklearn

        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        # Fitting logistic regression model.
        for cls in [LogisticRegression, DecisionTreeClassifier]:
            with self.subTest(cls=cls):
                clr = cls()  # Use logistic regression instead of decision tree.
                clr.fit(X_train, y_train)

                initial_type = [
                    ("float_input", FloatTensorType([None, 4]))
                ]  # Remove the batch dimension.
                onx = convert_sklearn(clr, initial_types=initial_type, target_opset=12)

                sess = rt.InferenceSession(
                    onx.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                input_name = sess.get_inputs()[0].name
                label_name = sess.get_outputs()[0].name
                pred_onx = sess.run(
                    [label_name], {input_name: X_test[:1].astype("float32")}
                )[
                    0
                ]  # Select a single sample.
                self.assertEqual(len(pred_onx.tolist()), 1)

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("1.16.0"),
        reason="opset 19 not implemented",
    )
    def test_issue_1055(self):
        import numpy as np
        from numpy.testing import assert_almost_equal
        import sklearn.feature_extraction.text
        import sklearn.linear_model
        import sklearn.pipeline
        import onnxruntime as rt
        import skl2onnx.common.data_types

        lr = sklearn.linear_model.LogisticRegression(
            C=100,
            multi_class="multinomial",
            solver="sag",
            class_weight="balanced",
            n_jobs=-1,
        )
        tf = sklearn.feature_extraction.text.TfidfVectorizer(
            token_pattern="\\w+|[^\\w\\s]+",
            ngram_range=(1, 1),
            max_df=1.0,
            min_df=1,
            sublinear_tf=True,
        )

        pipe = sklearn.pipeline.Pipeline([("transformer", tf), ("logreg", lr)])

        corpus = [
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
            "more text",
            "$words",
            "I keep writing things",
            "how many documents now?",
            "this is a really long setence",
            "is this a final document?",
        ]
        labels = ["1", "2", "1", "2", "1", "2", "1", "2", "1", "2"]

        pipe.fit(corpus, labels)

        onx = skl2onnx.convert_sklearn(
            pipe,
            "a model",
            initial_types=[
                ("input", skl2onnx.common.data_types.StringTensorType([None, 1]))
            ],
            target_opset=19,
            options={"zipmap": False},
        )
        for d in onx.opset_import:
            if d.domain == "":
                self.assertEqual(d.version, 19)
            elif d.domain == "com.microsoft":
                self.assertEqual(d.version, 1)
            elif d.domain == "ai.onnx.ml":
                self.assertEqual(d.version, 1)

        expected = pipe.predict_proba(corpus)
        sess = rt.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, {"input": np.array(corpus).reshape((-1, 1))})
        assert_almost_equal(expected, got[1], decimal=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
