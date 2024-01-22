# SPDX-License-Identifier: Apache-2.0
import unittest


class TestInvestigate(unittest.TestCase):
    def test_issue_1053(self):
        import onnxruntime as rt
        from sklearn.datasets import load_iris
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
