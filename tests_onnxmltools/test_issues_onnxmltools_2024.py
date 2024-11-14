# SPDX-License-Identifier: Apache-2.0
import unittest
import packaging.version as pv
import onnx
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from onnxruntime import __version__ as ort_version


class TestInvestigateOnnxmltools(unittest.TestCase):

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("1.17.3"),
        reason="opset 19 not implemented",
    )
    @ignore_warnings(category=(ConvergenceWarning, FutureWarning))
    def test_issue_1102(self):
        from typing import Any
        from sklearn.datasets import make_regression
        import lightgbm
        import numpy
        import onnxruntime
        import skl2onnx
        from onnx.reference import ReferenceEvaluator
        from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
            convert_lightgbm,
        )
        from skl2onnx import update_registered_converter, to_onnx
        from skl2onnx.common import shape_calculator
        from sklearn import base, multioutput, pipeline, preprocessing

        def Normalizer() -> list[tuple[str, Any]]:
            return [
                ("cast64", skl2onnx.sklapi.CastTransformer(dtype=numpy.float64)),
                ("scaler", preprocessing.StandardScaler()),
                ("cast32", skl2onnx.sklapi.CastTransformer(dtype=numpy.float32)),
            ]

        def Embedder(**kwargs: dict[str, Any]) -> list[tuple[str, Any]]:
            return [("basemodel", lightgbm.LGBMRegressor(**kwargs))]

        def BoL2EmotionV2(
            backbone_kwargs: dict[str, Any] | None = None,
        ) -> base.BaseEstimator:
            backbone = Embedder(**(backbone_kwargs or {}))
            normalizer = Normalizer()
            model = pipeline.Pipeline([*normalizer, *backbone])
            return multioutput.MultiOutputRegressor(model)

        model = BoL2EmotionV2()
        X, y = make_regression(100, n_features=4, n_targets=2)
        model.fit(X, y)

        update_registered_converter(
            lightgbm.LGBMRegressor,
            "LightGbmLGBMRegressor",
            shape_calculator.calculate_linear_regressor_output_shapes,
            convert_lightgbm,
            options={"split": None},
        )

        sample = X.astype(numpy.float32)

        exported = to_onnx(
            model,
            X=sample,
            name="BoL2emotion",
            target_opset={"": 19, "ai.onnx.ml": 2},
        )
        expected = model.predict(sample)
        onnx.shape_inference.infer_shapes(exported)

        ref = ReferenceEvaluator(exported)
        got = ref.run(None, dict(X=sample))[0]
        numpy.testing.assert_allclose(expected, got, 1e-4)

        sess = onnxruntime.InferenceSession(
            exported.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, dict(X=sample))[0]
        numpy.testing.assert_allclose(expected, got, 1e-4)

        with open("dump_model.onnx", "wb") as f:
            f.write(exported.SerializeToString())

        sess = onnxruntime.InferenceSession(
            "dump_model.onnx", providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, dict(X=sample))[0]
        numpy.testing.assert_allclose(expected, got, 1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
