# SPDX-License-Identifier: Apache-2.0

"""
Tests pipeline within pipelines.
"""

import numpy as np
import unittest
try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    # not available in 0.19
    ColumnTransformer = None
try:
    from sklearn.impute import SimpleImputer
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler, StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import dump_data_and_model, TARGET_OPSET


class TestSklearnPipelineWithinPipeline(unittest.TestCase):
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_pipeline_pca_pipeline_minmax(self):
        model = Pipeline(
            memory=None,
            steps=[
                (
                    "PCA",
                    PCA(
                        copy=True,
                        iterated_power="auto",
                        n_components=0.15842105263157896,
                        random_state=None,
                        tol=0.0,
                        svd_solver="auto",
                        whiten=False,
                    ),
                ),
                (
                    "Pipeline",
                    Pipeline(
                        memory=None,
                        steps=[(
                            "MinMax scaler",
                            MinMaxScaler(
                                copy=True,
                                feature_range=(0, 3.7209871159509307),
                            ),
                        )],
                    ),
                ),
            ],
        )

        data = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.float32)
        y = [0, 0, 1, 1]
        model.fit(data, y)
        model_onnx = convert_sklearn(
            model,
            "pipelinewithinpipeline",
            [("input", FloatTensorType(data.shape))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data,
            model,
            model_onnx,
            basename="SklearnPipelinePcaPipelineMinMax",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_pipeline_pca_pipeline_none_lin(self):
        model = Pipeline(
            memory=None,
            steps=[
                (
                    "PCA",
                    PCA(
                        copy=True,
                        iterated_power="auto",
                        n_components=0.15842105263157896,
                        random_state=None,
                        tol=0.0,
                        svd_solver="auto",
                        whiten=False,
                    ),
                ),
                (
                    "Pipeline",
                    Pipeline(
                        memory=None,
                        steps=[
                            (
                                "MinMax scaler",
                                MinMaxScaler(
                                    copy=True,
                                    feature_range=(0, 3.7209871159509307),
                                ),
                            ),
                            ("logreg", LogisticRegression(solver="liblinear")),
                        ],
                    ),
                ),
            ],
        )

        data = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.float32)
        y = [0, 0, 1, 1]
        model.fit(data, y)
        model_onnx = convert_sklearn(
            model,
            "pipelinewithinpipeline",
            [("input", FloatTensorType(data.shape))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data,
            model,
            model_onnx,
            basename="SklearnPipelinePcaPipelineMinMaxLogReg",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_pipeline_pca_pipeline_multinomial(self):
        model = Pipeline(
            memory=None,
            steps=[
                (
                    "PCA",
                    PCA(
                        copy=True,
                        iterated_power="auto",
                        n_components=2,
                        random_state=None,
                        svd_solver="auto",
                        tol=0.0,
                        whiten=False,
                    ),
                ),
                (
                    "Pipeline",
                    Pipeline(
                        memory=None,
                        steps=[
                            (
                                "MinMax scaler",
                                MinMaxScaler(
                                    copy=True,
                                    feature_range=(0, 3.7209871159509307),
                                ),
                            ),
                            (
                                "MultinomialNB",
                                MultinomialNB(
                                    alpha=0.7368421052631579,
                                    class_prior=None,
                                    fit_prior=True,
                                ),
                            ),
                        ],
                    ),
                ),
            ],
        )

        data = np.array(
            [[0, 0, 0], [0, 0, 0.1], [1, 1, 1.1], [1, 1.1, 1]],
            dtype=np.float32,
        )
        y = [0, 0, 1, 1]
        model.fit(data, y)
        model_onnx = convert_sklearn(
            model,
            "pipelinewithinpipeline",
            [("input", FloatTensorType(data.shape))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data,
            model,
            model_onnx,
            basename="SklearnPipelinePcaPipelineMinMaxNB2",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_pipeline_pca_pipeline_multinomial_none(self):
        model = Pipeline(
            memory=None,
            steps=[
                (
                    "PCA",
                    PCA(
                        copy=True,
                        iterated_power="auto",
                        n_components=0.15842105263157896,
                        random_state=None,
                        tol=0.0,
                        svd_solver="auto",
                        whiten=False,
                    ),
                ),
                (
                    "Pipeline",
                    Pipeline(
                        memory=None,
                        steps=[
                            (
                                "MinMax scaler",
                                MinMaxScaler(
                                    copy=True,
                                    feature_range=(0, 3.7209871159509307),
                                ),
                            ),
                            (
                                "MultinomialNB",
                                MultinomialNB(
                                    alpha=0.7368421052631579,
                                    class_prior=None,
                                    fit_prior=True,
                                ),
                            ),
                        ],
                    ),
                ),
            ],
        )

        data = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.float32)
        y = [0, 0, 1, 1]
        model.fit(data, y)
        model_onnx = convert_sklearn(
            model,
            "pipelinewithinpipeline",
            [("input", FloatTensorType(data.shape))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            data,
            model,
            model_onnx,
            basename="SklearnPipelinePcaPipelineMinMaxNBNone",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(
        ColumnTransformer is None,
        reason="ColumnTransformer not available in 0.19",
    )
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_pipeline_column_transformer_pipeline_imputer_scaler_lr(self):
        X = np.array([[1, 2], [3, np.nan], [3, 0]], dtype=np.float32)
        y = np.array([1, 0, 1])
        model = Pipeline([
            (
                "ct",
                ColumnTransformer([
                    (
                        "pipeline1",
                        Pipeline([
                            ("imputer", SimpleImputer()),
                            ("scaler", StandardScaler()),
                        ]),
                        [0],
                    ),
                    (
                        "pipeline2",
                        Pipeline([
                            ("imputer", SimpleImputer()),
                            ("scaler", RobustScaler()),
                        ]),
                        [1],
                    ),
                ]),
            ),
            ("lr", LogisticRegression(solver="liblinear")),
        ])
        model.fit(X, y)
        model_onnx = convert_sklearn(
            model,
            "pipelinewithinpipeline",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnPipelineCTPipelineImputerScalerLR",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )


if __name__ == "__main__":
    unittest.main()
