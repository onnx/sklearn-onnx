# SPDX-License-Identifier: Apache-2.0


import unittest
from distutils.version import StrictVersion
import pandas as pd
import numpy as np
from onnxruntime import __version__ as ort_version
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder
try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    # old version of scikit-learn
    ColumnTransformer = None
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.pipeline import Pipeline
from test_utils import (
    dump_data_and_model,
    TARGET_OPSET
)


class TestSklearnArrayFeatureExtractor(unittest.TestCase):

    @unittest.skipIf(
        ColumnTransformer is None or
        StrictVersion(ort_version) <= StrictVersion("0.4.0"),
        reason="onnxruntime too old")
    def test_array_feature_extractor(self):
        data_to_cluster = pd.DataFrame(
            [[1, 2, 3.5, 4.5], [1, 2, 1.7, 4.0],
             [2, 4, 2.4, 4.3], [2, 4, 2.5, 4.0]],
            columns=[1, 2, 3, 4])
        cat_attributes_clustering = [1, 2]
        num_attributes_clustering = [3, 4]  # this is of length 12 in reality
        gmm = GaussianMixture(n_components=2, random_state=1)
        ohe_cat = [OneHotEncoder(categories='auto', sparse=False, drop=None)
                   for i in cat_attributes_clustering]
        ct_cat = ColumnTransformer([
            ("oneHotEncoder" + str(i), ohe_cat[i], [i])
            for i, item in enumerate(cat_attributes_clustering)
        ], remainder='passthrough')
        onehotencoding_pipeline = Pipeline([("columnTransformer", ct_cat), ])
        clustering_pipeline = Pipeline([
            ('onehotencoder_and_scaler', onehotencoding_pipeline),
            ('clustering', gmm)])
        clustering_pipeline.fit(X=data_to_cluster)
        initial_type = [
            ('float_input', FloatTensorType(
                [None, len([*cat_attributes_clustering,
                            *num_attributes_clustering])]))]
        data = data_to_cluster.values.astype(np.float32)

        # checks the first step
        model_onnx = to_onnx(
            clustering_pipeline.steps[0][1], initial_types=initial_type,
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            data, clustering_pipeline.steps[0][1], model_onnx,
            basename="SklearnArrayFeatureExtractorStep0")

        # checks the whole pipeline
        model_onnx = to_onnx(
            clustering_pipeline, initial_types=initial_type,
            target_opset=TARGET_OPSET)
        dump_data_and_model(
            data, clustering_pipeline, model_onnx,
            basename="SklearnArrayFeatureExtractor")


if __name__ == "__main__":
    unittest.main()
