"""
Tests scikit-learn's binarizer converter.
"""
import unittest
import numpy
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn, update_registered_converter
from test_utils import dump_data_and_model


def has_lightgbm():
    try:
        import lightgbm
        return True
    except ImportError:
        return False


def has_onnxmltools():
    try:
        import onnxmltools
        return True
    except ImportError:
        return False


class TestOtherLibrariesInPipeline(unittest.TestCase):

    @unittest.skipIf(not has_lightgbm(), "lightgbm missing, cannot train lightgbm model")
    @unittest.skipIf(not has_onnxmltools(), "onnxmltools missing, cannot convert lightgbm model")
    def test_lightgbm_scaler(self):
        from lightgbm import LGBMClassifier
        from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
        from skl2onnx.common.shape_calculator_deprecated import calculate_linear_classifier_output_shapes

        update_registered_converter(LGBMClassifier, 'LightGbmLGBMClassifier',                                    
                                    calculate_linear_classifier_output_shapes,
                                    convert_lightgbm)
        
        data = load_iris()
        X = data.data[:, :2]
        y = data.target
        
        model = LGBMClassifier(n_estimators=3, min_child_samples=1)
        pipe = Pipeline([('scaler', StandardScaler()), ('lgbm', model)])
        pipe.fit(X, y)

        model_onnx = convert_sklearn(pipe, 'pipeline', [('input', FloatTensorType([1, 2]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(numpy.float32), pipe, model_onnx,
                            basename="SklearnPipelineScalerLightGbm")

    def test_random_forest(self):
        data = load_iris()
        X = data.data[:, :2]
        y = data.target
        
        model = RandomForestClassifier(n_estimators=3)
        pipe = Pipeline([('scaler1', StandardScaler()), ('rf', model)])
        pipe.fit(X, y)

        model_onnx = convert_sklearn(model, 'pipeline', [('input', FloatTensorType([1, 2]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(X.astype(numpy.float32), model, model_onnx,
                            basename="SklearnPipelineScalerRandomForest")
        

if __name__ == "__main__":
    unittest.main()
