"""
Tests scikit-learn's binarizer converter.
"""
import unittest
import numpy
import pandas
import inspect
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import to_onnx, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from skl2onnx.operator_converters.LinearClassifier import to_onnx_linear_classifier
from skl2onnx.common._registration import get_shape_calculator
from skl2onnx._parse import _get_sklearn_operator_name, _parse_sklearn_simple_model, update_registered_parser

from test_utils import dump_data_and_model


def has_scikit_pandas():
    try:
        import sklearn_pandas
        return True
    except ImportError:
        return False


def dataframe_mapper_shape_calculator(operator):
    if len(operator.inputs) == 1:
        raise RuntimeError("DataFrameMapper has no associated parser.")


class TestOtherLibrariesInPipelineScikitPandas(unittest.TestCase):

    @unittest.skipIf(not has_scikit_pandas(), reason="scikit-pandas not installed")
    def test_scikit_pandas(self):
        from sklearn_pandas import DataFrameMapper

        df = pandas.DataFrame({
                'feat1': [1, 2, 3, 4, 5, 6],
                'feat2': [1.0, 2.0, 3.0, 2.0, 3.0, 4.0]
        })

        mapper = DataFrameMapper([(['feat1', 'feat2'], StandardScaler()),
                                  (['feat1', 'feat2'], MinMaxScaler())])
        df2 = mapper.fit_transform(df)

        try:
            model_onnx = to_onnx(mapper, 'predictable_tsne',
                                         [('input', FloatTensorType([1, df.shape[1]]))],
                                         custom_shape_calculators={DataFrameMapper: dataframe_mapper_shape_calculator})
        except RuntimeError as e:
            assert "DataFrameMapper has no associated parser." in str(e)


if __name__ == "__main__":
    unittest.main()
