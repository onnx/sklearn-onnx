# SPDX-License-Identifier: Apache-2.0

"""
Tests scikit-learn's binarizer converter.
"""
import unittest
import pandas
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn


def has_scikit_pandas():
    try:
        import sklearn_pandas  # noqa
        return True
    except ImportError:
        return False


def dataframe_mapper_shape_calculator(operator):
    if len(operator.inputs) == 1:
        raise RuntimeError("DataFrameMapper has no associated parser.")


class TestOtherLibrariesInPipelineScikitPandas(unittest.TestCase):
    @unittest.skipIf(not has_scikit_pandas(),
                     reason="scikit-pandas not installed")
    def test_scikit_pandas(self):
        from sklearn_pandas import DataFrameMapper

        df = pandas.DataFrame({
            "feat1": [1, 2, 3, 4, 5, 6],
            "feat2": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
        })

        mapper = DataFrameMapper([
            (["feat1", "feat2"], StandardScaler()),
            (["feat1", "feat2"], MinMaxScaler()),
        ])

        try:
            model_onnx = convert_sklearn(  # noqa
                mapper,
                "predictable_tsne",
                [("input", FloatTensorType([None, df.shape[1]]))],
                custom_shape_calculators={
                    DataFrameMapper: dataframe_mapper_shape_calculator
                },
            )
        except RuntimeError as e:
            assert "DataFrameMapper has no associated parser." in str(e)


if __name__ == "__main__":
    unittest.main()
