import unittest
import numpy as np
import onnxruntime as rt
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
import skl2onnx
from skl2onnx.common.data_types import (
    FloatTensorType, Int64TensorType, DoubleTensorType)
from test_utils import TARGET_OPSET


def convert_dataframe_schema(df, drop=None, batch_axis=False):
    inputs = []
    nrows = None if batch_axis else 1
    for k, v in zip(df.columns, df.dtypes):
        if drop is not None and k in drop:
            continue
        if v == 'int64':
            t = Int64TensorType([nrows, 1])
        elif v == 'float32':
            t = FloatTensorType([nrows, 1])
        elif v == 'float64':
            t = DoubleTensorType([nrows, 1])
        else:
            raise Exception("Bad type")
        inputs.append((k, t))
    return inputs


class TestExampleAzure(unittest.TestCase):

    def test_boston(self):
        # source: https://github.com/MicrosoftDocs/azure-docs/blob/
        # master/articles/azure-sql-edge/deploy-onnx.md.
        # data
        boston = load_boston()
        df = pd.DataFrame(data=np.c_[boston['data'], boston['target']],
                          columns=boston['feature_names'].tolist() + ['MEDV'])
        # Without that line, the dataframe contains double
        # and the schema is populated with columns of type
        # DoubleTensorType instead of FloatTensorType.
        df = df.astype(np.float32)
        target_column = 'MEDV'
        x_train = pd.DataFrame(df.drop([target_column], axis=1))
        y_train = pd.DataFrame(
            df.iloc[:, df.columns.tolist().index(target_column)])

        # train
        continuous_transformer = Pipeline(steps=[('scaler', RobustScaler())])
        preprocessor = ColumnTransformer(
            transformers=[
                ('continuous', continuous_transformer,
                 [i for i in range(len(x_train.columns))])])
        model = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('regressor', LinearRegression())])
        model.fit(x_train, y_train)

        # convert
        schema = convert_dataframe_schema(x_train)
        assert isinstance(schema[0][1], FloatTensorType)
        onnx_model = skl2onnx.convert_sklearn(
            model, 'Boston Data', schema, target_opset=TARGET_OPSET)

        # test
        sess = rt.InferenceSession(onnx_model.SerializeToString())
        y_pred = np.full(shape=(len(x_train)), fill_value=np.nan)
        for i in range(len(x_train)):
            inputs = {}
            for j in range(len(x_train.columns)):
                inputs[x_train.columns[j]] = np.full(
                    shape=(1, 1), fill_value=x_train.iloc[i, j])

            sess_pred = sess.run(None, inputs)
            y_pred[i] = sess_pred[0][0][0]
        onnx_r2_score = r2_score(y_train, y_pred)
        onnx_mse = mean_squared_error(y_train, y_pred)

        y_pred = model.predict(x_train)
        sklearn_r2_score = r2_score(y_train, y_pred)
        sklearn_mse = mean_squared_error(y_train, y_pred)
        self.assertEqual(int(abs(sklearn_r2_score - onnx_r2_score) * 1e7), 0)
        self.assertEqual(int(abs(sklearn_mse - onnx_mse) * 1e6), 0)


if __name__ == "__main__":
    unittest.main()
