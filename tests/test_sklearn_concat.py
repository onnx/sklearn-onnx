# SPDX-License-Identifier: Apache-2.0

import os
import unittest
import onnxruntime as rt
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    ColumnTransformer = None
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    BooleanTensorType, FloatTensorType,
    Int64TensorType, StringTensorType)
from test_utils import TARGET_OPSET


def _column_tranformer_fitted_from_df(data):
    def transformer_for_column(column: pd.Series):
        if column.dtype in ['float64', 'float32', 'int64']:
            return StandardScaler()
        if column.dtype in ['bool']:
            return 'passthrough'
        if column.dtype in ['O']:
            try:
                return OneHotEncoder(drop='first')
            except TypeError:
                # older version of scikit-learn
                return OneHotEncoder()
        raise ValueError(
            'Unexpected column dtype for {column.name}:{column.dtype}'.format(
                column=column))

    return ColumnTransformer(
        [(col, transformer_for_column(
            data[col]), [col]) for col in data.columns],
        remainder='drop'
    ).fit(data)


def _convert_dataframe_schema(data):
    def type_for_column(column: pd.Series):
        if column.dtype in ['float64', 'float32']:
            return FloatTensorType([None, 1])
        if column.dtype in ['int64']:
            return Int64TensorType([None, 1])
        if column.dtype in ['bool']:
            return BooleanTensorType([None, 1])
        if column.dtype in ['O']:
            return StringTensorType([None, 1])
        raise ValueError(
            'Unexpected column dtype for {column.name}:{column.dtype}'.format(
                column=column))
    return [(col, type_for_column(data[col])) for col in data.columns]


def _predict(session: rt.InferenceSession, data: pd.DataFrame) -> pd.Series:
    def _correctly_typed_column(column: pd.Series) -> pd.Series:
        if column.dtype in ['float64']:
            return column.astype(np.float32)
        return column

    def _correctly_shaped_values(values):
        return values.reshape((values.shape[0], 1))

    inputs = {
        c: _correctly_shaped_values(_correctly_typed_column(data[c]).values)
        for c in data.columns
    }

    return pd.Series(
        session.run(None, inputs)[0].reshape(-1),
        index=data.index
    )


class TestSklearnPipeline(unittest.TestCase):

    @unittest.skipIf(ColumnTransformer is None, reason="too old scikit-learn")
    def test_concat(self):
        data = os.path.join(os.path.dirname(__file__),
                            "datasets", "small_titanic.csv")
        data = pd.read_csv(data)
        data['female'] = data['sex'] == 'female'
        data = data[['age', 'fare', 'female', 'embarked',
                     'pclass', 'survived']]

        for col in data:
            dtype = data[col].dtype
            if dtype in ['float64', 'float32']:
                data[col].fillna(0., inplace=True)
            if dtype in ['int64']:
                data[col].fillna(0, inplace=True)
            elif dtype in ['O']:
                data[col].fillna('N/A', inplace=True)

        full_df = data.drop('survived', axis=1)
        full_labels = data['survived']

        train_df, test_df, train_labels, test_labels = train_test_split(
            full_df, full_labels, test_size=.2, random_state=0)

        col_transformer = _column_tranformer_fitted_from_df(full_df)

        regressor = DecisionTreeRegressor(random_state=0)
        regressor.fit(
            col_transformer.transform(train_df),
            train_labels)
        model = Pipeline(
            steps=[('preprocessor', col_transformer),
                   ('regressor', regressor)])

        initial_types = _convert_dataframe_schema(full_df)
        itypes = set(_[1].__class__ for _ in initial_types)
        self.assertIn(BooleanTensorType, itypes)
        self.assertIn(FloatTensorType, itypes)
        onx = convert_sklearn(model, initial_types=initial_types,
                              target_opset=TARGET_OPSET)

        session = rt.InferenceSession(onx.SerializeToString())

        pred_skl = model.predict(test_df)
        pred_onx = _predict(session, test_df)

        diff = np.sort(
            np.abs(np.squeeze(pred_skl) - np.squeeze(pred_onx)))
        if diff[0] != diff[-1]:
            raise AssertionError(
                "Discrepencies\nSKL\n{}\nORT\n{}".format(pred_skl, pred_onx))


if __name__ == "__main__":
    unittest.main()
