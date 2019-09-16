import unittest
from distutils.version import StrictVersion
import numpy
import pandas
from sklearn import __version__ as sklearn_version
from sklearn import datasets

try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    # not available in 0.19
    ColumnTransformer = None
from sklearn.decomposition import PCA, TruncatedSVD

try:
    from sklearn.impute import SimpleImputer
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    FloatTensorType,
    Int64TensorType,
    StringTensorType,
)
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import dump_data_and_model, fit_classification_model
from onnxruntime import __version__ as ort_version


def check_scikit_version():
    # StrictVersion does not work with development versions
    vers = '.'.join(sklearn_version.split('.')[:2])
    return StrictVersion(vers) >= StrictVersion("0.21.0")


class PipeConcatenateInput:
    def __init__(self, pipe):
        self.pipe = pipe

    def transform(self, inp):
        if isinstance(inp, (numpy.ndarray, pandas.DataFrame)):
            return self.pipe.transform(inp)
        elif isinstance(inp, dict):
            keys = list(sorted(inp.keys()))
            dim = inp[keys[0]].shape[0], len(keys)
            x2 = numpy.zeros(dim)
            for i in range(x2.shape[1]):
                x2[:, i] = inp[keys[i]]
            res = self.pipe.transform(x2)
            return res
        else:
            raise TypeError("Unable to predict with type {0}".format(
                type(inp)))


class TestSklearnPipeline(unittest.TestCase):
    def test_pipeline(self):
        data = numpy.array([[0, 0], [0, 0], [1, 1], [1, 1]],
                           dtype=numpy.float32)
        scaler = StandardScaler()
        scaler.fit(data)
        model = Pipeline([("scaler1", scaler), ("scaler2", scaler)])

        model_onnx = convert_sklearn(model, "pipeline",
                                     [("input", FloatTensorType([None, 2]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(data,
                            model,
                            model_onnx,
                            basename="SklearnPipelineScaler")

    def test_combine_inputs(self):
        data = numpy.array(
            [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]],
            dtype=numpy.float32,
        )
        scaler = StandardScaler()
        scaler.fit(data)
        model = Pipeline([("scaler1", scaler), ("scaler2", scaler)])

        model_onnx = convert_sklearn(
            model,
            "pipeline",
            [
                ("input1", FloatTensorType([None, 1])),
                ("input2", FloatTensorType([None, 1])),
            ],
        )
        self.assertTrue(len(model_onnx.graph.node[-1].output) == 1)
        self.assertTrue(model_onnx is not None)
        data = {"input1": data[:, 0], "input2": data[:, 1]}
        dump_data_and_model(
            data,
            PipeConcatenateInput(model),
            model_onnx,
            basename="SklearnPipelineScaler11-OneOff",
        )

    def test_combine_inputs_union_in_pipeline(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        data = numpy.array(
            [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]],
            dtype=numpy.float32,
        )
        model = Pipeline([
            ("scaler1", StandardScaler()),
            (
                "union",
                FeatureUnion([
                    ("scaler2", StandardScaler()),
                    ("scaler3", MinMaxScaler()),
                ]),
            ),
        ])
        model.fit(data)
        model_onnx = convert_sklearn(
            model,
            "pipeline",
            [
                ("input1", FloatTensorType([None, 1])),
                ("input2", FloatTensorType([None, 1])),
            ],
        )
        self.assertTrue(len(model_onnx.graph.node[-1].output) == 1)
        self.assertTrue(model_onnx is not None)
        data = {"input1": data[:, 0], "input2": data[:, 1]}
        dump_data_and_model(
            data,
            PipeConcatenateInput(model),
            model_onnx,
            basename="SklearnPipelineScaler11Union-OneOff",
        )

    def test_combine_inputs_floats_ints(self):
        data = [[0, 0.0], [0, 0.0], [1, 1.0], [1, 1.0]]
        scaler = StandardScaler()
        scaler.fit(data)
        model = Pipeline([("scaler1", scaler), ("scaler2", scaler)])

        model_onnx = convert_sklearn(
            model,
            "pipeline",
            [
                ("input1", Int64TensorType([None, 1])),
                ("input2", FloatTensorType([None, 1])),
            ],
        )
        self.assertTrue(len(model_onnx.graph.node[-1].output) == 1)
        self.assertTrue(model_onnx is not None)
        data = numpy.array(data)
        data = {
            "input1": data[:, 0].astype(numpy.int64),
            "input2": data[:, 1].astype(numpy.float32),
        }
        dump_data_and_model(
            data,
            PipeConcatenateInput(model),
            model_onnx,
            basename="SklearnPipelineScalerMixed-OneOff",
        )

    @unittest.skipIf(
        ColumnTransformer is None,
        reason="ColumnTransformer not available in 0.19",
    )
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(StrictVersion(ort_version) <= StrictVersion("0.4.0"),
                     reason="issues with shapes")
    def test_pipeline_column_transformer(self):

        iris = datasets.load_iris()
        X = iris.data[:, :3]
        y = iris.target
        X_train = pandas.DataFrame(X, columns=["vA", "vB", "vC"])
        X_train["vcat"] = X_train["vA"].apply(lambda x: "cat1"
                                              if x > 0.5 else "cat2")
        X_train["vcat2"] = X_train["vB"].apply(lambda x: "cat3"
                                               if x > 0.5 else "cat4")
        y_train = y % 2
        numeric_features = [0, 1, 2]  # ["vA", "vB", "vC"]
        categorical_features = [3, 4]  # ["vcat", "vcat2"]

        classifier = LogisticRegression(
            C=0.01,
            class_weight=dict(zip([False, True], [0.2, 0.8])),
            n_jobs=1,
            max_iter=10,
            solver="lbfgs",
            tol=1e-3,
        )

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_transformer = Pipeline(steps=[
            (
                "onehot",
                OneHotEncoder(sparse=True, handle_unknown="ignore"),
            ),
            (
                "tsvd",
                TruncatedSVD(n_components=1, algorithm="arpack", tol=1e-4),
            ),
        ])

        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ])

        model = Pipeline(steps=[("precprocessor",
                                 preprocessor), ("classifier", classifier)])

        model.fit(X_train, y_train)
        initial_type = [
            ("numfeat", FloatTensorType([None, 3])),
            ("strfeat", StringTensorType([None, 2])),
        ]

        X_train = X_train[:11]
        model_onnx = convert_sklearn(model, initial_types=initial_type)

        dump_data_and_model(
            X_train,
            model,
            model_onnx,
            basename="SklearnPipelineColumnTransformerPipeliner",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.3') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.4.0')",
        )

        if __name__ == "__main__":
            from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer

            pydot_graph = GetPydotGraph(
                model_onnx.graph,
                name=model_onnx.graph.name,
                rankdir="TP",
                node_producer=GetOpNodeProducer("docstring"),
            )
            pydot_graph.write_dot("graph.dot")

            import os

            os.system("dot -O -G=300 -Tpng graph.dot")

    @unittest.skipIf(
        ColumnTransformer is None,
        reason="ColumnTransformer not available in 0.19",
    )
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(
        not check_scikit_version(),
        reason="Scikit 0.20 causes some mismatches",
    )
    def test_pipeline_column_transformer_titanic(self):

        # fit
        titanic_url = (
            "https://raw.githubusercontent.com/amueller/"
            "scipy-2017-sklearn/091d371/notebooks/datasets/titanic3.csv")
        data = pandas.read_csv(titanic_url)
        X = data.drop("survived", axis=1)
        y = data["survived"]

        # SimpleImputer on string is not available for string
        # in ONNX-ML specifications.
        # So we do it beforehand.
        for cat in ["embarked", "sex", "pclass"]:
            X[cat].fillna("missing", inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2)

        numeric_features = ["age", "fare"]
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_features = ["embarked", "sex", "pclass"]
        categorical_transformer = Pipeline(steps=[
            # --- SimpleImputer on string is not available
            # for string in ONNX-ML specifications.
            # ('imputer',
            #  SimpleImputer(strategy='constant', fill_value='missing')),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ])

        clf = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(solver="lbfgs")),
        ])

        # inputs

        def convert_dataframe_schema(df, drop=None):
            inputs = []
            for k, v in zip(df.columns, df.dtypes):
                if drop is not None and k in drop:
                    continue
                if v == "int64":
                    t = Int64TensorType([None, 1])
                elif v == "float64":
                    t = FloatTensorType([None, 1])
                else:
                    t = StringTensorType([None, 1])
                inputs.append((k, t))
            return inputs

        to_drop = {
            "parch",
            "sibsp",
            "cabin",
            "ticket",
            "name",
            "body",
            "home.dest",
            "boat",
        }
        X_train = X_train.drop(to_drop, axis=1)
        X_test = X_test.drop(to_drop, axis=1)
        clf.fit(X_train, y_train)
        X_train["pclass"] = X_train["pclass"].astype(str)
        X_test["pclass"] = X_test["pclass"].astype(str)
        inputs = convert_dataframe_schema(X_train, to_drop)
        model_onnx = convert_sklearn(clf, "pipeline_titanic", inputs)

        dump_data_and_model(
            X_test[:5],
            clf,
            model_onnx,
            basename="SklearnPipelineColumnTransformerPipelinerTitanic-DF",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.3') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(
        ColumnTransformer is None,
        reason="ColumnTransformer not available in 0.19",
    )
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_column_transformer_weights(self):
        model, X = fit_classification_model(
            ColumnTransformer(
                [('pca', PCA(n_components=5), slice(0, 10)),
                 ('svd', TruncatedSVD(n_components=5), slice(10, 100))],
                transformer_weights={'pca': 2, 'svd': 3}), 3)
        model_onnx = convert_sklearn(
            model,
            "column transformer weights",
            [("input", FloatTensorType([None, X.shape[1]]))],
            dtype=numpy.float32,
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnColumnTransformerWeights",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')",
        )


if __name__ == "__main__":
    unittest.main()
