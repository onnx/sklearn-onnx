# SPDX-License-Identifier: Apache-2.0

import unittest
import urllib.error as url_error
from distutils.version import StrictVersion
from io import StringIO
import warnings
import numpy
from numpy.testing import assert_almost_equal
import pandas
from sklearn import __version__ as sklearn_version
from sklearn import datasets

try:
    # scikit-learn >= 0.22
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    # scikit-learn < 0.22
    from sklearn.utils.testing import ignore_warnings
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
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    FloatTensorType,
    Int64TensorType,
    StringTensorType,
)
from skl2onnx.common.data_types import onnx_built_with_ml
from test_utils import (
    dump_data_and_model, fit_classification_model, TARGET_OPSET)
from onnxruntime import __version__ as ort_version, InferenceSession


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
                x2[:, i] = inp[keys[i]].ravel()
            res = self.pipe.transform(x2)
            return res
        else:
            raise TypeError("Unable to predict with type {0}".format(
                type(inp)))


class TestSklearnPipeline(unittest.TestCase):

    @ignore_warnings(category=FutureWarning)
    def test_pipeline(self):
        data = numpy.array([[0, 0], [0, 0], [1, 1], [1, 1]],
                           dtype=numpy.float32)
        scaler = StandardScaler()
        scaler.fit(data)
        model = Pipeline([("scaler1", scaler), ("scaler2", scaler)])

        model_onnx = convert_sklearn(model, "pipeline",
                                     [("input", FloatTensorType([None, 2]))],
                                     target_opset=TARGET_OPSET)
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(data, model, model_onnx,
                            basename="SklearnPipelineScaler")

    @ignore_warnings(category=FutureWarning)
    def test_combine_inputs(self):
        data = numpy.array(
            [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]],
            dtype=numpy.float32)
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
            target_opset=TARGET_OPSET)
        self.assertTrue(len(model_onnx.graph.node[-1].output) == 1)
        self.assertTrue(model_onnx is not None)
        data = {
            "input1": data[:, 0].reshape((-1, 1)),
            "input2": data[:, 1].reshape((-1, 1)),
        }
        dump_data_and_model(
            data, PipeConcatenateInput(model),
            model_onnx, basename="SklearnPipelineScaler11")

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion('0.4.0'),
        reason="onnxruntime too old")
    @ignore_warnings(category=FutureWarning)
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
            target_opset=TARGET_OPSET)
        self.assertTrue(len(model_onnx.graph.node[-1].output) == 1)
        self.assertTrue(model_onnx is not None)
        data = {
            "input1": data[:, 0].reshape((-1, 1)),
            "input2": data[:, 1].reshape((-1, 1)),
        }
        dump_data_and_model(
            data, PipeConcatenateInput(model),
            model_onnx, basename="SklearnPipelineScaler11Union")

    @unittest.skipIf(
        StrictVersion(ort_version) <= StrictVersion('0.4.0'),
        reason="onnxruntime too old")
    @ignore_warnings(category=FutureWarning)
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
            target_opset=TARGET_OPSET)
        self.assertTrue(len(model_onnx.graph.node[-1].output) == 1)
        self.assertTrue(model_onnx is not None)
        data = numpy.array(data)
        data = {
            "input1": data[:, 0].reshape((-1, 1)).astype(numpy.int64),
            "input2": data[:, 1].reshape((-1, 1)).astype(numpy.float32),
        }
        dump_data_and_model(
            data, PipeConcatenateInput(model),
            model_onnx, basename="SklearnPipelineScalerMixed")

    @unittest.skipIf(
        ColumnTransformer is None,
        reason="ColumnTransformer not available in 0.19",
    )
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(StrictVersion(ort_version) <= StrictVersion("0.4.0"),
                     reason="issues with shapes")
    @ignore_warnings(category=(RuntimeWarning, FutureWarning))
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
            n_jobs=1, max_iter=10, solver="lbfgs", tol=1e-3)

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
        model_onnx = convert_sklearn(model, initial_types=initial_type,
                                     target_opset=TARGET_OPSET)

        dump_data_and_model(
            X_train, model, model_onnx,
            basename="SklearnPipelineColumnTransformerPipeliner",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.3') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.4.0')")

        if __name__ == "__main__":
            from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer

            pydot_graph = GetPydotGraph(
                model_onnx.graph,
                name=model_onnx.graph.name,
                rankdir="TP",
                node_producer=GetOpNodeProducer("docstring"))
            pydot_graph.write_dot("graph.dot")

            import os

            os.system("dot -O -G=300 -Tpng graph.dot")

    @unittest.skipIf(
        ColumnTransformer is None,
        reason="ColumnTransformer not available in 0.19")
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(
        not check_scikit_version(),
        reason="Scikit 0.20 causes some mismatches")
    @ignore_warnings(category=FutureWarning)
    def test_pipeline_column_transformer_titanic(self):

        # fit
        try:
            titanic_url = (
                "https://raw.githubusercontent.com/amueller/"
                "scipy-2017-sklearn/091d371/notebooks/datasets/titanic3.csv")
            data = pandas.read_csv(titanic_url)
        except url_error.URLError:
            # Do not fail the test if the data cannot be fetched.
            warnings.warn("Unable to fetch titanic data.")
            return
        X = data.drop("survived", axis=1)
        y = data["survived"]

        # SimpleImputer on string is not available for string
        # in ONNX-ML specifications.
        # So we do it beforehand.
        for cat in ["embarked", "sex", "pclass"]:
            X[cat].fillna("missing", inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)

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
            # ("classifier", LogisticRegression(solver="lbfgs")),
        ])

        # inputs

        def convert_dataframe_schema(df, drop=None):
            inputs = []
            for k, v in zip(df.columns, df.dtypes):
                if drop is not None and k in drop:
                    continue
                if v == 'int64':
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

        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train['pclass'] = X_train['pclass'].astype(numpy.int64)
        X_test['pclass'] = X_test['pclass'].astype(numpy.int64)
        X_train = X_train.drop(to_drop, axis=1)
        X_test = X_test.drop(to_drop, axis=1)

        # Step 1: without classifier
        clf.fit(X_train, y_train)
        initial_inputs = convert_dataframe_schema(X_train, to_drop)
        model_onnx = convert_sklearn(clf, "pipeline_titanic", initial_inputs,
                                     target_opset=TARGET_OPSET)

        data = X_test
        pred = clf.transform(data)
        data_types = {
            'pclass': numpy.int64,
            'age': numpy.float32,
            'sex': numpy.str_,
            'fare': numpy.float32,
            'embarked': numpy.str_,
        }
        inputs = {k: data[k].values.astype(data_types[k]).reshape(-1, 1)
                  for k in data.columns}
        sess = InferenceSession(model_onnx.SerializeToString())
        run = sess.run(None, inputs)
        got = run[-1]
        assert_almost_equal(pred, got, decimal=5)

        # Step 2: with classifier
        clf = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(solver="lbfgs")),
        ]).fit(X_train, y_train)
        pred = clf.predict_proba(data)
        model_onnx = convert_sklearn(clf, "pipeline_titanic", initial_inputs,
                                     target_opset=TARGET_OPSET,
                                     options={id(clf): {'zipmap': False}})
        sess = InferenceSession(model_onnx.SerializeToString())
        run = sess.run(None, inputs)
        got = run[-1]
        assert_almost_equal(pred, got, decimal=5)

    @unittest.skipIf(
        ColumnTransformer is None,
        reason="ColumnTransformer not available in 0.19")
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=FutureWarning)
    def test_column_transformer_weights(self):
        model, X = fit_classification_model(
            ColumnTransformer(
                [('pca', PCA(n_components=5), slice(0, 10)),
                 ('svd', TruncatedSVD(n_components=5), slice(10, 100))],
                transformer_weights={'pca': 2, 'svd': 3}), 3, n_features=100)
        model_onnx = convert_sklearn(
            model,
            "column transformer weights",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnColumnTransformerWeights-Dec4",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')")

    @unittest.skipIf(
        ColumnTransformer is None,
        reason="ColumnTransformer not available in 0.19")
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=FutureWarning)
    def test_column_transformer_drop(self):
        model, X = fit_classification_model(
            ColumnTransformer(
                [('pca', PCA(n_components=5), slice(0, 10)),
                 ('svd', TruncatedSVD(n_components=5), slice(80, 100))],
                remainder='drop'), 3, n_features=100)
        model_onnx = convert_sklearn(
            model,
            "column transformer drop",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnColumnTransformerDrop",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')")

    @unittest.skipIf(
        ColumnTransformer is None,
        reason="ColumnTransformer not available in 0.19")
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=FutureWarning)
    def test_column_transformer_passthrough(self):
        model, X = fit_classification_model(
            ColumnTransformer(
                [('pca', PCA(n_components=5), slice(0, 10)),
                 ('svd', TruncatedSVD(n_components=5), slice(80, 100))],
                transformer_weights={'pca': 2, 'svd': 3},
                remainder='passthrough'), 3, n_features=100)
        model_onnx = convert_sklearn(
            model,
            "column transformer passthrough",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnColumnTransformerPassthrough",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')")

    @unittest.skipIf(
        ColumnTransformer is None,
        reason="ColumnTransformer not available in 0.19")
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @ignore_warnings(category=FutureWarning)
    def test_column_transformer_passthrough_no_weights(self):
        model, X = fit_classification_model(
            ColumnTransformer(
                [('pca', PCA(n_components=5), slice(0, 10)),
                 ('svd', TruncatedSVD(n_components=5), slice(70, 80))],
                remainder='passthrough'), 3, n_features=100)
        model_onnx = convert_sklearn(
            model,
            "column transformer passthrough",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnColumnTransformerPassthroughNoWeights",
            allow_failure="StrictVersion(onnxruntime.__version__)"
            "<= StrictVersion('0.2.1')")

    @unittest.skipIf(
        ColumnTransformer is None,
        reason="ColumnTransformer not available in 0.19")
    @ignore_warnings(category=FutureWarning)
    def test_pipeline_dataframe(self):
        text = """
                fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol,quality,color
                7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,red
                7.8,0.88,0.0,2.6,0.098,25.0,67.0,0.9968,3.2,0.68,9.8,5,red
                7.8,0.76,0.04,2.3,0.092,15.0,54.0,0.997,3.26,0.65,9.8,5,red
                11.2,0.28,0.56,1.9,0.075,17.0,60.0,0.998,3.16,0.58,9.8,6,red
                """.replace("                ", "")
        X_train = pandas.read_csv(StringIO(text))
        for c in X_train.columns:
            if c != 'color':
                X_train[c] = X_train[c].astype(numpy.float32)
        numeric_features = [c for c in X_train if c != 'color']

        pipe = Pipeline([
            ("prep", ColumnTransformer([
                ("color", Pipeline([
                    ('one', OneHotEncoder()),
                    ('select', ColumnTransformer(
                        [('sel1', 'passthrough', [0])]))
                ]), ['color']),
                ("others", "passthrough", numeric_features)
            ])),
        ])

        init_types = [
            ('fixed_acidity', FloatTensorType(shape=[None, 1])),
            ('volatile_acidity', FloatTensorType(shape=[None, 1])),
            ('citric_acid', FloatTensorType(shape=[None, 1])),
            ('residual_sugar', FloatTensorType(shape=[None, 1])),
            ('chlorides', FloatTensorType(shape=[None, 1])),
            ('free_sulfur_dioxide', FloatTensorType(shape=[None, 1])),
            ('total_sulfur_dioxide', FloatTensorType(shape=[None, 1])),
            ('density', FloatTensorType(shape=[None, 1])),
            ('pH', FloatTensorType(shape=[None, 1])),
            ('sulphates', FloatTensorType(shape=[None, 1])),
            ('alcohol', FloatTensorType(shape=[None, 1])),
            ('quality', FloatTensorType(shape=[None, 1])),
            ('color', StringTensorType(shape=[None, 1]))
        ]

        pipe.fit(X_train)
        model_onnx = convert_sklearn(
            pipe, initial_types=init_types, target_opset=TARGET_OPSET)
        oinf = InferenceSession(model_onnx.SerializeToString())

        pred = pipe.transform(X_train)
        inputs = {c: X_train[c].values for c in X_train.columns}
        inputs = {c: v.reshape((v.shape[0], 1)) for c, v in inputs.items()}
        onxp = oinf.run(None, inputs)
        got = onxp[0]
        assert_almost_equal(pred, got)

    @ignore_warnings(category=(FutureWarning, UserWarning))
    def test_pipeline_tfidf_svc(self):
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf_svc', SVC(probability=True, kernel='linear'))])
        data = numpy.array(["first sentance", "second sentence",
                            "many sentances", "dummy sentance",
                            "no sentance at all"])
        y = numpy.array([0, 0, 1, 0, 1])
        pipe.fit(data, y)
        expected_label = pipe.predict(data)
        expected_proba = pipe.predict_proba(data)
        df = pandas.DataFrame(data)
        df.columns = ['text']

        # first conversion if shape=[None, 1]
        model_onnx = convert_sklearn(
            pipe, initial_types=[('text', StringTensorType([None, 1]))],
            target_opset=TARGET_OPSET,
            options={id(pipe): {'zipmap': False}})
        sess = InferenceSession(model_onnx.SerializeToString())
        got = sess.run(None, {'text': data.reshape((-1, 1))})
        assert_almost_equal(expected_proba, got[1])
        assert_almost_equal(expected_label, got[0])
        # sess.run(None, {'text': df}) --> failures
        # sess.run(None, {'text': df["text"]}) --> failures

        # second conversion with shape=[None]
        model_onnx = convert_sklearn(
            pipe, initial_types=[('text', StringTensorType([None]))],
            target_opset=TARGET_OPSET,
            options={id(pipe): {'zipmap': False}})
        sess = InferenceSession(model_onnx.SerializeToString())
        got = sess.run(None, {'text': data})
        assert_almost_equal(expected_proba, got[1])
        assert_almost_equal(expected_label, got[0])
        # sess.run(None, {'text': df})  failure
        # sess.run(None, {'text': df["text"]})  failure
        sess.run(None, {'text': df["text"].values})  # success

    @ignore_warnings(category=(FutureWarning, UserWarning))
    def test_pipeline_voting_tfidf_svc(self):
        pipe1 = Pipeline([
            ('tfidf1', TfidfVectorizer()),
            ('svc', SVC(probability=True, kernel='linear'))])
        pipe2 = Pipeline([
            ('tfidf2', TfidfVectorizer(norm='l2', use_idf=False)),
            ('sgd', SGDClassifier(alpha=0.0001, penalty='l2',
                                  loss='modified_huber'))])
        pipe3 = Pipeline([
            ('tfidf3', TfidfVectorizer()),
            ('mnb', MultinomialNB())])
        voting = VotingClassifier(
            [('p1', pipe1), ('p2', pipe2), ('p3', pipe3)],
            voting='soft', flatten_transform=False)
        data = numpy.array(["first sentance", "second sentence",
                            "many sentances", "dummy sentance",
                            "no sentance at all"])
        y = numpy.array([0, 0, 1, 0, 1])
        voting.fit(data, y)
        expected_label = voting.predict(data)
        expected_proba = voting.predict_proba(data)
        df = pandas.DataFrame(data)
        df.columns = ['text']

        # first conversion if shape=[None, 1]
        model_onnx = convert_sklearn(
            voting, initial_types=[('text', StringTensorType([None, 1]))],
            target_opset=TARGET_OPSET,
            options={id(voting): {'zipmap': False}})
        sess = InferenceSession(model_onnx.SerializeToString())
        got = sess.run(None, {'text': data.reshape((-1, 1))})
        assert_almost_equal(expected_proba, got[1])
        assert_almost_equal(expected_label, got[0])


if __name__ == "__main__":
    unittest.main()
