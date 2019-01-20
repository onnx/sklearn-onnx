import unittest
import numpy
import pandas
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType, StringTensorType
from test_utils import dump_data_and_model


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
            raise TypeError("Unable to predict with type {0}".format(type(inp)))


class TestSklearnPipeline(unittest.TestCase):

    def test_pipeline(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        data = numpy.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=numpy.float32)
        scaler = StandardScaler()
        scaler.fit(data)
        model = Pipeline([('scaler1',scaler), ('scaler2', scaler)])

        model_onnx = convert_sklearn(model, 'pipeline', [('input', FloatTensorType([1, 2]))])
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(data, model, model_onnx, basename="SklearnPipelineScaler")

    def test_combine_inputs(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        data = numpy.array([[0., 0.], [0., 0.], [1., 1.], [1., 1.]], dtype=numpy.float32)
        scaler = StandardScaler()
        scaler.fit(data)
        model = Pipeline([('scaler1', scaler), ('scaler2', scaler)])

        model_onnx = convert_sklearn(model, 'pipeline',
                                     [('input1', FloatTensorType([1, 1])),
                                      ('input2', FloatTensorType([1, 1]))])
        self.assertTrue(len(model_onnx.graph.node[-1].output) == 1)
        self.assertTrue(model_onnx is not None)
        data = {'input1': data[:, 0], 'input2': data[:, 1]}
        dump_data_and_model(data, PipeConcatenateInput(model), model_onnx,
                            basename="SklearnPipelineScaler11-OneOff")

    def test_combine_inputs_floats_ints(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        data = [[0, 0.],[0, 0.],[1, 1.],[1, 1.]]
        scaler = StandardScaler()
        scaler.fit(data)
        model = Pipeline([('scaler1', scaler), ('scaler2', scaler)])

        model_onnx = convert_sklearn(model, 'pipeline',
                                     [('input1', Int64TensorType([1, 1])),
                                      ('input2', FloatTensorType([1, 1]))])
        self.assertTrue(len(model_onnx.graph.node[-1].output) == 1)
        self.assertTrue(model_onnx is not None)
        data = numpy.array(data)
        data = {'input1': data[:, 0].astype(numpy.int64), 
                'input2': data[:, 1].astype(numpy.float32)}
        dump_data_and_model(data, PipeConcatenateInput(model), model_onnx,
                            basename="SklearnPipelineScalerMixed-OneOff")
    
    def test_pipeline_column_transformer(self):
        
        iris = datasets.load_iris()
        X = iris.data[:, :3]
        y = iris.target
        X_train = pandas.DataFrame(X, columns=["vA", "vB", "vC"])
        X_train["vcat"] = X_train["vA"].apply(lambda x: "cat1" if x > 0.5 else "cat2")
        X_train["vcat2"] = X_train["vB"].apply(lambda x: "cat3" if x > 0.5 else "cat4")
        y_train = y % 2
        numeric_features = [0, 1, 2] # ["vA", "vB", "vC"]
        categorical_features = [3, 4] # ["vcat", "vcat2"]
        
        classifier = LogisticRegression(C=0.01, class_weight=dict(zip([False, True], [0.2, 0.8])),
                                        n_jobs=1, max_iter=10, solver='lbfgs', tol=1e-3)

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(sparse=True, handle_unknown='ignore')),
            ('tsvd', TruncatedSVD(n_components=1, algorithm='arpack', tol=1e-4))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        model = Pipeline(steps=[
            ('precprocessor', preprocessor),
            ('classifier', classifier)
        ])

        model.fit(X_train, y_train)
        initial_type = [('numfeat', FloatTensorType([1, 3])),
                        ('strfeat', StringTensorType([1, 2]))]

        X_train = X_train[:11]
        model_onnx = convert_sklearn(model, initial_types=initial_type)
        
        dump_data_and_model(X_train, model, model_onnx,
                            basename="SklearnPipelineColumnTransformerPipeliner",
                            allow_failure="StrictVersion(onnx.__version__) < StrictVersion('1.2')")


if __name__ == "__main__":
    unittest.main()
