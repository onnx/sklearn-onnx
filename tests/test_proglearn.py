import unittest
import numpy as np
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from numpy.testing import assert_almost_equal
import packaging.version as pv

from onnxruntime import InferenceSession
from onnxruntime import __version__ as ort_version

try:
    from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument
except ImportError:
    # onnxruntime <= 0.5
    InvalidArgument = RuntimeError

from proglearn.transformers import TreeClassificationTransformer
from proglearn.deciders import SimpleArgmaxAverage
from proglearn.voters import TreeClassificationVoter
from proglearn.progressive_learner import ClassificationProgressiveLearner

from skl2onnx import update_registered_converter
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from skl2onnx.helpers.dict_wrapper import DictWrapper
from skl2onnx.shape_calculators.proglearn import dict_shape_calculator
from skl2onnx.operator_converters.proglearn import dict_custom_converter
from skl2onnx.shape_calculators.proglearn import prog_transformer_shape_calculator
from skl2onnx.operator_converters.proglearn import (
    prog_transformer_converter,
    progresive_parser,
)

# set the seed
np.random.seed(42)

# set the constants
TARGET_OPSET = 14  # minimum working opset version
TARGET_OPSET_ML = 2


def create_data(n_samples, n_features, n_classes):
    """
    Generate a synthetic dataset and split it into training and testing sets.

    Parameters
    ----------
    n_samples : int
        The total number of samples in the synthetic dataset.
    n_features : int
        The total number of features in the synthetic dataset.
    n_classes : int
        The number of class labels in the synthetic dataset.

    Returns
    -------
    X_train : np.ndarray
        The training data, an array of shape (n_samples * 0.7, n_features).
    X_test : np.ndarray
        The testing data, an array of shape (n_samples * 0.3, n_features).
    y_train : np.ndarray
        The training labels, an array of shape (n_samples * 0.7,).
    y_test : np.ndarray
        The testing labels, an array of shape (n_samples * 0.3,).

    Notes
    -----
    The function uses `sklearn.datasets.make_classification` to generate a synthetic dataset and
    `sklearn.model_selection.train_test_split` to split the dataset into training and testing sets.
    """
    n_samples = n_samples
    n_features = n_features
    n_classes = n_classes

    # Generate the synthetic dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_clusters_per_class=1,
        n_redundant=0,
        n_informative=n_features,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    return X_train, X_test, y_train, y_test


def create_model(X_train, y_train, tasks, num_transfs, task_id=0, kappa=None, **kwargs):
    """
    Create a ClassificationProgressiveLearner model and add tasks to it.

    Parameters
    ----------
    X_train : np.ndarray
        The training data.
    y_train : np.ndarray
        The training labels.
    tasks : int
        The number of tasks to add to the model.
    num_transfs : int
        The number of transformers to add for each task.
    task_id : int, optional
        The id of the task. Default is 0.
    kappa : float, optional
        The kappa value for the TreeClassificationVoter. If not provided, no kappa value is used.
    **kwargs : dict, optional
        Additional keyword arguments for the TreeClassificationTransformer.

    Returns
    -------
    model : ClassificationProgressiveLearner
        The created model with tasks added.

    Raises
    ------
    ValueError
        If the provided task_id is not in the model's task_ids.

    Notes
    -----
    The function uses `progressive_learner.ClassificationProgressiveLearner` to create a model and
    `progressive_learner.ClassificationProgressiveLearner.add_task` to add tasks to the model.
    """
    defaults_kwargs = {"random_state": 42}
    default_transformer_class = TreeClassificationTransformer
    default_transformer_kwargs = {"kwargs": defaults_kwargs if not kwargs else kwargs}
    default_voter_class = TreeClassificationVoter
    default_voter_kwargs = dict() if not kappa else {"kappa": kappa}
    default_decider_class = SimpleArgmaxAverage

    model = ClassificationProgressiveLearner(
        default_transformer_class=default_transformer_class,
        default_transformer_kwargs=default_transformer_kwargs,
        default_voter_class=default_voter_class,
        default_voter_kwargs=default_voter_kwargs,
        default_decider_class=default_decider_class,
        default_decider_kwargs={"classes": np.unique(y_train)},
    )

    for i in range(tasks):
        model.add_task(
            X=X_train,
            y=y_train,
            num_transformers=num_transfs,
            transformer_voter_decider_split=[0.67, 0.33, 0],
            decider_kwargs={"classes": np.unique(y_train)},
        )

    if task_id not in model.get_task_ids():
        raise ValueError("Invalid task_id: %d" % task_id)

    model.task_id = task_id
    return model


class TestProgressiveLearnerClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        update_registered_converter(
            DictWrapper,
            "DictWrapper",
            dict_shape_calculator,
            dict_custom_converter,
            overwrite=True,
        )

        update_registered_converter(
            ClassificationProgressiveLearner,
            "ClassificationProgressiveLearnerTransformer",
            prog_transformer_shape_calculator,
            prog_transformer_converter,
            parser=progresive_parser,
            overwrite=True,
        )

    @unittest.skipIf(
        pv.Version(ort_version.split("+")[0]) < pv.Version("1.8.0"),
        "ONNX runtime >= 1.8.0 is required for TARGET_OPSET >= 14",
    )
    def setUp(self):
        return super().setUp()

    def test_progLearn_binary_dt(self):
        KWARGS = {
            "criterion": "entropy",
            "splitter": "random",
            "max_depth": 16,
            "max_features": "log2",
            "random_state": 42,
            "class_weight": "balanced",
        }
        # create data
        X_train, X_test, y_train, _ = create_data(
            n_samples=100, n_features=5, n_classes=2
        )

        # create model
        model = create_model(
            X_train=X_train, y_train=y_train, tasks=2, num_transfs=2, **KWARGS
        )

        n_features = model.task_id_to_X[0].shape[-1]
        model_onnx = convert_sklearn(
            model,
            initial_types=[("float_input", FloatTensorType([None, n_features]))],
            target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML},
            verbose=0,
        )

        self.assertTrue(model_onnx is not None)
        try:
            sess = InferenceSession(
                model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        except InvalidArgument as e:
            raise AssertionError("Cannot load model\n%r" % str(model_onnx)) from e

        expected = model.predict(X_test, task_id=model.task_id)
        expected_proba = model.predict_proba(X_test, task_id=model.task_id)
        res = sess.run(None, {"float_input": X_test})
        self.assertEqual(expected.tolist(), res[0].tolist())
        assert_almost_equal(expected_proba, res[1])

    def test_progLearn_binary(self):
        # create data
        X_train, X_test, y_train, _ = create_data(
            n_samples=100, n_features=5, n_classes=2
        )

        # create model
        model = create_model(X_train=X_train, y_train=y_train, tasks=2, num_transfs=2)

        n_features = model.task_id_to_X[0].shape[-1]
        model_onnx = convert_sklearn(
            model,
            initial_types=[("float_input", FloatTensorType([None, n_features]))],
            target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML},
            verbose=0,
        )

        self.assertTrue(model_onnx is not None)
        try:
            sess = InferenceSession(
                model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        except InvalidArgument as e:
            raise AssertionError("Cannot load model\n%r" % str(model_onnx)) from e

        expected = model.predict(X_test, task_id=model.task_id)
        expected_proba = model.predict_proba(X_test, task_id=model.task_id)
        res = sess.run(None, {"float_input": X_test})
        self.assertEqual(expected.tolist(), res[0].tolist())
        assert_almost_equal(expected_proba, res[1])

    def test_progLearn_binary_kappa(self):
        KAPPA = 0.5
        # create data
        X_train, X_test, y_train, _ = create_data(
            n_samples=100, n_features=5, n_classes=2
        )

        # create model
        model = create_model(
            X_train=X_train, y_train=y_train, tasks=2, num_transfs=2, kappa=KAPPA
        )

        n_features = model.task_id_to_X[0].shape[-1]
        model_onnx = convert_sklearn(
            model,
            initial_types=[("float_input", FloatTensorType([None, n_features]))],
            target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML},
            verbose=0,
        )

        self.assertTrue(model_onnx is not None)
        try:
            sess = InferenceSession(
                model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        except InvalidArgument as e:
            raise AssertionError("Cannot load model\n%r" % str(model_onnx)) from e

        expected = model.predict(X_test, task_id=model.task_id)
        expected_proba = model.predict_proba(X_test, task_id=model.task_id)
        res = sess.run(None, {"float_input": X_test})
        self.assertEqual(expected.tolist(), res[0].tolist())
        assert_almost_equal(expected_proba, res[1])

    def test_progLearn_binary_single_taskId(self):
        TASK_ID = 1

        # create data
        X_train, X_test, y_train, _ = create_data(
            n_samples=100, n_features=5, n_classes=2
        )

        # create model
        model = create_model(
            X_train=X_train, y_train=y_train, tasks=2, num_transfs=2, task_id=TASK_ID
        )

        n_features = model.task_id_to_X[0].shape[-1]
        model_onnx = convert_sklearn(
            model,
            initial_types=[("float_input", FloatTensorType([None, n_features]))],
            target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML},
            verbose=0,
        )

        self.assertTrue(model_onnx is not None)
        try:
            sess = InferenceSession(
                model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        except InvalidArgument as e:
            raise AssertionError("Cannot load model\n%r" % str(model_onnx)) from e

        expected = model.predict(X_test, task_id=TASK_ID)
        expected_proba = model.predict_proba(X_test, task_id=TASK_ID)
        res = sess.run(None, {"float_input": X_test})
        self.assertEqual(expected.tolist(), res[0].tolist())
        assert_almost_equal(expected_proba, res[1])

    def test_progLearn_iris(self):
        X, y = load_iris(return_X_y=True)
        X = X.astype(np.float32)

        model = create_model(X_train=X, y_train=y, tasks=4, num_transfs=4)

        model_onnx = convert_sklearn(
            model,
            initial_types=[("float_input", FloatTensorType([None, X.shape[-1]]))],
            target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML},
            verbose=0,
        )

        self.assertTrue(model_onnx is not None)
        try:
            sess = InferenceSession(
                model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        except InvalidArgument as e:
            raise AssertionError("Cannot load model\n%r" % str(model_onnx)) from e

        res = sess.run(None, {"float_input": X})
        self.assertEqual(
            model.predict(X, task_id=model.task_id).tolist(), res[0].tolist()
        )
        assert_almost_equal(model.predict_proba(X, task_id=model.task_id), res[1])

    def test_progLearn_binary_all_taskIds(self):
        # create data
        X_train, X_test, y_train, _ = create_data(
            n_samples=100, n_features=5, n_classes=2
        )

        # create model w/ 5 task ids
        model = create_model(X_train=X_train, y_train=y_train, tasks=5, num_transfs=3)
        n_features = model.task_id_to_X[0].shape[-1]

        for i in model.get_task_ids():
            with self.subTest(f"Error when running for task_id={i}", i=i):
                model.task_id = i  # set the task_id
                model_onnx = convert_sklearn(
                    model,
                    initial_types=[
                        ("float_input", FloatTensorType([None, n_features]))
                    ],
                    target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML},
                    verbose=0,
                )

                self.assertTrue(model_onnx is not None)
                try:
                    sess = InferenceSession(
                        model_onnx.SerializeToString(),
                        providers=["CPUExecutionProvider"],
                    )
                except InvalidArgument as e:
                    raise AssertionError(
                        "Cannot load model\n%r" % str(model_onnx)
                    ) from e

                res = sess.run(None, {"float_input": X_test})
                self.assertEqual(
                    model.predict(X_test, task_id=i).tolist(), res[0].tolist()
                )
                assert_almost_equal(model.predict_proba(X_test, task_id=i), res[1])

    def test_progLearn_multi(self):
        # create data
        X_train, X_test, y_train, _ = create_data(
            n_samples=300, n_features=5, n_classes=4
        )

        # create model
        model = create_model(X_train=X_train, y_train=y_train, tasks=4, num_transfs=3)

        n_features = model.task_id_to_X[0].shape[-1]
        model_onnx = convert_sklearn(
            model,
            initial_types=[("float_input", FloatTensorType([None, n_features]))],
            target_opset={"": TARGET_OPSET, "ai.onnx.ml": TARGET_OPSET_ML},
            verbose=0,
        )

        self.assertTrue(model_onnx is not None)
        try:
            sess = InferenceSession(
                model_onnx.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        except InvalidArgument as e:
            raise AssertionError("Cannot load model\n%r" % str(model_onnx)) from e

        expected = model.predict(X_test, task_id=model.task_id)
        expected_proba = model.predict_proba(X_test, task_id=model.task_id)
        res = sess.run(None, {"float_input": X_test})
        self.assertEqual(expected.tolist(), res[0].tolist())
        assert_almost_equal(expected_proba, res[1])


if __name__ == "__main__":
    unittest.main(verbosity=2, catchbreak=True)
