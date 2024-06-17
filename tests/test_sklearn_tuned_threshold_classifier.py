# SPDX-License-Identifier: Apache-2.0

import unittest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils._testing import ignore_warnings
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from test_utils import dump_data_and_model, TARGET_OPSET


def has_tuned_theshold_classifier():
    try:
        from sklearn.model_selection import TunedThresholdClassifierCV  # noqa: F401
    except ImportError:
        return False
    return True


class TestSklearnTunedThresholdClassifierConverter(unittest.TestCase):
    @unittest.skipIf(
        not has_tuned_theshold_classifier(),
        reason="TunedThresholdClassifierCV not available",
    )
    @ignore_warnings(category=FutureWarning)
    def test_tuned_threshold_classifier(self):
        from sklearn.model_selection import TunedThresholdClassifierCV

        X, y = make_classification(
            n_samples=1_000, weights=[0.9, 0.1], class_sep=0.8, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=42
        )
        classifier = RandomForestClassifier(random_state=0)

        classifier_tuned = TunedThresholdClassifierCV(
            classifier, scoring="balanced_accuracy"
        ).fit(X_train, y_train)

        model_onnx = to_onnx(
            classifier_tuned,
            initial_types=[("X", FloatTensorType([None, X_train.shape[1]]))],
            target_opset=TARGET_OPSET - 1,
            options={"zipmap": False},
        )
        self.assertTrue(model_onnx is not None)
        dump_data_and_model(
            X_test[:10].astype(np.float32),
            classifier_tuned,
            model_onnx,
            basename="SklearnTunedThresholdClassifier",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
