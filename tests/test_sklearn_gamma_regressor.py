# SPDX-License-Identifier: Apache-2.0

"""Tests scikit-learn's SGDClassifier converter."""

import unittest
from matplotlib import projections
import numpy as np
try:
    from sklearn.linear_model import GammaRegressor
except ImportError:
    GammaRegressor = None
from onnxruntime import __version__ as ort_version
from skl2onnx import convert_sklearn

from skl2onnx.common.data_types import (
    FloatTensorType,
)

from test_utils import (
    dump_data_and_model,
    TARGET_OPSET
)

ort_version = ".".join(ort_version.split(".")[:2])


class TestGammaRegressorConverter(unittest.TestCase):
    @unittest.skipIf(GammaRegressor is None,
                     reason="scikit-learn<1.0")
    def test_gamma_regressor(self):

        clf = GammaRegressor()
        X = [[1,2], [2,3], [3,4], [4,3]]
        y = [19, 26, 33, 30]
        clf.fit(X, y)
        print("score=", clf.score(X, y))
        print("coef=", clf.coef_)
        print("intercept=", clf.intercept_)
        test_x = [[1,2], [2,3], [3,4], [4,3], [1,0], [2,8]]
        result = clf.predict(test_x)
        print("predict result = ", result)
        
        import matplotlib.pyplot as plt
        from mpl_toolkits import mplot3d

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        test_X = np.array(test_x)
        ax.scatter(test_X[:,0], test_X[:,1], result)
        plt.show()

        '''
        model_onnx = convert_sklearn(
            model,
            "scikit-learn SGD OneClass SVM",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)

        self.assertIsNotNone(model_onnx)
        dump_data_and_model(test_x.astype(np.float32), model, model_onnx,
                            basename="SklearnSGDOneClassSVMBinaryHinge")
        '''

if __name__ == "__main__":
    unittest.main(verbosity=3)
