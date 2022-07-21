

from sklearn.linear_model import GammaRegressor
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    FloatTensorType,
)

from tests.test_utils import (
    dump_data_and_model,
    TARGET_OPSET
)

def test_gamma_regressor():

    model = GammaRegressor()
    X = np.array([[1,2], [2,3], [3,4], [4,3]])
    y = np.array([19, 26, 33, 30])
    model.fit(X, y)
    print("score=", model.score(X, y))
    print("coef=", model.coef_)
    print("intercept=", model.intercept_)
    test_x = np.array([[1,0], [2,8]])
    result = model.predict(test_x)
    print("predict result = ", result)
    
    model_onnx = convert_sklearn(
        model,
        "scikit-learn Gamma Regressor",
        [("input", FloatTensorType([None, X.shape[1]]))],
        target_opset=TARGET_OPSET)

    #self.assertIsNotNone(model_onnx)
    dump_data_and_model(test_x.astype(np.float32), model, model_onnx,
                        basename="SklearnGammaRegressor")

if __name__=="__main__":
    test_gamma_regressor()