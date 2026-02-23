<!--- SPDX-License-Identifier: Apache-2.0 -->

<p align="center"><img width="50%" src="https://onnx.ai/sklearn-onnx/_static/logo_main.png" /></p>

[![PyPI - Version](https://img.shields.io/pypi/v/skl2onnx.svg)](https://pypi.org/project/skl2onnx)
[![Linux](https://github.com/onnx/sklearn-onnx/actions/workflows/linux-ci.yml/badge.svg)](https://github.com/onnx/sklearn-onnx/actions/workflows/linux-ci.yml)
[![Windows/Macos](https://github.com/onnx/sklearn-onnx/actions/workflows/windows-macos-ci.yml/badge.svg)](https://github.com/onnx/sklearn-onnx/actions/workflows/windows-macos-ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Introduction
*sklearn-onnx* converts [scikit-learn](https://scikit-learn.org/stable/) models to [ONNX](https://github.com/onnx/onnx).
Once in the ONNX format, you can use tools like [ONNX Runtime](https://github.com/Microsoft/onnxruntime) for high performance scoring.
All converters are tested with [onnxruntime](https://onnxruntime.ai/).
Any external converter can be registered to convert scikit-learn pipeline
including models or transformers coming from external libraries.

## Documentation
Full documentation including tutorials is available at [https://onnx.ai/sklearn-onnx/](https://onnx.ai/sklearn-onnx/).
[Supported scikit-learn Models](https://onnx.ai/sklearn-onnx/supported.html)
Last supported opset is 21.

You may also find answers in [existing issues](https://github.com/onnx/sklearn-onnx/issues?utf8=%E2%9C%93&q=is%3Aissue)
or submit a new one.

## Installation
You can install from [PyPi](https://pypi.org/project/skl2onnx/):
```
pip install skl2onnx
```
Or you can install from the source with the latest changes.
```
pip install git+https://github.com/onnx/sklearn-onnx.git
```

## Getting started

```python
# Train a model.
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
X, y = iris.data, iris.target
X = X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y)
clr = RandomForestClassifier()
clr.fit(X_train, y_train)

# Convert into ONNX format.
from skl2onnx import to_onnx

onx = to_onnx(clr, X[:1])
with open("rf_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())

# Compute the prediction with onnxruntime.
import onnxruntime as rt

sess = rt.InferenceSession("rf_iris.onnx", providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]
```

## Contribute
We welcome contributions in the form of feedback, ideas, or code.

## PR
Before you submit any PR, you should apply the following command lines
to fix the style issues.

```bash
black .
ruff check .
```

## License
[Apache License v2.0](LICENSE)
