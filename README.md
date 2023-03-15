<!--- SPDX-License-Identifier: Apache-2.0 -->

<p align="center"><img width="50%" src="docs/logo_main.png" /></p>

| Linux | Windows |
|-------|---------|
| [![Build Status](https://dev.azure.com/onnxmltools/sklearn-onnx/_apis/build/status/sklearn-onnx-linux-conda-ci?branchName=master)](https://dev.azure.com/onnxmltools/sklearn-onnx/_build/latest?definitionId=5?branchName=master) | [![Build Status](https://dev.azure.com/onnxmltools/sklearn-onnx/_apis/build/status/sklearn-onnx-win32-conda-ci?branchName=master)](https://dev.azure.com/onnxmltools/sklearn-onnx/_build/latest?definitionId=5?branchName=master)|

## Introduction
*sklearn-onnx* converts [scikit-learn](https://scikit-learn.org/stable/) models to [ONNX](https://github.com/onnx/onnx).
Once in the ONNX format, you can use tools like [ONNX Runtime](https://github.com/Microsoft/onnxruntime) for high performance scoring.
All converters are tested with [onnxruntime](https://onnxruntime.ai/).

## Documentation
Full documentation including tutorials is available at [https://onnx.ai/sklearn-onnx/](https://onnx.ai/sklearn-onnx/).
[Supported scikit-learn Models](https://onnx.ai/sklearn-onnx/supported.html)
Last supported opset is 15.

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

## Contribute
We welcome contributions in the form of feedback, ideas, or code.

## License
[Apache License v2.0](LICENSE)
