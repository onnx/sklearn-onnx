# SPDX-License-Identifier: Apache-2.0


from skl2onnx.proto import onnx_proto
from skl2onnx.common import utils as convert_utils


def set_model_domain(model, domain):
    """
    Sets the domain on the ONNX model.

    :param model: instance of an ONNX model
    :param domain: string containing the domain name of the model

    Example:

    ::
        from test_utils import set_model_domain
        onnx_model = load_model("SqueezeNet.onnx")
        set_model_domain(onnx_model, "com.acme")
    """
    if model is None or not isinstance(model, onnx_proto.ModelProto):
        raise ValueError("Parameter model is not an onnx model.")
    if not convert_utils.is_string_type(domain):
        raise ValueError("Parameter domain must be a string type.")
    model.domain = domain


def set_model_version(model, version):
    """
    Sets the version of the ONNX model.

    :param model: instance of an ONNX model
    :param version: integer containing the version of the model

    Example:

    ::
        from test_utils import set_model_version
        onnx_model = load_model("SqueezeNet.onnx")
        set_model_version(onnx_model, 1)
    """
    if model is None or not isinstance(model, onnx_proto.ModelProto):
        raise ValueError("Parameter model is not an onnx model.")
    if not convert_utils.is_numeric_type(version):
        raise ValueError("Parameter version must be a numeric type.")
    model.model_version = version


def set_model_doc_string(model, doc, override=False):
    """
    Sets the doc string of the ONNX model.

    :param model: instance of an ONNX model
    :param doc: string containing the doc string that describes the model.
    :param override: bool if true will always override the doc
        string with the new value

    Example:

    ::
        from test_utils import set_model_doc_string
        onnx_model = load_model("SqueezeNet.onnx")
        set_model_doc_string(onnx_model, "Sample doc string")
    """
    if model is None or not isinstance(model, onnx_proto.ModelProto):
        raise ValueError("Parameter model is not an onnx model.")
    if not convert_utils.is_string_type(doc):
        raise ValueError("Parameter doc must be a string type.")
    if model.doc_string and not doc and override is False:
        raise ValueError(
            "Failed to overwrite the doc string with a blank string,"
            " set override to True if intentional."
        )
    model.doc_string = doc
