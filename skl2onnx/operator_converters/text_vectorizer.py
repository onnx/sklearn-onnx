# Copyright (c) ...
# Licensed under the MIT License.

from skl2onnx.common._apply_operation import apply_tokenizer
from skl2onnx.common.data_types import StringTensorType
from skl2onnx.common._registration import register_converter


def convert_sklearn_count_vectorizer(scope, operator, container):
    op = operator.raw_operator
    input_var = operator.inputs[0]
    output_var = operator.outputs[0]

    if not isinstance(input_var.type, StringTensorType):
        raise RuntimeError("CountVectorizer input must be a string tensor")

    analyzer = getattr(op, "analyzer", "word")

    if analyzer == "word":
        # existing word-level tokenizer
        apply_tokenizer(
            scope,
            input_var.full_name,
            output_var.full_name,
            container,
            op.vocabulary_,
            mark=operator.full_name,
        )

    elif analyzer in ("char", "char_wb"):
        ngram_range = getattr(op, "ngram_range", (1, 1))

        # ONNX Tokenizer regex for single characters
        pattern = "."
        if analyzer == "char_wb":
            # Approximate: capture chars incl. word boundaries
            # Note: real sklearn pads words with spaces, this regex
            # is an approximation (still useful for deployment).
            pattern = r"\b.|.\b|."

        apply_tokenizer(
            scope,
            input_var.full_name,
            output_var.full_name,
            container,
            op.vocabulary_,
            mark=operator.full_name,
            pattern=pattern,
            ngram_range=ngram_range,
        )

    else:
        raise NotImplementedError(
            f"Analyzer={analyzer!r} not yet supported in skl2onnx."
        )


register_converter(
    "SklearnCountVectorizer",
    convert_sklearn_count_vectorizer,
)

