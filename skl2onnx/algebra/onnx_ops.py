"""
Place holder for all ONNX operators.
"""
import sys
import onnx
from .automation import get_rst_doc


def ClassFactory(class_name, op_name, inputs, outputs,
                 input_range, output_range,
                 domain, attr_names, doc,
                 deprecated, since_version):
    from .onnx_operator import OnnxOperator

    def __init__(self, *args, **kwargs):

        if len(args) == 0 and input_range[0] == input_range[1]:
            args = [_[0] for _ in self.__class__.expected_inputs]
        if not (input_range[0] <= len(args) <= input_range[1]):
            raise RuntimeError("Unexpected number of inputs, "
                               "got {}, expecting {} for operator "
                               "'{}'.".format(
                                   len(args), len(inputs), op_name))

        for key in kwargs:
            if key in {'output_names', 'op_version', 'domain'}:
                continue
            if key not in attr_names:
                raise TypeError("Argument '%s' not valid for '%s'"
                                % (key, op_name))

        OnnxOperator.__init__(self, *args, **kwargs)

    newclass = type(class_name, (OnnxOperator,),
                    {"__init__": __init__, '__doc__': doc,
                     'expected_inputs': inputs,
                     'expected_outputs': outputs,
                     'operator_name': op_name,
                     'input_range': input_range,
                     'output_range': output_range,
                     'domain': domain,
                     'is_deprecated': deprecated,
                     'since_version': since_version})
    return newclass


def dynamic_class_creation():
    """
    Automatically generates classes for each of the operators
    module *onnx* defines and described at
    `Operators
    <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`_
    and `Operators
    <https://github.com/onnx/onnx/blob/master/docs/
    Operators-ml.md>`_.
    """
    res = {}
    for schema in onnx.defs.get_all_schemas_with_history():
        if schema.support_level == schema.SupportType.EXPERIMENTAL:
            # Skips experimental operators.
            continue
        # Multiple version can coexist. The last one is kept.
        res[schema.name] = schema
    cls = {}

    def _c(obj, label, i):
        name = '%s%d' % (obj.name or label, i)
        tys = obj.typeStr or ''
        return (name, tys)

    for name in sorted(res):
        schema = res[name]
        doc = get_rst_doc(schema)
        inputs = [_c(o, 'I', i) for i, o in enumerate(schema.inputs)]
        outputs = [_c(o, 'O', i) for i, o in enumerate(schema.outputs)]
        args = [p for p in schema.attributes]
        class_name = "Onnx" + schema.name
        cl = ClassFactory(class_name, schema.name, inputs, outputs,
                          [schema.min_input, schema.max_input],
                          [schema.min_output, schema.max_output],
                          schema.domain, args,
                          "**Version**" + doc.split('**Version**')[-1],
                          getattr(schema, 'deprecated', False),
                          schema.since_version)
        cls[class_name] = cl
    return cls


def _update_module():
    """
    Dynamically updates the module with operators defined
    by *ONNX*.
    """
    res = dynamic_class_creation()
    this = sys.modules[__name__]
    for k, v in res.items():
        setattr(this, k, v)


_update_module()
