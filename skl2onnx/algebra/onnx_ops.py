"""
Place holder for all ONNX operators.
"""
import sys
import onnx
from .automation import get_rst_doc


def ClassFactory(class_name, op_name, inputs, outputs,
                 input_range, output_range,
                 domain, attr_names, doc,
                 deprecated, since_version,
                 past_version):
    from .onnx_operator import OnnxOperator

    def __init__(self, *args, **kwargs):

        op_version = kwargs.pop('op_version', None)

        if op_version is None:
            if len(args) == 0 and input_range[0] == input_range[1]:
                args = [_[0] for _ in self.__class__.expected_inputs]
            if not (input_range[0] <= len(args) <= input_range[1]):
                raise RuntimeError("Unexpected number of inputs, "
                                   "got {}, expecting {} for operator "
                                   "'{}'.".format(
                                       len(args), len(inputs), op_name))

        attr_names = self.attr_names
        if '_' in self.__class__.__name__:
            op_version_class = int(self.__class__.__name__.split('_')[-1])
            op_version = min(op_version, op_version_class)
        else:
            op_version_class = None

        found = None
        if op_version is not None:
            # attr_names refers to the most recent version of
            # this operator. We may need an older one.
            for op in range(op_version, 0, -1):
                name = '{}_{}'.format(self.__class__.__name__, op)
                if name in self.past_version:
                    found = (name, op)
                    attr_names = self.past_version[name].attr_names
                    break
        if (op_version_class is not None and found is not None and
                found[-1] != op_version_class):
            raise RuntimeError(
                "op_version={} does not refer to the same opset as the class "
                "name ('{}').".format(op_version, self.__class__.__name__))
        for key in kwargs:
            if key in {'output_names', 'op_version', 'domain', 'ir_version'}:
                continue
            if key not in attr_names:
                raise TypeError("Argument '%s' not valid for '%s'"
                                % (key, op_name))

        if op_version is not None:
            kwargs['op_version'] = op_version
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
                     'since_version': since_version,
                     'past_version': past_version,
                     'attr_names': attr_names})
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
        if schema.name in res:
            if schema.since_version > res[schema.name].since_version:
                # We keep the most recent one.
                res[schema.name] = schema
        else:
            res[schema.name] = schema
        res[schema.name + '_' + str(schema.since_version)] = schema
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

        if '_' in name:
            class_name = "Onnx" + name
        else:
            class_name = "Onnx" + schema.name

        cl = ClassFactory(class_name, schema.name, inputs, outputs,
                          [schema.min_input, schema.max_input],
                          [schema.min_output, schema.max_output],
                          schema.domain, args,
                          "**Version**" + doc.split('**Version**')[-1],
                          getattr(schema, 'deprecated', False),
                          schema.since_version, {})
        cls[class_name] = cl

    # Retrieves past classes.
    for name in cls:
        if '_' not in name:
            continue
        main, version = name.split('_')
        last = cls[main]
        last.past_version[name] = cls[name]

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
