# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import textwrap
import onnx
import onnx.defs
from onnx.defs import OpSchema


def _get_doc_template():
    try:
        from jinja2 import Template
    except ImportError:
        class Template:
            def __init__(self, *args):
                pass

            def render(self, **context):
                schemas = context['schemas']
                rows = []
                for sch in schemas:
                    doc = sch.doc or ''
                    name = sch.name
                    if name is None:
                        raise RuntimeError("An operator must have a name.")
                    rows.extend([name, "=" * len(name),
                                 "", doc, ""])
                return "\n".join(rows)

    return Template(textwrap.dedent("""
        {% for sch in schemas %}

        {{format_name_with_domain(sch)}}
        {{'=' * len(format_name_with_domain(sch))}}
        
        **Version**

        *Onnx name: {{sch.name}}*

        {% if sch.support_level == OpSchema.SupportType.EXPERIMENTAL %}
        No versioning maintained for experimental ops.
        {% else %}
        This version of the operator has been {% if
        sch.deprecated %}deprecated{% else %}available{% endif %} since
        version {{sch.since_version}}{% if
        sch.domain %} of domain {{sch.domain}}{% endif %}.
        {% if len(sch.versions) > 1 %}
        Other versions of this operator:
        {% for v in sch.version[:-1] %} {{v}} {% endfor %}
        {% endif %}
        {% endif %}

        {% if sch.attributes %}
        **Attributes**

        {% for _, attr in sorted(sch.attributes.items()) %}* *{{attr.name}}*{%
          if attr.required %} (required){% endif %}: {{attr.description}} {%
          if attr.default_value %}Default value is ``{{attr.default_value
          }}``{% endif %}
        {% endfor %}
        {% endif %}

        {% if sch.inputs %}
        **Inputs**

        {% if sch.min_input != sch.max_input %}Between {{sch.min_input
        }} and {{sch.max_input}} inputs.
        {% endif %}
        {% for ii, inp in enumerate(sch.inputs) %}
        * *{{getname(inp, ii)}}*{{format_option(inp)}}{{inp.typeStr}}: {{
        inp.description}}{% endfor %}
        {% endif %}

        {% if sch.outputs %}
        **Outputs**

        {% if sch.min_output != sch.max_output %}Between {{sch.min_output
        }} and {{sch.max_output}} outputs.
        {% endif %}
        {% for ii, out in enumerate(sch.outputs) %}
        * *{{getname(out, ii)}}*{{format_option(out)}}{{out.typeStr}}: {{
        out.description}}{% endfor %}
        {% endif %}

        {% if sch.type_constraints %}
        **Type Constraints**

        {% for ii, type_constraint in enumerate(sch.type_constraints)
        %}* {{getconstraint(type_constraint, ii)}}: {{
        type_constraint.description}}
        {% endfor %}
        {% endif %}

        **Summary**

        {{sch.doc}}

        {% endfor %}
    """))


_template_operator = _get_doc_template()


def get_domain_list():
    """
    Returns the list of available domains.
    """
    return list(sorted(set(map(lambda s: s.domain,
                               onnx.defs.get_all_schemas_with_history()))))


def get_rst_doc(op_name=None):
    """
    Returns a documentation in RST format

    :param op_name: operator name of None for all
    :return: string

    The function relies on module *jinja2* or replaces it
    with a simple rendering if not present.
    """
    if op_name is None:
        schemas = onnx.defs.get_all_schemas_with_history()
    elif isinstance(op_name, str):
        schemas = [schema for schema in onnx.defs.get_all_schemas_with_history(
        ) if schema.name == op_name]
        if len(schemas) > 1:
            raise RuntimeError(
                "Multiple operators have the same name '{}'.".format(op_name))
    elif not isinstance(op_name, list):
        schemas = [op_name]
    if len(schemas) == 0:
        raise ValueError(
            "Unable to find any operator with name '{}'.".format(op_name))

    # from onnx.backend.sample.ops import collect_sample_implementations
    # from onnx.backend.test.case import collect_snippets
    # SNIPPETS = collect_snippets()
    # SAMPLE_IMPLEMENTATIONS = collect_sample_implementations()
    def format_name_with_domain(sch):
        if sch.domain:
            return '{} ({})'.format(sch.name, sch.domain)
        else:
            return sch.name

    def format_option(obj):
        opts = []
        if OpSchema.FormalParameterOption.Optional == obj.option:
            opts.append('optional')
        elif OpSchema.FormalParameterOption.Variadic == obj.option:
            opts.append('variadic')
        if obj.isHomogeneous:
            opts.append('heterogeneous')
        if opts:
            return " (%s)" % ", ".join(opts)
        else:
            return ""

    def getconstraint(const, ii):
        if const.type_param_str:
            name = const.type_param_str
        else:
            name = str(ii)
        if const.allowed_type_strs:
            name += " " + ", ".join(const.allowed_type_strs)
        return name

    def getname(obj, i):
        name = obj.name
        if len(name) == 0:
            return str(i)
        else:
            return name

    fnwd = format_name_with_domain
    return _template_operator.render(schemas=schemas, OpSchema=OpSchema,
                                     len=len,
                                     getattr=getattr, sorted=sorted,
                                     format_option=format_option,
                                     getconstraint=getconstraint,
                                     getname=getname, enumerate=enumerate,
                                     format_name_with_domain=fnwd)


def ClassFactory(class_name, op_name, inputs, outputs,
                 input_range, output_range,
                 domain, attr_names, doc):
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
                     'domain': domain})
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
        name = obj.name or '%s%d' % (label, i)
        tys = obj.typeStr or ''
        return (name, tys)

    for name in sorted(res):
        schema = res[name]
        doc = get_rst_doc(schema)
        if name == "Abs":
            inputs = [('X', 'FloatTensorType')]
            outputs = [('Y', 'FloatTensorType')]
        else:
            inputs = [_c(o, 'I', i) for i, o in enumerate(schema.inputs)]
            outputs = [_c(o, 'O', i) for i, o in enumerate(schema.outputs)]
        args = [p for p in schema.attributes]
        class_name = "Onnx" + schema.name
        cl = ClassFactory(class_name, schema.name, inputs, outputs,
                          [schema.min_input, schema.max_input],
                          [schema.min_output, schema.max_output],
                          schema.domain, args, doc.split('**Summary**')[-1])
        cls[class_name] = cl
    return cls
