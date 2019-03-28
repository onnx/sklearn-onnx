# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import textwrap
import onnx
import onnx.defs
from onnx.defs import OpSchema
from onnx.backend.test.case import collect_snippets
from onnx.backend.sample.ops import collect_sample_implementations


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
                    rows.extend([sch.name, "=" * len(sch.name), "", sch.description, ""])
                return "\n".join(rows)
            
    return Template(textwrap.dedent("""
    {% for sch in schemas %}

    {{format_name_with_domain(sch)}}
    {{'=' * len(format_name_with_domain(sch))}}

    **Summary**

    {{sch.doc}}

    **Version**    

    {% if sch.support_level == OpSchema.SupportType.EXPERIMENTAL %}    
    No versioning maintained for experimental ops.    
    {% else %}    
    This version of the operator has been {% if 
    sch.deprecated %}deprecated{% else %}available{% endif %} since
    version {{sch.since_version}}{% if 
    sch.domaine %} of domain {{sch.domain}}{% endif %}.
    {% if len(sch.versions) > 1 %}    
    Other versions of this operator:
    {% for v in sch.version[:-1] %} {{v}} {% endfor %}    
    {% endif %}    
    {% endif %}

    {% if sch.attributes %}
    **Attributes**

    {% for _, attr in sorted(sch.attributes.items()) %}* *{{attr.name}}*{% 
      if attr.required %}(required){% endif %}: {{attr.description}} {% 
      if attr.default_value %}Default value is ``{{attr.default_value}}``{% endif %}
    {% endfor %}
    {% endif %}

    {% if sch.inputs %}    
    **Inputs**

    {% if sch.min_input != sch.max_input %}Between {{sch.min_input}} and {{sch.max_input}} inputs.
    {% endif %}
    {% for ii, inp in enumerate(sch.inputs) %}
    * *{{getname(inp, ii)}}*{{format_option(inp)}}{{inp.typeStr}}: {{inp.description}}{% endfor %}
    {% endif %}

    {% if sch.outputs %}    
    **Outputs**

    {% if sch.min_output != sch.max_output %}Between {{sch.min_output}} and {{sch.max_output}} outputs.
    {% endif %}
    {% for ii, out in enumerate(sch.outputs) %}
    * *{{getname(out, ii)}}*{{format_option(out)}}{{out.typeStr}}: {{out.description}}{% endfor %}
    {% endif %}

    {% if sch.type_constraints %}    
    **Type Constraints**

    {% for type_constraint in sch.type_constraints %}* {{type_constraint.type_param_str}}:{% 
        for allowedType in type_constraint.allowed_type_strs %}{{allowedType}}, {% 
        endfor %}{{type_constraint.description}}
    {% endfor %}
    {% endif %}

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
    
    The function relies on module *jinja2*.
    """    
    if op_name is None:
        schemas = onnx.defs.get_all_schemas_with_history()
    elif isinstance(op_name, str):
        schemas = [schema for schema in onnx.defs.get_all_schemas_with_history() if schema.name == op_name]
        if len(schemas) > 1:
            raise RuntimeError("Multiple operators have the same name '{}'.".format(op_name))
    elif not isinstance(op_name, list):
        schemas = [op_name]
    if len(schemas) == 0:
        raise ValueError("Unable to find any operator with name '{}'.".format(op_name))
        
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
            
    def getname(obj, i):
        name = obj.name
        if len(name) == 0:
            return str(i)
        else:
            return name
    
    return _template_operator.render(schemas=schemas, OpSchema=OpSchema, len=len,
                                     getattr=getattr, sorted=sorted,
                                     format_option=format_option,
                                     getname=getname, enumerate=enumerate,
                                     format_name_with_domain=format_name_with_domain)


def ClassFactory(name, input_names, attr_names, doc):
    from .onnx_operator import OnnxOperator
    
    def __init__(self, *args, **kwargs):
        
        if len(args) != len(input_names):
            raise RuntimeError("Unexpected number of inputs, "
                               "got {}, expecting {}.".format(
                len(args), len(input_names)))
            
        for key in kwargs:
            if key in {'outputs'}:
                continue
            if key not in attr_names:
                raise TypeError("Argument '%s' not valid for '%s'" 
                    % (key, self.__class__.__name__))
                    
        OnnxOperator.__init__(self, *args, **kwargs)
        
    newclass = type(name, (OnnxOperator,),
                    {"__init__": __init__, '__doc__': doc})    
    return newclass


def dynamic_class_creation():
    """
    Automatically generates classes for each of the operators
    module *onnx* defines and described at
    `Operators <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`_ and
    `Operators <https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md>`_.
    """
    res = {}
    for schema in onnx.defs.get_all_schemas_with_history():
        # Multiple version can coexist. The last one is kept.
        res[schema.name] = schema
    cls = {}
    for name in sorted(res):
        schema = res[name]
        doc = get_rst_doc(schema)
        inputs = [i.name for i in schema.inputs]
        args = [p for p in schema.attributes]
        cl = ClassFactory(schema.name, inputs, args, doc.split('**Summary**')[-1])
        cls[schema.name] = cl
    return cls    
