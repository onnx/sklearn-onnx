# SPDX-License-Identifier: Apache-2.0

import textwrap
import onnx
import onnx.defs


def _get_doc_template():
    try:
        from jinja2 import Template
    except ImportError:

        class Template:
            def __init__(self, *args):
                pass

            def render(self, **context):
                schemas = context["schemas"]
                rows = []
                for sch in schemas:
                    doc = sch.doc or ""
                    name = sch.name
                    if name is None:
                        raise RuntimeError("An operator must have a name.")
                    rows.extend([name, "=" * len(name), "", doc, ""])
                return "\n".join(rows)

    return Template(
        textwrap.dedent(
            """
        {% for sch in schemas %}

        {{format_name_with_domain(sch)}}
        {{'=' * len(format_name_with_domain(sch))}}

        **Version**

        *Onnx name:* `{{sch.name}} <{{build_doc_url(sch)}}{{sch.name}}>`_

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

        **Summary**

        {{process_documentation(sch.doc)}}

        {% if sch.attributes %}
        **Attributes**

        {% for _, attr in sorted(sch.attributes.items()) %}* *{{attr.name}}*{%
          if attr.required %} (required){% endif %}: {{attr.description}} {%
          if attr.default_value %}Default value is
          ``{{str(attr.default_value).replace('\\n', ' ').strip()}}``{%
          endif %}
        {% endfor %}
        {% endif %}

        {% if sch.inputs %}
        **Inputs**

        {% if sch.min_input != sch.max_input %}Between {{sch.min_input
        }} and {{sch.max_input}} inputs.
        {% endif %}
        {% for ii, inp in enumerate(sch.inputs) %}
        * *{{getname(inp, ii)}}*{{format_option(inp)}}{{get_type_str(inp)}}: {{
        inp.description}}{% endfor %}
        {% endif %}

        {% if sch.outputs %}
        **Outputs**

        {% if sch.min_output != sch.max_output %}Between {{sch.min_output
        }} and {{sch.max_output}} outputs.
        {% endif %}
        {% for ii, out in enumerate(sch.outputs) %}
        * *{{getname(out, ii)}}*{{format_option(out)}}{{get_type_str(out)}}: {{
        out.description}}{% endfor %}
        {% endif %}

        {% if sch.type_constraints %}
        **Type Constraints**

        {% for ii, type_constraint in enumerate(sch.type_constraints)
        %}* {{getconstraint(type_constraint, ii)}}: {{
        type_constraint.description}}
        {% endfor %}
        {% endif %}

        {% endfor %}
    """
        )
    )


_template_operator = _get_doc_template()


def get_domain_list():
    """
    Returns the list of available domains.
    """
    return list(
        sorted(set(map(lambda s: s.domain, onnx.defs.get_all_schemas_with_history())))
    )


def _get_doc_template_sklearn():
    try:
        from jinja2 import Template
    except ImportError:

        class Template:
            def __init__(self, *args):
                pass

            def render(self, **context):
                schemas = context["schemas"]
                rows = []
                for sch in schemas:
                    doc = sch.doc or ""
                    name = sch.name
                    if name is None:
                        raise RuntimeError("An operator must have a name.")
                    rows.extend([name, "=" * len(name), "", doc, ""])
                return "\n".join(rows)

    return Template(
        textwrap.dedent(
            """
        {% for cl in classes %}

        .. _l-sklops-{{cl.__name__}}:

        {{cl.__name__}}
        {{'=' * len(cl.__name__)}}

        Corresponding :class:`OnnxSubGraphOperatorMixin
        <skl2onnx.algebra.onnx_subgraph_operator_mixin.
        OnnxSubGraphOperatorMixin>` for model
        **{{cl.operator_name}}**.

        * Shape calculator: *{{cl._fct_shape_calc.__name__}}*
        * Converter: *{{cl._fct_converter.__name__}}*

        {{format_doc(cl)}}

        {% endfor %}
    """
        )
    )


_template_operator_sklearn = _get_doc_template_sklearn()


def get_rst_doc_sklearn():
    """
    Returns a documentation in RST format
    for all :class:`OnnxSubGraphOperatorMixin`.

    :param op_name: operator name of None for all
    :return: string

    The function relies on module *jinja2* or replaces it
    with a simple rendering if not present.
    """

    def format_doc(cl):
        return "\n".join(cl.__doc__.split("\n")[1:])

    from .sklearn_ops import dynamic_class_creation_sklearn

    classes = dynamic_class_creation_sklearn()
    tmpl = _template_operator_sklearn
    values = list(sorted(classes.items()))
    values = [_[1] for _ in values]
    docs = tmpl.render(len=len, classes=values, format_doc=format_doc)
    return docs
