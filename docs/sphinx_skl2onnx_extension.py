# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Extension for sphinx.
"""
import sphinx
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from sphinx.util.nodes import nested_parse_with_titles
import skl2onnx
from skl2onnx.algebra.automation import dynamic_class_creation
import onnxruntime


def skl2onnx_version_role(role, rawtext, text, lineno, inliner, options=None, content=None):
    """
    Defines custom role *skl2onnx-version* which returns
    *skl2onnx* version.
    """
    if options is None:
        options = {}
    if content is None:
        content = []
    if text == 'v':
        version = 'v' + skl2onnx.__version__
    elif text == 'rt':
        version = 'v' + onnxruntime.__version__
    else:
        raise RuntimeError("skl2onnx_version_role cannot interpret content '{0}'.".format(text))
    node = nodes.Text(version)
    return [node], []


class SupportedSkl2OnnxDirective(Directive):
    """
    Automatically displays the list of models
    *skl2onnx* can currently convert.
    """
    required_arguments = False
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {}
    has_content = False

    def run(self):
        models = skl2onnx.supported_converters(True)
        bullets = nodes.bullet_list()
        ns = [bullets]
        for mod in models:
            par = nodes.paragraph()
            par += nodes.Text(mod)
            bullets += nodes.list_item('', par)
        return ns


class SupportedOnnxOpsDirective(Directive):
    """
    Automatically displays the list of supported ONNX models
    *skl2onnx* can use to build converters.
    """
    required_arguments = False
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {}
    has_content = False

    def run(self):
        cls = dynamic_class_creation()
        rows = []
        sorted_keys = list(sorted(cls))
        main = nodes.container()
        for name in sorted_keys:
            cl = cls[name]
            rows.append('* :ref:`l-onnx-{}`'.format(cl.__name__))
            node = nodes.container()
        st = StringList(rows)
        nested_parse_with_titles(self.state, st, node)
        main += node
            
        rows.append('')
        for name in sorted_keys:            
            rows = []
            cl = cls[name]
            rows.append('.. _l-onnx-{}:'.format(cl.__name__))
            rows.append('')
            rows.append(cl.__name__)
            rows.append('=' * len(cl.__name__))
            rows.append('')
            rows.append(".. autoclass:: skl2onnx.algebra.onnx_ops.{}".format(name))
            st = StringList(rows)
            node = nodes.container()
            nested_parse_with_titles(self.state, st, node)
            main += node

        return [main]


def setup(app):
    # Placeholder to initialize the folder before
    # generating the documentation.
    app.add_role('skl2onnxversion', skl2onnx_version_role)
    app.add_directive('supported-skl2onnx', SupportedSkl2OnnxDirective)
    app.add_directive('supported-onnx-ops', SupportedOnnxOpsDirective)
    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}

