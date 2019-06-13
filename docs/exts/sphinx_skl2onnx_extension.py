# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Extension for sphinx.
"""
from importlib import import_module
import sphinx
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from sphinx.util.nodes import nested_parse_with_titles
from tabulate import tabulate
import skl2onnx
from skl2onnx.algebra.onnx_ops import dynamic_class_creation
from skl2onnx.algebra.sklearn_ops import dynamic_class_creation_sklearn
from skl2onnx.validate import sklearn_operators
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

        def make_ref(name):
            cl = cls[name]
            return ":ref:`l-onnx-{}`".format(cl.__name__)

        table = []
        cut = len(sorted_keys) // 3 + (1 if len(sorted_keys) % 3 else 0)
        for i in range(cut):
            row = []
            row.append(make_ref(sorted_keys[i]))
            if i + cut < len(sorted_keys):
                row.append(make_ref(sorted_keys[i + cut]))
                if i + cut * 2 < len(sorted_keys):
                    row.append(make_ref(sorted_keys[i + cut * 2]))
                else:
                    row.append('')
            else:
                row.append('')
                row.append('')
            table.append(row)

        rst = tabulate(table, tablefmt="rst")
        rows = rst.split("\n")

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


class SupportedSklearnOpsDirective(Directive):
    """
    Automatically displays the list of available converters.
    """
    required_arguments = False
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {}
    has_content = False

    def run(self):
        cls = dynamic_class_creation_sklearn()
        rows = []
        sorted_keys = list(sorted(cls))
        main = nodes.container()

        def make_ref(name):
            cl = cls[name]
            return ":ref:`l-sklops-{}`".format(cl.__name__)

        table = []
        cut = len(sorted_keys) // 3 + (1 if len(sorted_keys) % 3 else 0)
        for i in range(cut):
            row = []
            row.append(make_ref(sorted_keys[i]))
            if i + cut < len(sorted_keys):
                row.append(make_ref(sorted_keys[i + cut]))
                if i + cut * 2 < len(sorted_keys):
                    row.append(make_ref(sorted_keys[i + cut * 2]))
                else:
                    row.append('')
            else:
                row.append('')
                row.append('')
            table.append(row)

        rst = tabulate(table, tablefmt="rst")
        rows = rst.split("\n")

        node = nodes.container()
        st = StringList(rows)
        nested_parse_with_titles(self.state, st, node)
        main += node

        rows.append('')
        for name in sorted_keys:
            rows = []
            cl = cls[name]
            rows.append('.. _l-sklops-{}:'.format(cl.__name__))
            rows.append('')
            rows.append(cl.__name__)
            rows.append('=' * len(cl.__name__))
            rows.append('')
            rows.append(".. autoclass:: skl2onnx.algebra.sklearn_ops.{}".format(name))
            st = StringList(rows)
            node = nodes.container()
            nested_parse_with_titles(self.state, st, node)
            main += node

        return [main]
    

class AllSklearnOpsDirective(Directive):
    """
    Displays the list of models implemented in scikit-learn
    and whether or not there is an associated converter.
    """
    required_arguments = False
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {}
    has_content = False

    def run(self):
        from sklearn import __version__ as skver
        found = [(d['name'], d['subfolder'], d['cl'], d['supported'])
                 for d in sklearn_operators()]
        nbconverters = 0
        rows = [".. list-table::", "    :header-rows: 1", "    :widths: 10 7 4",
                "", "    * - Name", "      - Package", "      - Supported"]
        for name, sub, cl, supported in found:
            rows.append("    * - " + name)
            rows.append("      - " + sub)
            if supported:
                rows.append("      - Yes")
                nbconverters += 1
            else:
                rows.append("      -")
            
        rows.append("")
        rows.append("scikit-learn's version is **{0}**.".format(skver))
        rows.append("{0}/{1} models are covered.".format(nbconverters, len(found)))

        node = nodes.container()
        st = StringList(rows)
        nested_parse_with_titles(self.state, st, node)
        main = nodes.container()
        main += node
        return [main]


def df2rst(df, add_line=True, align="l", column_size=None, index=False,
           list_table=False, title=None, header=True, sep=',',
           number_format=None):
    if isinstance(df, str):
        import pandas
        df = pandas.read_csv(df, encoding="utf-8", sep=sep)

    if number_format is not None:
        if isinstance(number_format, int):
            number_format = "{:.%dg}" % number_format
            import numpy
            import pandas
            typ1 = numpy.float64
            _df = pandas.DataFrame({'f': [0.12]})
            typ2 = list(_df.dtypes)[0]
            number_format = {typ1: number_format, typ2: number_format}
        df = df.copy()
        for name, typ in zip(df.columns, df.dtypes):
            if name in number_format:
                pattern = number_format[name]
                df[name] = df[name].apply(lambda x: pattern.format(x))
            elif typ in number_format:
                pattern = number_format[typ]
                df[name] = df[name].apply(lambda x: pattern.format(x))

    if index:
        df = df.reset_index(drop=False).copy()
        ind = df.columns[0]

        def boldify(x):
            try:
                return "**{0}**".format(x)
            except Exception as e:
                raise Exception(
                    "Unable to boldify type {0}".format(type(x))) from e

        try:
            values = df[ind].apply(boldify)
        except Exception:
            warnings.warn("Unable to boldify the index (1).", SyntaxWarning)

        try:
            df[ind] = values
        except Exception:
            warnings.warn("Unable to boldify the index (2).", SyntaxWarning)

    import numpy
    typstr = str

    def align_string(s, align, length):
        if len(s) < length:
            if align == "l":
                return s + " " * (length - len(s))
            elif align == "r":
                return " " * (length - len(s)) + s
            elif align == "c":
                m = (length - len(s)) // 2
                return " " * m + s + " " * (length - m - len(s))
            else:
                raise ValueError(
                    "align should be 'l', 'r', 'c' not '{0}'".format(align))
        else:
            return s

    def complete(cool):
        if list_table:
            i, s = cool
            if s is None:
                s = ""
            if isinstance(s, float) and numpy.isnan(s):
                s = ""
            else:
                s = typstr(s).replace("\n", " ")
            return (" " + s) if s else s
        else:
            i, s = cool
            if s is None:
                s = " " * 4
            if isinstance(s, float) and numpy.isnan(s):
                s = ""
            else:
                s = typstr(s).replace("\n", " ")
            i -= 2
            s = align_string(s.strip(), align, i)
            return s

    if list_table:

        def format_on_row(row):
            one = "\n      -".join(map(complete, enumerate(row)))
            res = "    * -" + one
            return res

        rows = [".. list-table:: {0}".format(title if title else "").strip()]
        if column_size is None:
            rows.append("    :widths: auto")
        else:
            rows.append("    :widths: " + " ".join(map(str, column_size)))
        if header:
            rows.append("    :header-rows: 1")
        rows.append("")
        if header:
            rows.append(format_on_row(df.columns))
        rows.extend(map(format_on_row, df.values))
        rows.append("")
        table = "\n".join(rows)
        return table
    else:
        length = [(len(_) if isinstance(_, typstr) else 5) for _ in df.columns]
        for row in df.values:
            for i, v in enumerate(row):
                length[i] = max(length[i], len(typstr(v).strip()))
        if column_size is not None:
            if len(length) != len(column_size):
                raise ValueError("length and column_size should have the same size {0} != {1}".format(
                    len(length), len(column_size)))
            for i in range(len(length)):
                if not isinstance(column_size[i], int):
                    raise TypeError(
                        "column_size[{0}] is not an integer".format(i))
                length[i] *= column_size[i]

        ic = 2
        length = [_ + ic for _ in length]
        line = ["-" * l for l in length]
        lineb = ["=" * l for l in length]
        sline = "+%s+" % ("+".join(line))
        slineb = "+%s+" % ("+".join(lineb))
        res = [sline]

        res.append("| %s |" % " | ".join(
            map(complete, zip(length, df.columns))))
        res.append(slineb)
        res.extend(["| %s |" % " | ".join(map(complete, zip(length, row)))
                    for row in df.values])
        if add_line:
            t = len(res)
            for i in range(t - 1, 3, -1):
                res.insert(i, sline)
        res.append(sline)
        table = "\n".join(res) + "\n"
        return table


class AllSklearnOpsOpsetDirective(Directive):
    """
    Displays the list of models implemented in scikit-learn
    and whether or not there is an associated converter.
    """
    required_arguments = False
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {}
    has_content = False

    def run(self):
        from skl2onnx.validate import validate_operator_opsets
        import pandas
        import numpy
        
        obs = validate_operator_opsets()
                
        def aggfunc(values):
            if len(values) != 1:
                raise ValueError(values)
            val = values.iloc[0]
            if isinstance(val, float) and numpy.isnan(val):
                return ""
            else:
                return val

        df = pandas.DataFrame(obs)
        piv = pandas.pivot_table(df, values="available", 
                                 index=['name', 'problem', 'scenario'], 
                                 columns='opset', 
                                 aggfunc=aggfunc).reset_index(drop=False)
        cols = piv.columns
        versions = ["opset%d" % t for t in range(1, piv.shape[1] - 2)]
        indices = ["name","problem","scenario"]
        piv.columns = indices + versions
        piv = piv[indices + list(reversed(versions))]

        rest = df2rst(piv)
        rows = rest.split('\n')

        node = nodes.container()
        st = StringList(rows)
        nested_parse_with_titles(self.state, st, node)
        main = nodes.container()
        main += node
        return [main]
    

def setup(app):
    # Placeholder to initialize the folder before
    # generating the documentation.
    app.add_role('skl2onnxversion', skl2onnx_version_role)
    app.add_directive('supported-skl2onnx', SupportedSkl2OnnxDirective)
    app.add_directive('supported-onnx-ops', SupportedOnnxOpsDirective)
    app.add_directive('supported-sklearn-ops', SupportedSklearnOpsDirective)
    app.add_directive('covered-sklearn-ops', AllSklearnOpsDirective)
    app.add_directive('supported-onnx-ops-opset', AllSklearnOpsOpsetDirective)
    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}
