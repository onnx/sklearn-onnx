# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Extension for sphinx.
"""
import os
from textwrap import dedent, indent
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


def missing_ops():
    """
    Builds the list of supported and not supported models.
    """
    from sklearn import __all__
    from sklearn.base import BaseEstimator
    found = []
    for sub in __all__:
        try:
            mod = import_module("{0}.{1}".format("sklearn", sub))
        except ImportError:
            continue
        cls = getattr(mod, "__all__", None)
        if cls is None:
            cls = list(mod.__dict__)
        cls = [mod.__dict__[cl] for cl in cls]
        for cl in cls:
            try:
                issub = issubclass(cl, BaseEstimator)
            except TypeError:
                continue
            if cl.__name__ in {'Pipeline', 'ColumnTransformer',
                               'FeatureUnion', 'BaseEstimator'}:
                continue
            if (sub in {'calibration', 'dummy', 'manifold'} and
                'Calibrated' not in cl.__name__):
                continue
            if issub:
                found.append((cl.__name__, sub, cl))
    found.sort()
    return found
    

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


_meanings = {
    'bin-class': 'binary classification',
    'multi-class': 'multi-class classification',
    'regression': 'regression',
    'multi-reg': 'regression multi-output',
    'num-transform': 'numerical transform',
    'scoring': 'numerical scoring (target is usually needed)',
    'outlier': 'outlier prediction',
    'linearsvc': 'classifier (no *predict_proba*)',
    'cluster': 'clustering (labels)',
    'num+y-trans': 'numerical transform with targets',
    'num-trans-cluster': 'clustering (scores)',
    'clnoproba': 'binary classification (no probabilities)',
}


def covered_opset_converters(app):
    from skl2onnx.validate import (
        summary_report, enumerate_validated_operator_opsets
    )
    import pandas
    import numpy
    
    text = dedent("""
    Availability of Converters for each Opset
    =========================================

    Some ONNX operators converters are using were not all 
    available in older version of ONNX. This version is called
    *opset number*. ONNX 1.4.0 is opset 9, ONNX 1.5.0 is opset 10...
    Next table shows which operator is available in which opset.
    An empty cell means it is not available. Other cells
    contains concatenated flags whose meaning is the following:

    The runtime is sometimes unable to compute the predictions
    for multiple observations at the same time, it needs to be
    called for each observation. This configuration is checked
    for only *onxruntime*. Batch predictions might be working with
    other runtime.

    {0}

    """)
    probs = ["* `{0}`: {1}".format(k, v) for k, v in sorted(_meanings.items())]
    text = text.format("\n".join(probs))
    
    def create_onnx_link(row):
        if row.get('Opset', '') == '':
            return ''
        name = row.get("name", "")
        problem = row.get("problem", "")
        scenario = row.get("scenario", "")
        if "" not in [name, problem, scenario]:
            lab = "l-dot-{name}-{problem}-{scenario}".format(
                name=name, problem=problem, scenario=scenario)
            return ":ref:`ONNX <{}>`".format(lab)
        return ""

    rows = list(enumerate_validated_operator_opsets(verbose=1, debug=False,
                                                    dot_graph=True,
                                                    store_models=True))
    df = pandas.DataFrame(rows)
    piv = summary_report(df)
    piv['ONNX'] = piv.apply(create_onnx_link, axis=1)
    rest = text + df2rst(piv)
    srcdir = app.builder.srcdir
    dest = os.path.join(srcdir, "supported_covered.rst")
    with open(dest, "w", encoding="utf-8") as f:
        f.write(rest)
    
    # second part: visual representation
    opset_max = max(row.get('opset', -1) for row in rows)

    keys = ['MODEL', 'DOT', 'name', 'scenario', 'problem']
    dots = []
    for row in rows:
        if row.get('opset', -1) != opset_max:
            continue
        dot = {k: v for k, v in row.items() if k in keys}
        if len(dot) == len(keys):
            dot["sub"] = dot["MODEL"].__class__.__module__.split('.')[-2]
            dots.append(dot)
    if len(dots) == 0:
        import pprint
        raise RuntimeError("List is empty:\n{}".format(pprint.pformat(rows)))
    
    def clean_dot(dot):
        rep = 'URL="javascript:alert(\'\')", '
        res = dot.replace(rep, "")
        res = res.replace(", ", " ")
        assert "javascript" not in res
        return res
    
    # write
    folder = os.path.join(srcdir, "graphs")
    if not os.path.exists(folder):
        os.makedirs(folder)
    dots = [(dot['sub'], dot['name'], dot['problem'], dot['scenario'],
             dot) for dot in dots]
    dots.sort()
    title = "Visual Representation of Converted Models"
    contents = [title, "=" * len(title), "",
                ".. contents::", "    :local:", ""]
    last_sub = None
    for sub, name, problem, scenario, dot in dots:
        if sub == 'base':
            import pprint
            raise RuntimeError("Issue with module {}\n{}".format(
                dot['MODEL'].__class__.__module__,
                pprint.pformat(dot)))
        filename = "dot_{name}_{problem}_{scenario}".format(
            name=name, problem=problem, scenario=scenario)
        if last_sub is None or sub != last_sub:
            contents.append("")
            contents.append(sub)
            contents.append("+" * len(sub))
            contents.append("")
            contents.append(".. toctree::")            
            contents.append("")
            last_sub = sub
        contents.append("    %s" % filename)
        title = "{0}, {1}, {2}".format(name, problem, scenario)
        
        with open(os.path.join(folder, "%s.rst" % filename), "w") as f:
            f.write(dedent("""
            .. _l-dot-{name}-{problem}-{scenario}:
            
            {title}
            {equal}
            
            The model was trained on a {problem2} problem:
            
            ::
            
            {skl}
            
            And its representation once converted into *ONNX*:
            
            .. graphviz::
                
            {dot}
            
            """).format(title=title, name=name,
                        equal="=" * len(title),
                        problem=problem,
                        scenario=scenario,
                        problem2=_meanings[problem],
                        dot=indent(clean_dot(dot['DOT']), "    "),
                        skl=indent(repr(dot['MODEL']), "    ")))
        
    dest = os.path.join(folder, "index.rst")
    with open(dest, "w") as f:
        f.write("\n".join(contents))
    

def setup(app):
    # Placeholder to initialize the folder before
    # generating the documentation.
    app.add_role('skl2onnxversion', skl2onnx_version_role)
    app.add_directive('supported-skl2onnx', SupportedSkl2OnnxDirective)
    app.add_directive('supported-onnx-ops', SupportedOnnxOpsDirective)
    app.add_directive('supported-sklearn-ops', SupportedSklearnOpsDirective)
    app.add_directive('covered-sklearn-ops', AllSklearnOpsDirective)
    app.connect('builder-inited', covered_opset_converters)
    return {'version': sphinx.__display_version__,
            'parallel_read_safe': False,
            'parallel_write_safe': False}
