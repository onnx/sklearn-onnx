# SPDX-License-Identifier: Apache-2.0

"""
Helpers to test runtimes.
"""
import numpy
from onnx import AttributeProto, numpy_helper  # noqa
from onnx.defs import onnx_opset_version


def _to_str(s):
    if isinstance(s, bytes):
        return s.decode('utf-8')
    return s


def _attribute_value(attr):
    if attr.HasField("f"):
        return attr.f
    if attr.HasField("i"):
        return attr.i
    if attr.HasField("s"):
        return _to_str(attr.s)
    if attr.HasField("t"):
        return numpy_helper.to_array(attr.t)
    if attr.floats:
        return list(attr.floats)
    if attr.ints:
        return list(attr.ints)
    if attr.strings:
        return list(map(_to_str, attr.strings))
    raise NotImplementedError(
        "Unable to return a value for attribute %r." % attr)


class TreeEnsembleAttributes:

    def __init__(self):
        self._names = []

    def add(self, name, value):
        if not name.endswith("_as_tensor"):
            self._names.append(name)
        setattr(self, name, value)

    def __str__(self):
        rows = ["Attributes"]
        for name in self._names:
            if name.endswith("_as_tensor"):
                name = name.replace("_as_tensor", "")
            rows.append(f"  {name}={getattr(self, name)}")
        return "\n".join(rows)


class TreeEnsemble:
    """
    Implementation of a tree.
    """
    def __init__(self, **kwargs):
        self.atts = TreeEnsembleAttributes()

        for name, value in kwargs.items():
            self.atts.add(name, value)

        self.tree_ids = list(sorted(set(self.atts.nodes_treeids)))
        self.root_index = {tid: len(self.atts.nodes_treeids)
                           for tid in self.tree_ids}
        for index, tree_id in enumerate(self.atts.nodes_treeids):
            self.root_index[tree_id] = min(self.root_index[tree_id], index)
        self.node_index = {(tid, nid): i for i, (tid, nid) in enumerate(
            zip(self.atts.nodes_treeids, self.atts.nodes_nodeids))}

    def __str__(self):
        rows = ["TreeEnsemble", f"root_index={self.root_index}",
                str(self.atts)]
        return "\n".join(rows)

    def leaf_index_tree(self, X, tree_id):
        """
        Computes the leaf index for one tree.
        """
        index = self.root_index[tree_id]
        while self.atts.nodes_modes[index] != "LEAF":
            x = X[self.atts.nodes_featureids[index]]
            rule = self.atts.nodes_modes[index]
            th = self.atts.nodes_values[index]
            if rule == "BRANCH_LEQ":
                r = x <= th
            elif rule == "BRANCH_LT":
                r = x < th
            elif rule == "BRANCH_GTE":
                r = x >= th
            elif rule == "BRANCH_GT":
                r = x > th
            elif rule == "BRANCH_EQ":
                r = x == th
            elif rule == "BRANCH_NEQ":
                r = x != th
            else:
                raise ValueError(
                    f"Unexpected rule {rule!r} for node index {index}.")
            nid = (self.atts.nodes_truenodeids[index]
                   if r else self.atts.nodes_falsenodeids[index])
            index = self.node_index[tree_id, nid]
        return index

    def leave_index_tree(self, X):
        """
        Computes the leave index for all trees.
        """
        if len(X.shape) == 1:
            X = X.reshape((1, -1))
        outputs = []
        for row in X:
            outs = []
            for tree_id in self.tree_ids:
                outs.append(self.leaf_index_tree(row, tree_id))
            outputs.append(outs)
        return numpy.array(outputs)


if onnx_opset_version() >= 18:
    from onnx.reference.op_run import OpRun

    class TreeEnsembleRegressor(OpRun):

        op_domain = "ai.onnx.ml"

        def _run(
                self,
                X,
                aggregate_function=None,
                base_values=None,
                base_values_as_tensor=None,
                n_targets=None,
                nodes_falsenodeids=None,
                nodes_featureids=None,
                nodes_hitrates=None,
                nodes_hitrates_as_tensor=None,
                nodes_missing_value_tracks_true=None,
                nodes_modes=None,
                nodes_nodeids=None,
                nodes_treeids=None,
                nodes_truenodeids=None,
                nodes_values=None,
                nodes_values_as_tensor=None,
                post_transform=None,
                target_ids=None,
                target_nodeids=None,
                target_treeids=None,
                target_weights=None,
                target_weights_as_tensor=None):
            nmv = nodes_missing_value_tracks_true
            tr = TreeEnsemble(
                aggregate_function=aggregate_function,
                base_values=base_values,
                base_values_as_tensor=base_values_as_tensor,
                n_targets=n_targets,
                nodes_falsenodeids=nodes_falsenodeids,
                nodes_featureids=nodes_featureids,
                nodes_hitrates=nodes_hitrates,
                nodes_hitrates_as_tensor=nodes_hitrates_as_tensor,
                nodes_missing_value_tracks_true=nmv,
                nodes_modes=nodes_modes,
                nodes_nodeids=nodes_nodeids,
                nodes_treeids=nodes_treeids,
                nodes_truenodeids=nodes_truenodeids,
                nodes_values=nodes_values,
                nodes_values_as_tensor=nodes_values_as_tensor,
                post_transform=post_transform,
                target_ids=target_ids,
                target_nodeids=target_nodeids,
                target_treeids=target_treeids,
                target_weights=target_weights,
                target_weights_as_tensor=target_weights_as_tensor)
            leaves_index = tr.leave_index_tree(X)
            res = numpy.empty((leaves_index.shape[0], n_targets),
                              dtype=X.dtype)
            if base_values is None:
                res[:, :] = 0
            else:
                res[:, :] = base_values.reshape((1, -1))
            target_index = {(tid, nid): i for i, (tid, nid) in enumerate(
                zip(target_treeids, target_nodeids))}
            for i in range(res.shape[0]):
                indices = leaves_index[i]
                t_index = [target_index[nodes_treeids[i],
                                        nodes_nodeids[i]]
                           for i in indices]
                for it in t_index:
                    res[i, target_ids[it]] += tr.atts.target_weights[it]
            if post_transform in (None, 'NONE'):
                return (res, )
            raise NotImplementedError(
                f"post_transform={post_transform!r} not implemented.")

    if __name__ == "__main__":
        from onnx.reference import ReferenceEvaluator
        from sklearn.datasets import make_regression
        from sklearn.ensemble import RandomForestRegressor
        from skl2onnx import to_onnx
        X, y = make_regression(100, n_features=3)
        model = RandomForestRegressor(max_depth=3, n_estimators=2).fit(X, y)
        onx = to_onnx(model, X.astype(numpy.float32))
        tr = ReferenceEvaluator(onx, new_ops=[TreeEnsembleRegressor])
        print(tr.run(None, {'X': X[:5].astype(numpy.float32)}))
        print(model.predict(X[:5].astype(numpy.float32)))
