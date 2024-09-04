# SPDX-License-Identifier: Apache-2.0
"""
Helpers to test runtimes.
"""

import numpy as np
from onnx import numpy_helper
from onnx.defs import onnx_opset_version


def _to_str(s):
    if isinstance(s, bytes):
        return s.decode("utf-8")
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
    raise NotImplementedError("Unable to return a value for attribute %r." % attr)


class TreeEnsembleAttributes:
    def __init__(self):
        self._names = []

    def add(self, name, value):
        if not name.endswith("_as_tensor"):
            self._names.append(name)
        if isinstance(value, list):
            if name in {
                "base_values",
                "class_weights",
                "nodes_values",
                "nodes_hitrates",
            }:
                value = np.array(value, dtype=np.float32)
            elif name.endswith("as_tensor"):
                value = np.array(value)
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
        self.root_index = {tid: len(self.atts.nodes_treeids) for tid in self.tree_ids}
        for index, tree_id in enumerate(self.atts.nodes_treeids):
            self.root_index[tree_id] = min(self.root_index[tree_id], index)
        self.node_index = {
            (tid, nid): i
            for i, (tid, nid) in enumerate(
                zip(self.atts.nodes_treeids, self.atts.nodes_nodeids)
            )
        }

    def __str__(self):
        rows = ["TreeEnsemble", f"root_index={self.root_index}", str(self.atts)]
        return "\n".join(rows)

    def leaf_index_tree(self, X, tree_id):
        """
        Computes the leaf index for one tree.
        """
        index = self.root_index[tree_id]
        while self.atts.nodes_modes[index] != "LEAF":
            x = X[self.atts.nodes_featureids[index]]
            if np.isnan(x):
                r = self.atts.nodes_missing_value_tracks_true[index] >= 1
            else:
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
                        f"Unexpected rule {rule!r} for node index {index}."
                    )
            nid = (
                self.atts.nodes_truenodeids[index]
                if r
                else self.atts.nodes_falsenodeids[index]
            )
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
        return np.array(outputs)


if onnx_opset_version() >= 18:
    from onnx.reference.op_run import OpRun

    try:
        from .reference_implementation_helper import ComputeProbit, write_scores
    except ImportError:
        from reference_implementation_helper import ComputeProbit, write_scores

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
            target_weights_as_tensor=None,
        ):
            nmv = nodes_missing_value_tracks_true
            tr = TreeEnsemble(
                base_values=base_values,
                base_values_as_tensor=base_values_as_tensor,
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
                target_weights=target_weights,
                target_weights_as_tensor=target_weights_as_tensor,
            )
            self._tree = tr
            leaves_index = tr.leave_index_tree(X)
            res = np.empty((leaves_index.shape[0], n_targets), dtype=X.dtype)
            if base_values is None:
                res[:, :] = 0
            else:
                res[:, :] = np.array(base_values).reshape((1, -1))

            target_index = {}
            for i, (tid, nid) in enumerate(zip(target_treeids, target_nodeids)):
                if (tid, nid) not in target_index:
                    target_index[tid, nid] = []
                target_index[tid, nid].append(i)
            for i in range(res.shape[0]):
                indices = leaves_index[i]
                t_index = [
                    target_index[nodes_treeids[i], nodes_nodeids[i]] for i in indices
                ]
                if aggregate_function == "SUM":
                    for its in t_index:
                        for it in its:
                            res[i, target_ids[it]] += tr.atts.target_weights[it]
                else:
                    raise NotImplementedError(
                        f"aggregate_transform={aggregate_function!r} "
                        f"not supported yet."
                    )

            if post_transform in (None, "NONE"):
                return (res,)
            raise NotImplementedError(
                f"post_transform={post_transform!r} not implemented."
            )

    class TreeEnsembleClassifier(OpRun):
        op_domain = "ai.onnx.ml"

        def _run(
            self,
            X,
            base_values=None,
            base_values_as_tensor=None,
            class_ids=None,
            class_nodeids=None,
            class_treeids=None,
            class_weights=None,
            class_weights_as_tensor=None,
            classlabels_int64s=None,
            classlabels_strings=None,
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
        ):
            nmv = nodes_missing_value_tracks_true
            tr = TreeEnsemble(
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
                class_weights=class_weights,
                class_weights_as_tensor=class_weights_as_tensor,
            )
            self._tree = tr
            if X.dtype not in (np.float32, np.float64):
                X = X.astype(np.float32)
            leaves_index = tr.leave_index_tree(X)
            n_classes = max(
                len(classlabels_int64s or []), len(classlabels_strings or [])
            )
            res = np.empty((leaves_index.shape[0], n_classes), dtype=np.float32)
            if base_values is None:
                res[:, :] = 0
            else:
                res[:, :] = np.array(base_values).reshape((1, -1))

            class_index = {}
            for i, (tid, nid) in enumerate(zip(class_treeids, class_nodeids)):
                if (tid, nid) not in class_index:
                    class_index[tid, nid] = []
                class_index[tid, nid].append(i)
            for i in range(res.shape[0]):
                indices = leaves_index[i]
                t_index = [
                    class_index[nodes_treeids[i], nodes_nodeids[i]] for i in indices
                ]
                for its in t_index:
                    for it in its:
                        res[i, class_ids[it]] += tr.atts.class_weights[it]

            # post_transform
            binary = len(set(class_ids)) == 1
            if post_transform in (None, "NONE"):
                if binary:
                    classes = classlabels_int64s or classlabels_strings
                    if res.shape[1] == 1 and len(classes) == 1:
                        new_res = np.zeros((res.shape[0], 2), res.dtype)
                        new_res[:, 1] = res[:, 0]
                        new_res[:, 0] = 1 - new_res[:, 1]
                        res = new_res
                    else:
                        res[:, 1] = res[:, 0]
                        res[:, 0] = 1 - res[:, 1]
                new_scores = res
            elif post_transform == "PROBIT" and n_classes == 1:
                assert res.shape[1] == 1
                res[:, 0] = [ComputeProbit(x) for x in res[:, 0]]
                new_scores = res
            else:
                nc = res.shape[1]
                add_second_class = -1
                if binary and res.shape[1] == 2:
                    res = res[:, :1]
                    if post_transform == "LOGISTIC":
                        add_second_class = 2
                new_scores = np.empty((res.shape[0], nc), dtype=res.dtype)
                for i in range(res.shape[0]):
                    new_scores[i, :] = write_scores(
                        res.shape[1], res[i], post_transform, add_second_class
                    )

            # labels
            labels = np.argmax(new_scores, axis=1).astype(np.int64)
            if classlabels_int64s is not None:
                if len(classlabels_int64s) == 1:
                    if classlabels_int64s[0] == 1:
                        d = {1: 1}
                        labels = np.array([d.get(i, 0) for i in labels], dtype=np.int64)
                    else:
                        raise NotImplementedError(
                            f"classlabels_int64s={classlabels_int64s}, "
                            f"not supported."
                        )
                else:
                    labels = np.array(
                        [classlabels_int64s[i] for i in labels], dtype=np.int64
                    )
            elif classlabels_strings is not None:
                if len(classlabels_strings) == 1:
                    raise NotImplementedError(
                        f"classlabels_strings={classlabels_strings}, not supported."
                    )
                labels = np.array([classlabels_strings[i] for i in labels])

            return labels, new_scores

    if __name__ == "__main__":
        from onnx.reference import ReferenceEvaluator
        from onnx.reference.ops.op_argmax import _ArgMax
        from sklearn.datasets import make_regression, make_classification
        from sklearn.ensemble import (
            RandomForestRegressor,
            RandomForestClassifier,
            BaggingClassifier,
        )
        from skl2onnx import to_onnx
        from reference_implementation_afe import ArrayFeatureExtractor

        class ArgMax(_ArgMax):
            def _run(self, data, axis=None, keepdims=None, select_last_index=None):
                if select_last_index == 0:  # type: ignore
                    return _ArgMax._run(self, data, axis=axis, keepdims=keepdims)
                raise NotImplementedError("Unused in sklearn-onnx.")

        # classification 1
        X, y = make_classification(
            100, n_features=6, n_classes=3, n_informative=3, n_redundant=0
        )
        model = BaggingClassifier().fit(X, y)
        onx = to_onnx(model, X.astype(np.float32), options={"zipmap": False})
        tr = ReferenceEvaluator(
            onx, new_ops=[TreeEnsembleClassifier, ArrayFeatureExtractor, ArgMax]
        )
        print("-----------------------")
        print(tr.run(None, {"X": X[:10].astype(np.float32)}))
        print("--")
        print(model.predict(X[:10].astype(np.float32)))
        print(model.predict_proba(X[:10].astype(np.float32)))
        print("-----------------------")

        # classification 2
        model = RandomForestClassifier(max_depth=3, n_estimators=2).fit(X, y)
        onx = to_onnx(model, X.astype(np.float32), options={"zipmap": False})
        tr = ReferenceEvaluator(onx, new_ops=[TreeEnsembleClassifier])
        print(tr.run(None, {"X": X[:5].astype(np.float32)}))
        print(model.predict(X[:5].astype(np.float32)))
        print(model.predict_proba(X[:5].astype(np.float32)))

        # regression
        X, y = make_regression(100, n_features=3)
        model = RandomForestRegressor(max_depth=3, n_estimators=2).fit(X, y)
        onx = to_onnx(model, X.astype(np.float32))
        tr = ReferenceEvaluator(onx, new_ops=[TreeEnsembleRegressor])
        print(tr.run(None, {"X": X[:5].astype(np.float32)}))
        print(model.predict(X[:5].astype(np.float32)))
