# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from .graph_state import GraphState


class AlgebraSymbolic:

    def __init__(self):
        pass

    class Array:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, index):
            return AlgebraSymbolic.ArrayPos(self, index)

    class ArrayPosition:
        def __init__(self, array, position):
            self.array = array
            self.position = position

        def __getitem__(self, index):
            return AlgebraSymbolic.ArrayPos(self, index)

    @property
    def In(self):
        return AlgebraSymbolic.Array("I")

    @property
    def Out(self):
        return AlgebraSymbolic.Array("O")


class OnnxOperator:
    """
    Creates functions easier to use in order to
    create converters.
    """

    def __init__(self, *inputs, outputs=None, **kwargs):
        self.state = None
        self.inputs = list(inputs)
        self.kwargs = kwargs
        if outputs is None:
            # It means intermediate outputs. We suppose there is one.
            outputs = [None]
        self.known_outputs = outputs

    def _check_names(self, scope):
        for i in range(len(self.known_outputs)):
            o = self.known_outputs[i]
            if o is None:
                self.known_outputs[i] = scope.get_unique_variable_name(
                    self.__class__.__name__ + '-o')

    def add_to(self, scope, container):
        if self.state is None:
            if self.kwargs.get('op_version', '') is None:
                kwargs = self.kwargs.copy()
                del kwargs['op_version']
            else:
                kwargs = self.kwargs

            self._check_names(scope)
            self.state = GraphState(self.inputs, self.known_outputs,
                                    self.__class__.__name__,
                                    scope, container, None,
                                    **self.kwargs)
            self.state.run()

    @property
    def outputs(self):
        if self.state is None:
            raise RuntimeError("Method add was not called.")
        return self.state.outputs


OP = AlgebraSymbolic()
