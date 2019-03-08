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
    
    class Array:
        def __init__(self, array, position):
            self.array = array
            self.position = position
        def __getitem__(self, index):
            return AlgebraSymbolic.ArrayPos(self, index)
    
    @property
    def I(self):
        return AlgebraSymbolic.Array("I")

    @property
    def O(self):
        return AlgebraSymbolic.Array("O")


class OnnxOperator:
    """
    Creates functions easier to use in order to
    create converters.
    """
    def __init__(self, *inputs, outputs=None, **kwargs):
        self.state = None
        self.inputs = list(inputs)
        self.known_outputs = outputs
        self.kwargs = kwargs

    def add(self, scope, operator, container):
        if self.state is None:
            if self.kwargs.get('op_version', '') is None:
                kwargs = self.kwargs.copy()
                del kwargs['op_version']
            else:
                kwargs = self.kwargs
                
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
