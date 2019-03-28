# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------


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


OP = AlgebraSymbolic()
