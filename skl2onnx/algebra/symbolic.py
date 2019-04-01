# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------


class AlgebraSymbolic:

    def __init__(self):
        pass

    class Symbolic:
        def __init__(self, name):
            self.name = name

        def __repr__(self):
            return "%s('%s')" % (self.__class__.__name__, self.name)

    class Array(Symbolic):
        def __getitem__(self, index):
            return AlgebraSymbolic.ArrayPos(self, index)

    class ArrayPosition(Symbolic):
        def __init__(self, array, position):
            self.array = array
            self.position = position

        def __getitem__(self, index):
            return AlgebraSymbolic.ArrayPos(self, index)

        def __repr__(self):
            return "%s('%s', %d)" % (self.__class__.__name__,
                                    self.name, self.position)

    class Input(Symbolic):
        pass

    class Output(Symbolic):
        pass


Symbolic = AlgebraSymbolic()
