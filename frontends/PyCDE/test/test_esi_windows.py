# RUN: %PYTHON% %s | FileCheck %s

from pycde.signals import Struct
from pycde.types import Window, StructType, Bits, List, TypeAlias

# Test default_of with a simple struct
# CHECK: Window<"default_window", struct { a: Bits<8>, b: Bits<8>}, frames=[Frame(None, ['a', 'b'])]>
simple_struct = StructType({"a": Bits(8), "b": Bits(8)})
w1 = Window.default_of(simple_struct)
print(w1)

# Test default_of with a struct containing a list
# CHECK: Window<"default_window", list_struct, frames=[Frame(None, ['a', 'l'])]>
list_struct = StructType({"a": Bits(8), "l": List(Bits(8))})
w2 = Window.default_of(TypeAlias(list_struct, "list_struct"))
print(w2)

# Test default_of with a non-struct type
# CHECK: Window<"default_window", struct { data: Bits<8>}, frames=[Frame(None, ['data'])]>
w3 = Window.default_of(Bits(8))
print(w3)

# Test default_of with a struct containing a list in the middle
# CHECK: Window<"default_window", mid_list, frames=[Frame(None, ['a', 'b', 'l'])]>
mid_list_struct = StructType({"a": Bits(8), "l": List(Bits(8)), "b": Bits(8)})
TypeAlias(mid_list_struct, "mid_list")
w4 = Window.default_of(TypeAlias(mid_list_struct, "mid_list"))
print(w4)


class ListRegStruct(Struct):
  x: Bits(4)
  y: List(Bits(4))


# CHECK: Window<"default_window", ListRegStruct, frames=[Frame(None, ['x', 'y'])]>
w5 = Window.default_of(ListRegStruct)
print(w5)
# CHECK: ListRegStruct_default_window
print(w5.lowered_type)
