#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .support import get_user_loc

import circt.support as support
from circt.dialects import hw, seq

import mlir.ir as ir


class Value:

  @staticmethod
  def get(value, type=None):
    from .pycde_types import PyCDEType
    value = support.get_value(value)
    if type is None:
      type = value.type
    type = PyCDEType(type)

    if isinstance(type.strip, hw.ArrayType):
      return ListValue(value, type)
    if isinstance(type.strip, hw.StructType):
      return StructValue(value, type)
    return RegularValue(value, type)

  def reg(self, clk, rst=None):
    return seq.reg(self.value, clk, rst)


class RegularValue(Value):

  def __init__(self, value, type):
    self.value = value
    self.type = type


class ListValue(Value):

  def __init__(self, value, type):
    self.value = value
    self.type = type

  def __getitem__(self, sub):
    if isinstance(sub, int):
      idx = int(sub)
      if idx >= self.type.size:
        raise ValueError("Subscript out-of-bounds")
    else:
      idx = support.get_value(sub)
      if idx is None or not isinstance(support.type_to_pytype(idx.type),
                                       ir.IntegerType):
        raise TypeError("Subscript on array must be either int or MLIR int"
                        f" Value, not {type(sub)}.")
    with get_user_loc():
      return Value.get(hw.ArrayGetOp.create(self.value, idx))

  def __len__(self):
    return self.type.strip.size


class StructValue(Value):

  def __init__(self, value, type):
    self.value = value
    self.type = type

  def __getitem__(self, sub):
    fields = self.type.strip.get_fields()
    if sub not in [name for name, _ in fields]:
      raise ValueError(f"Struct field '{sub}' not found in {self.type}")
    with get_user_loc():
      return Value.get(hw.StructExtractOp.create(self.value, sub))

  def __getattr__(self, attr):
    ty = self.type.strip
    fields = ty.get_fields()
    if attr in [name for name, _ in fields]:
      with get_user_loc():
        return Value.get(hw.StructExtractOp.create(self.value, attr))
    raise AttributeError(f"'Value' object has no attribute '{attr}'")
