#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import mlir.ir
from circt.dialects import hw


class _Types:
  """Python syntactic sugar to get types"""

  @staticmethod
  def __getattr__(name: str) -> mlir.ir.Type:
    return mlir.ir.Type.parse(name)

  @staticmethod
  def int(width: int):
    return mlir.ir.IntegerType.get_signless(width)

  @staticmethod
  def array(inner: mlir.ir.Type, size: int) -> hw.ArrayType:
    return hw.ArrayType.get(inner, size)

  @staticmethod
  def struct(members) -> hw.StructType:
    if isinstance(members, dict):
      return hw.StructType.get(list(members.items()))
    if isinstance(members, list):
      return hw.StructType.get(members)
    raise TypeError("Expected either list or dict.")


types = _Types()


def dim(inner_type_or_bitwidth, *size: int) -> hw.ArrayType:
  """Creates a multidimensional array from innermost to outermost dimension."""
  if isinstance(inner_type_or_bitwidth, int):
    ret = types.int(inner_type_or_bitwidth)
  else:
    ret = inner_type_or_bitwidth
  for s in size:
    ret = hw.ArrayType.get(ret, s)
  return ret
