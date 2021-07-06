#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import mlir.ir
from circt.dialects import hw


class _Types:
  """Python syntactic sugar to get types"""

  TYPE_SCOPE = "pycde"

  def __init__(self):
    self.registered_aliases = {}

  def __getattr__(self, name: str) -> mlir.ir.Type:
    return mlir.ir.Type.parse(name)

  def int(self, width: int, name: str = None):
    return self.maybe_create_alias(mlir.ir.IntegerType.get_signless(width),
                                   name)

  def array(self,
            inner: mlir.ir.Type,
            size: int,
            name: str = None) -> hw.ArrayType:
    return self.maybe_create_alias(hw.ArrayType.get(inner, size), name)

  def struct(self, members, name: str = None) -> hw.StructType:
    if isinstance(members, dict):
      return self.maybe_create_alias(hw.StructType.get(list(members.items())),
                                     name)
    if isinstance(members, list):
      return self.maybe_create_alias(hw.StructType.get(members), name)
    raise TypeError("Expected either list or dict.")

  def maybe_create_alias(self, inner_type, name):
    if name is not None:
      alias = hw.TypeAliasType.get(_Types.TYPE_SCOPE, name, inner_type)

      if name in self.registered_aliases:
        if alias != self.registered_aliases[name]:
          raise RuntimeError(
              f"Re-defining type alias for {name}! "\
              f"Given: {inner_type}, "\
              f"existing: {self.registered_aliases[name].inner_type}"
          )
        return self.registered_aliases[name]

      self.registered_aliases[name] = alias
      return alias

    return inner_type


types = _Types()


def dim(inner_type_or_bitwidth, *size: int, name: str = None) -> hw.ArrayType:
  """Creates a multidimensional array from innermost to outermost dimension."""
  if isinstance(inner_type_or_bitwidth, int):
    ret = types.int(inner_type_or_bitwidth)
  else:
    ret = inner_type_or_bitwidth
  for s in size:
    ret = hw.ArrayType.get(ret, s)
  return types.maybe_create_alias(ret, name)
