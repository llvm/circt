#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import OrderedDict

import mlir.ir
from circt.dialects import hw
import circt.support


class _Types:
  """Python syntactic sugar to get types"""

  TYPE_SCOPE = "pycde"

  def __init__(self):
    self.registered_aliases = OrderedDict()

  def __getattr__(self, name: str) -> mlir.ir.Type:
    return self.wrap(mlir.ir.Type.parse(name))

  def int(self, width: int, name: str = None):
    return self.wrap(mlir.ir.IntegerType.get_signless(width), name)

  def array(self,
            inner: mlir.ir.Type,
            size: int,
            name: str = None) -> hw.ArrayType:
    return self.wrap(hw.ArrayType.get(inner, size), name)

  def struct(self, members, name: str = None) -> hw.StructType:
    members = OrderedDict(members)
    if isinstance(members, dict):
      return self.wrap(hw.StructType.get(list(members.items())), name)
    if isinstance(members, list):
      return self.wrap(hw.StructType.get(members), name)
    raise TypeError("Expected either list or dict.")

  def wrap(self, type, name=None):
    if name is not None:
      type = self._create_alias(type, name)
    return PyCDEType(type)

  def _create_alias(self, inner_type, name):
    alias = hw.TypeAliasType.get(_Types.TYPE_SCOPE, name, inner_type)

    if name in self.registered_aliases:
      if alias != self.registered_aliases[name]:
        raise RuntimeError(
            f"Re-defining type alias for {name}! "
            f"Given: {inner_type}, "
            f"existing: {self.registered_aliases[name].inner_type}")
      return self.registered_aliases[name]

    self.registered_aliases[name] = alias
    return alias

  def declare_types(self, mod):
    if not self.registered_aliases:
      return

    type_scopes = [
        op for op in mod.body.operations if isinstance(op, hw.TypeScopeOp)
    ]
    if len(type_scopes) == 0:
      with mlir.ir.InsertionPoint.at_block_begin(mod.body):
        type_scopes.append(hw.TypeScopeOp.create(self.TYPE_SCOPE))

    assert len(type_scopes) == 1
    type_scope = type_scopes[0]
    with mlir.ir.InsertionPoint(type_scope.body):
      for (name, type) in self.registered_aliases.items():
        declared_aliases = [
            op for op in type_scope.body.operations
            if isinstance(op, hw.TypedeclOp) and op.sym_name.value == name
        ]
        if len(declared_aliases) != 0:
          continue
        hw.TypedeclOp.create(name, type.inner_type)


types = _Types()


def dim(inner_type_or_bitwidth, *size: int, name: str = None) -> hw.ArrayType:
  """Creates a multidimensional array from innermost to outermost dimension."""
  if isinstance(inner_type_or_bitwidth, int):
    ret = types.int(inner_type_or_bitwidth)
  else:
    ret = inner_type_or_bitwidth
  for s in size:
    ret = hw.ArrayType.get(ret, s)
  return types.wrap(ret, name)


# Parameterized class to subclass 'type'.
def PyCDEType(type):
  if type.__class__.__name__ == "_PyCDEType":
    return type
  type = circt.support.type_to_pytype(type)

  class _PyCDEType(type.__class__):
    """Add methods to an MLIR type class."""

    @property
    def strip(self):
      """Return self or inner type."""
      if isinstance(type, hw.TypeAliasType):
        return PyCDEType(self.inner_type)
      else:
        return self

    def create(self, obj, name: str = None):
      """Create a Value of this type from a python object."""
      from .support import obj_to_value
      v = obj_to_value(obj, self, self)
      if name is not None:
        v.name = name
      return v

  return _PyCDEType(type)
