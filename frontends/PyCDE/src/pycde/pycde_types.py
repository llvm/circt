#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import OrderedDict

from .value import RegularValue, ListValue, StructValue, BitVectorValue

import mlir.ir
from circt.dialects import hw, sv
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

    type_scopes = list()
    for op in mod.body.operations:
      if isinstance(op, hw.TypeScopeOp):
        type_scopes.append(op)
        continue
      if isinstance(op, sv.IfDefOp):
        if len(op.elseRegion.blocks) == 0:
          continue
        for ifdef_op in op.elseRegion.blocks[0]:
          if isinstance(ifdef_op, hw.TypeScopeOp):
            type_scopes.append(ifdef_op)

    assert len(type_scopes) <= 1
    if len(type_scopes) == 1:
      type_scope = type_scopes[0]
    else:
      with mlir.ir.InsertionPoint.at_block_begin(mod.body):
        guard_name = "__PYCDE_TYPES__"
        sv.VerbatimOp(mlir.ir.StringAttr.get("`ifndef " + guard_name), [],
                      mlir.ir.ArrayAttr.get([]))
        sv.VerbatimOp(mlir.ir.StringAttr.get("`define " + guard_name), [],
                      mlir.ir.ArrayAttr.get([]))
        type_scope = hw.TypeScopeOp.create(self.TYPE_SCOPE)
        sv.VerbatimOp(mlir.ir.StringAttr.get("`endif // " + guard_name), [],
                      mlir.ir.ArrayAttr.get([]))

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


# Parameterized class to subclass 'type'.
def PyCDEType(type):
  if isinstance(type, Type):
    return type
  type = circt.support.type_to_pytype(type)
  if isinstance(type, hw.ArrayType):
    return ArrayType(type)
  if isinstance(type, hw.StructType):
    return StructType(type)
  if isinstance(type, hw.TypeAliasType):
    return TypeAliasType(type)
  if isinstance(type, mlir.ir.IntegerType):
    return BitVectorType(type)
  return Type(type)


class Type(mlir.ir.Type):
  """PyCDE type hierarchy root class. Can wrap any MLIR/CIRCT type, but can only
  do anything useful with types for which subclasses exist."""
  __slots__ = ["_type"]

  def __init__(self, type):
    super().__init__(type)
    self._type = circt.support.type_to_pytype(type)

  @property
  def strip(self):
    return self

  def __call__(self, value_obj, name: str = None):
    """Create a Value of this type from a python object."""
    from .support import _obj_to_value
    v = _obj_to_value(value_obj, self._type, self._type)
    if name is not None:
      v.name = name
    return v

  def _get_value_class(self):
    """Return the class which should be instantiated to create a Value."""
    return RegularValue

  def get_value(self, value):
    """Get a pycde.Value."""
    # Separating out the instantiation from figuring out which Value class to
    # instantiate allows us to support type aliases. They need to construct the
    # Value class for their inner type, but with the type of the typealias.
    return self._get_value_class()(value, self)


class TypeAliasType(Type):

  @property
  def name(self):
    return self._type.name

  @property
  def inner_type(self):
    return PyCDEType(self._type.inner_type)

  def __str__(self):
    return self.name

  @property
  def strip(self):
    return PyCDEType(self._type.inner_type)

  def _get_value_class(self):
    return self.strip._get_value_class()

  def wrap(self, value):
    return self(value)


class ArrayType(Type):

  @property
  def element_type(self):
    return PyCDEType(self._type.element_type)

  @property
  def size(self):
    return self._type.size

  def __len__(self):
    return self.size

  def _get_value_class(self):
    return ListValue

  def __str__(self) -> str:
    return f"[{self.size}]{self.element_type}"


class StructType(Type):

  @property
  def fields(self):
    return self._type.get_fields()

  def __getattr__(self, attrname: str):
    for field in self.fields:
      if field[0] == attrname:
        return PyCDEType(self._type.get_field(attrname))
    return super().__getattribute__(attrname)

  def _get_value_class(self):
    return StructValue

  def __str__(self) -> str:
    ret = "struct { "
    first = True
    for field in self.fields:
      if first:
        first = False
      else:
        ret += ", "
      ret += field[0] + ": " + str(field[1])
    ret += "}"
    return ret


class BitVectorType(Type):

  @property
  def width(self):
    return self._type.width

  def _get_value_class(self):
    return BitVectorValue


def dim(inner_type_or_bitwidth, *size: int, name: str = None) -> ArrayType:
  """Creates a multidimensional array from innermost to outermost dimension."""
  if isinstance(inner_type_or_bitwidth, int):
    ret = PyCDEType(mlir.ir.IntegerType.get_signless(inner_type_or_bitwidth))
  else:
    ret = inner_type_or_bitwidth
  for s in size:
    ret = PyCDEType(hw.ArrayType.get(ret, s))
  return types.wrap(ret, name)
