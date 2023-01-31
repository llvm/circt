#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import OrderedDict

from .value import (BitsSignal, ChannelValue, ClockSignal, ArraySignal,
                    SIntValue, UIntValue, StructValue, UntypedSignal,
                    InOutSignal, Value)

from .circt import ir, support
from .circt.dialects import esi, hw, sv

from typing import Union
import typing


class _Types:
  """Python syntactic sugar to get types"""

  def __init__(self):
    self.registered_aliases = OrderedDict()

  def __getattr__(self, name: str) -> ir.Type:
    return self.wrap(ir.Type.parse(name))

  def int(self, width: int, name: str = None):
    return self.wrap(Bits(width), name)

  def array(self, inner: ir.Type, size: int, name: str = None) -> hw.ArrayType:
    return self.wrap(Array(inner, size), name)

  def inout(self, inner: ir.Type):
    return self.wrap(InOut(inner))

  def channel(self, inner):
    return self.wrap(Channel(inner))

  def struct(self, members, name: str = None) -> hw.StructType:
    s = Struct(members)
    if name is None:
      return s
    return TypeAlias(s, name)

  @property
  def any(self):
    return self.wrap(Any())

  def wrap(self, type, name=None):
    if name is not None:
      type = TypeAlias(type, name)
    return _FromCirctType(type)


types = _Types()


class Type:
  """PyCDE type hierarchy root class. Can wrap any MLIR/CIRCT type, but can only
  do anything useful with types for which subclasses exist."""

  # Global Type cache.
  _cache: typing.Dict[typing.Tuple[type, ir.Type], "Type"] = {}

  def __new__(cls, circt_type: ir.Type) -> "Type":
    """Look up a type in the Type cache. If present, return it. If not, create
    it and put it in the cache."""
    assert isinstance(circt_type, ir.Type)
    cache_key = (cls, circt_type)
    if cache_key not in Type._cache:
      t = super(Type, cls).__new__(cls)
      t._type = circt_type
      Type._cache[cache_key] = t
    return Type._cache[cache_key]

  def __init__(self, *args, **kwargs) -> None:
    pass

  @property
  def strip(self):
    return self

  @property
  def bitwidth(self):
    return hw.get_bitwidth(self._type)

  def __call__(self, value_obj, name: str = None):
    """Create a Value of this type from a python object."""
    from .support import _obj_to_value
    v = _obj_to_value(value_obj, self, self)
    if name is not None:
      v.name = name
    return v

  def _get_value_class(self):
    """Return the class which should be instantiated to create a Value."""
    return UntypedSignal

  def __repr__(self):
    return self._type.__repr__()


def _FromCirctType(type: Union[ir.Type, Type]) -> Type:
  if isinstance(type, Type):
    return type
  type = support.type_to_pytype(type)
  if isinstance(type, hw.ArrayType):
    return Type.__new__(Array, type)
  if isinstance(type, hw.StructType):
    return Type.__new__(Struct, type)
  if isinstance(type, hw.TypeAliasType):
    return Type.__new__(TypeAlias, type)
  if isinstance(type, hw.InOutType):
    return Type.__new__(InOut, type)
  if isinstance(type, ir.IntegerType):
    if type.is_signed:
      return Type.__new__(SInt, type)
    elif type.is_unsigned:
      return Type.__new__(UInt, type)
    else:
      return Type.__new__(Bits, type)
  if isinstance(type, esi.AnyType):
    return Type.__new__(Any, type)
  if isinstance(type, esi.ChannelType):
    return Type.__new__(Channel, type)
  return Type(type)


class InOut(Type):

  def __new__(cls, element_type: Type):
    return super(InOut, cls).__new__(cls, hw.InOutType.get(element_type._type))

  @property
  def element_type(self) -> Type:
    return _FromCirctType(self._type.element_type)

  def _get_value_class(self):
    return InOutSignal


class TypeAlias(Type):

  TYPE_SCOPE = "pycde"
  RegisteredAliases: typing.Optional[OrderedDict] = None

  def __new__(cls, inner_type: Type, name: str):
    if not TypeAlias.RegisteredAliases:
      TypeAlias.RegisteredAliases = OrderedDict()
    alias = hw.TypeAliasType.get(TypeAlias.TYPE_SCOPE, name, inner_type._type)

    if name in TypeAlias.RegisteredAliases:
      if alias != TypeAlias.RegisteredAliases[name]:
        raise RuntimeError(
            f"Re-defining type alias for {name}! "
            f"Given: {inner_type}, "
            f"existing: {TypeAlias.RegisteredAliases[name].inner_type}")
      return TypeAlias.RegisteredAliases[name]

    TypeAlias.RegisteredAliases[name] = alias
    return super(TypeAlias, cls).__new__(cls, alias)

  @staticmethod
  def declare_aliases(mod):
    if TypeAlias.RegisteredAliases is None:
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
      with ir.InsertionPoint.at_block_begin(mod.body):
        guard_name = "__PYCDE_TYPES__"
        sv.VerbatimOp(ir.StringAttr.get("`ifndef " + guard_name), [],
                      symbols=ir.ArrayAttr.get([]))
        sv.VerbatimOp(ir.StringAttr.get("`define " + guard_name), [],
                      symbols=ir.ArrayAttr.get([]))
        type_scope = hw.TypeScopeOp.create(TypeAlias.TYPE_SCOPE)
        sv.VerbatimOp(ir.StringAttr.get("`endif // " + guard_name), [],
                      symbols=ir.ArrayAttr.get([]))

    with ir.InsertionPoint(type_scope.body):
      for (name, type) in TypeAlias.RegisteredAliases.items():
        declared_aliases = [
            op for op in type_scope.body.operations
            if isinstance(op, hw.TypedeclOp) and op.sym_name.value == name
        ]
        if len(declared_aliases) != 0:
          continue
        hw.TypedeclOp.create(name, type.inner_type)

  @property
  def name(self):
    return self._type.name

  @property
  def inner_type(self):
    return _FromCirctType(self._type.inner_type)

  def __str__(self):
    return self.name

  @property
  def strip(self):
    return _FromCirctType(self._type.inner_type)

  def _get_value_class(self):
    return self.strip._get_value_class()

  def wrap(self, value):
    return self(value)


class Array(Type):

  def __new__(cls, element_type: Type, length: int):
    return super(Array,
                 cls).__new__(cls, hw.ArrayType.get(element_type._type, length))

  @property
  def inner_type(self):
    if isinstance(self.element_type, Array):
      return self.element_type.inner_type
    return self.element_type

  @property
  def element_type(self):
    return _FromCirctType(self._type.element_type)

  @property
  def size(self):
    return self._type.size

  @property
  def shape(self):
    _shape = [self.size]
    if isinstance(self.element_type, Array):
      _shape.extend(self.element_type.shape)
    return _shape

  def __len__(self):
    return self.size

  def _get_value_class(self):
    return ArraySignal

  def __str__(self) -> str:
    return f"[{self.size}]{self.element_type}"


class Struct(Type):

  def __new__(cls, fields: typing.Union[typing.List[typing.Tuple[str, Type]],
                                        typing.Dict[str, Type]]):
    if isinstance(fields, dict):
      fields = list(fields.items())
    if not isinstance(fields, list):
      raise TypeError("Expected either list or dict.")
    return super(Struct, cls).__new__(
        cls, hw.StructType.get([(n, t._type) for (n, t) in fields]))

  @property
  def fields(self):
    return [(n, _FromCirctType(t)) for n, t in self._type.get_fields()]

  def __getattr__(self, attrname: str):
    for field in self.fields:
      if field[0] == attrname:
        return _FromCirctType(self._type.get_field(attrname))
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
      ret += f"{field[0]}: {_FromCirctType(field[1])}"
    ret += "}"
    return ret


class BitVectorType(Type):

  @property
  def width(self):
    return self._type.width


class Bits(BitVectorType):

  def __new__(cls, width: int):
    return super(Bits, cls).__new__(
        cls,
        ir.IntegerType.get_signless(width),
    )

  def _get_value_class(self):
    return BitsSignal

  def __repr__(self):
    return f"bits{self.width}"


class SInt(BitVectorType):

  def __new__(cls, width: int):
    return super(SInt, cls).__new__(
        cls,
        ir.IntegerType.get_signed(width),
    )

  def _get_value_class(self):
    return SIntValue

  def __repr__(self):
    return f"sint{self.width}"


class UInt(BitVectorType):

  def __new__(cls, width: int):
    return super(UInt, cls).__new__(
        cls,
        ir.IntegerType.get_unsigned(width),
    )

  def _get_value_class(self):
    return UIntValue

  def __repr__(self):
    return f"uint{self.width}"


class ClockType(Bits):
  """A special single bit to represent a clock. Can't do any special operations
  on it, except enter it as a implicit clock block."""

  # TODO: the 'clock' type isn't represented in CIRCT IR. It may be useful to
  # have it there if for no other reason than being able to round trip this
  # type.

  def __new__(cls):
    super(ClockType, cls).__new__(cls, 1)

  def _get_value_class(self):
    return ClockSignal

  def __repr__(self):
    return "clk"


class Any(Type):

  def __new__(cls):
    return super(Any, cls).__new__(cls, esi.AnyType.get())


class Channel(Type):
  """An ESI channel type."""

  def __new__(cls, inner_type: Type):
    return super(Channel, cls).__new__(cls,
                                       esi.ChannelType.get(inner_type._type))

  @property
  def inner_type(self):
    return _FromCirctType(self._type.inner)

  def _get_value_class(self):
    return ChannelValue

  def __str__(self):
    return f"channel<{self.inner_type}>"

  @property
  def inner(self):
    return self.inner_type

  def wrap(self, value, valid):
    from .dialects import esi
    from .support import _obj_to_value
    value = _obj_to_value(value, self._type.inner)
    valid = _obj_to_value(valid, types.i1)
    wrap_op = esi.WrapValidReadyOp(self._type, types.i1, value.value,
                                   valid.value)
    return Value(wrap_op[0]), BitsSignal(wrap_op[1], types.i1)


def dim(inner_type_or_bitwidth: typing.Union[Type, int],
        *size: typing.List[int],
        name: str = None) -> Array:
  """Creates a multidimensional array from innermost to outermost dimension."""
  if isinstance(inner_type_or_bitwidth, int):
    ret = _FromCirctType(ir.IntegerType.get_signless(inner_type_or_bitwidth))
  elif isinstance(inner_type_or_bitwidth, Type):
    ret = inner_type_or_bitwidth
  else:
    raise ValueError(f"Expected 'Type', not {inner_type_or_bitwidth}")
  for s in size:
    ret = Array(ret, s)
  if name is None:
    return ret
  return TypeAlias(ret, name)
