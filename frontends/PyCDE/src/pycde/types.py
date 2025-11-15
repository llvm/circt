#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections import OrderedDict
from functools import singledispatchmethod

from .support import clog2, get_user_loc

from .circt import ir, support
from .circt.dialects import esi, hw, seq, sv
from .circt.dialects.esi import ChannelSignaling, ChannelDirection

import typing
from dataclasses import dataclass


class _Types:
  """Python syntactic sugar to get types"""

  def __init__(self):
    self.registered_aliases = OrderedDict()

  def __getattr__(self, name: str) -> ir.Type:
    return self.wrap(_FromCirctType(ir.Type.parse(name)))

  def int(self, width: int, name: str = None):
    return self.wrap(Bits(width), name)

  def array(self, inner: ir.Type, size: int, name: str = None) -> "Array":
    return self.wrap(Array(inner, size), name)

  def inout(self, inner: ir.Type):
    return self.wrap(InOut(inner))

  def channel(self, inner):
    return self.wrap(Channel(inner))

  def struct(self, members, name: str = None) -> "StructType":
    return self.wrap(StructType(members), name)

  @property
  def any(self):
    return self.wrap(Any())

  def wrap(self, type, name=None):
    if name is not None:
      type = TypeAlias(type, name)
    return type


types = _Types()


class Type:
  """PyCDE type hierarchy root class. Can wrap any MLIR/CIRCT type, but can only
  do anything useful with types for which subclasses exist."""

  # Global Type cache.
  _cache: typing.Dict[typing.Tuple[type, ir.Type], "Type"] = {}

  def __new__(cls, circt_type: ir.Type, incl_cls_in_key: bool = True) -> "Type":
    """Look up a type in the Type cache. If present, return it. If not, create
    it and put it in the cache."""
    assert isinstance(circt_type, ir.Type)
    if incl_cls_in_key:
      cache_key = (cls, circt_type)
    else:
      cache_key = circt_type

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
  def bitwidth(self) -> int:
    return hw.get_bitwidth(self._type)

  @property
  def is_hw_type(self) -> bool:
    assert False, "Subclass must override this method"

  def __call__(self, obj, name: str = None) -> "Signal":
    """Create a Value of this type from a python object."""
    assert not isinstance(
        obj, ir.Value
    ), "Not intended to be called on CIRCT Values, only Python objects."
    v = self._from_obj_or_sig(obj)
    if name is not None:
      v.name = name
    return v

  def _from_obj_or_sig(self,
                       obj,
                       alias: typing.Optional["TypeAlias"] = None) -> "Signal":
    """Implement the object-signal conversion wherein 'obj' can be a Signal. If
    'obj' is already a Signal, check its type and return it. Can be overriden by
    subclasses, though calls _from_obj() to do the type-specific const
    conversion so we recommend subclasses override that method."""

    from .signals import Signal
    if isinstance(obj, Signal):
      if obj.type != self:
        raise TypeError(f"Expected signal of type {self} but got {obj.type}")
      return obj
    return self._from_obj(obj, alias)

  def _from_obj(self,
                obj,
                alias: typing.Optional["TypeAlias"] = None) -> "Signal":
    """Do the type-specific object validity checks and return a Signal from the
    object. Can assume the 'obj' is NOT a Signal. Any subclass which wants to be
    created MUST override this method."""

    assert False, "Subclass must override this method"

  def _get_value_class(self):
    """Return the class which should be instantiated to create a Value."""
    from .signals import UntypedSignal
    return UntypedSignal

  def __mul__(self, len: int):
    """Create an array type"""
    return Array(self, len)

  def castable(self, value: Type) -> bool:
    """Return True if a value of 'value' can be cast to this type."""
    if not isinstance(value, Type):
      raise TypeError("Can only cast to a Type")
    return esi.check_inner_type_match(self._type, value._type)

  def __repr__(self):
    return self._type.__repr__()


def _FromCirctType(type: typing.Union[ir.Type, Type]) -> Type:
  if isinstance(type, Type):
    return type
  type = support.type_to_pytype(type)
  if isinstance(type, hw.ArrayType):
    return Type.__new__(Array, type)
  if isinstance(type, hw.StructType):
    return Type.__new__(StructType, type)
  if isinstance(type, hw.TypeAliasType):
    return Type.__new__(TypeAlias, type, incl_cls_in_key=False)
  if isinstance(type, hw.InOutType):
    return Type.__new__(InOut, type)
  if isinstance(type, ir.IntegerType):
    if type.is_signed:
      return Type.__new__(SInt, type)
    elif type.is_unsigned:
      return Type.__new__(UInt, type)
    else:
      return Type.__new__(Bits, type)
  if isinstance(type, seq.ClockType):
    return Type.__new__(ClockType, type)
  if isinstance(type, esi.AnyType):
    return Type.__new__(Any, type)
  if isinstance(type, esi.ChannelType):
    return Type.__new__(Channel, type)
  if isinstance(type, esi.BundleType):
    return Type.__new__(Bundle, type)
  if isinstance(type, esi.ListType):
    return Type.__new__(List, type)
  if hasattr(esi, "WindowType") and isinstance(type, esi.WindowType):
    return Type.__new__(Window, type)
  return Type(type)


class InOut(Type):

  def __new__(cls, element_type: Type):
    return super(InOut, cls).__new__(cls, hw.InOutType.get(element_type._type))

  @property
  def element_type(self) -> Type:
    return _FromCirctType(self._type.element_type)

  @property
  def is_hw_type(self) -> bool:
    return True

  def _get_value_class(self):
    from .signals import InOutSignal
    return InOutSignal

  def __repr__(self):
    return f"InOut<{repr(self.element_type)}"


class TypeAlias(Type):

  TYPE_SCOPE = "pycde"
  RegisteredAliases: typing.Optional[OrderedDict] = None

  def __new__(cls, inner_type: Type, name: str):
    if TypeAlias.RegisteredAliases is None:
      TypeAlias.RegisteredAliases = OrderedDict()

    if name in TypeAlias.RegisteredAliases:
      if inner_type._type != TypeAlias.RegisteredAliases[name].inner_type:
        raise RuntimeError(
            f"Re-defining type alias for {name}! "
            f"Given: {inner_type}, "
            f"existing: {TypeAlias.RegisteredAliases[name].inner_type}")
      alias = TypeAlias.RegisteredAliases[name]
    else:
      alias = hw.TypeAliasType.get(TypeAlias.TYPE_SCOPE, name, inner_type._type)
      TypeAlias.RegisteredAliases[name] = alias

    return super(TypeAlias, cls).__new__(cls, alias, incl_cls_in_key=False)

  @property
  def is_hw_type(self) -> bool:
    return self.inner_type.is_hw_type

  @staticmethod
  def declare_aliases(mod):
    if TypeAlias.RegisteredAliases is None:
      return

    guard_name = "__PYCDE_TYPES__"
    type_scope_attr = ir.StringAttr.get(TypeAlias.TYPE_SCOPE)
    type_scopes = list()
    for op in mod.body.operations:
      if isinstance(op, hw.TypeScopeOp) and op.sym_name == type_scope_attr:
        type_scopes.append(op)
        continue
      if isinstance(op, sv.IfDefOp):
        if len(op.elseRegion.blocks) == 0:
          continue
        for else_block_op in op.elseRegion.blocks[0]:
          if isinstance(
              else_block_op,
              hw.TypeScopeOp) and else_block_op.sym_name == type_scope_attr:
            type_scopes.append(else_block_op)

    assert len(type_scopes) <= 1
    if len(type_scopes) == 1:
      type_scope = type_scopes[0]
    else:
      with ir.InsertionPoint.at_block_begin(mod.body):
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
            if isinstance(op, hw.TypedeclOp) and
            ir.StringAttr(op.sym_name).value == name
        ]
        if len(declared_aliases) != 0:
          continue
        hw.TypedeclOp.create(name, type.inner_type)

  @property
  def name(self) -> str:
    return self._type.name

  @property
  def inner_type(self):
    return _FromCirctType(self._type.inner_type)

  def __repr__(self):
    return f"TypeAlias<'{self.name}', {repr(self.inner_type)}"

  def __str__(self):
    return self.name

  @property
  def strip(self):
    return _FromCirctType(self._type.inner_type)

  def _get_value_class(self):
    return self.strip._get_value_class()

  def wrap(self, value):
    return self(value)

  def _from_obj(self, obj, alias: typing.Optional["TypeAlias"] = None):
    return self.inner_type._from_obj_or_sig(obj, alias=self)


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
  def is_hw_type(self) -> bool:
    return True

  @property
  def size(self):
    return self._type.size

  @property
  def select_bits(self) -> int:
    return clog2(self.size)

  @property
  def shape(self):
    _shape = [self.size]
    if isinstance(self.element_type, Array):
      _shape.extend(self.element_type.shape)
    return _shape

  def __len__(self):
    return self.size

  def _get_value_class(self):
    from .signals import ArraySignal
    return ArraySignal

  def __repr__(self) -> str:
    return f"Array({self.size}, {self.element_type})"

  def __str__(self) -> str:
    return f"{self.element_type}[{self.size}]"

  def _from_obj(self, obj, alias: typing.Optional[TypeAlias] = None):
    from .dialects import hw
    if not isinstance(obj, (list, tuple)):
      raise ValueError(
          f"Arrays can only be created from lists or tuples, not '{type(obj)}'")
    if len(obj) != self.size:
      raise ValueError("List must have same size as array "
                       f"{len(obj)} vs {self.size}")
    elemty = self.element_type
    list_of_vals = list(map(lambda x: elemty._from_obj_or_sig(x), obj))
    with get_user_loc():
      # CIRCT's ArrayCreate op takes the array in reverse order.
      return hw.ArrayCreateOp(reversed(list_of_vals))


class StructType(Type):

  def __new__(
      cls, fields: typing.Union[typing.List[typing.Tuple[str, Type]],
                                typing.Dict[str, Type]]
  ) -> StructType:
    if len(fields) == 0:
      raise ValueError("Structs must have at least one field.")
    if isinstance(fields, dict):
      fields = list(fields.items())
    if not isinstance(fields, list):
      raise TypeError("Expected either list or dict.")
    return super(StructType, cls).__new__(
        cls, hw.StructType.get([(n, t._type) for (n, t) in fields]))

  @property
  def is_hw_type(self) -> bool:
    return True

  @property
  def fields(self):
    return [(n, _FromCirctType(t)) for n, t in self._type.get_fields()]

  def __getattr__(self, attrname: str):
    for field in self.fields:
      if field[0] == attrname:
        return _FromCirctType(self._type.get_field(attrname))
    return super().__getattribute__(attrname)

  def _get_value_class(self):
    from .signals import StructSignal
    return StructSignal

  def _from_obj(self, x, alias: typing.Optional[TypeAlias] = None):
    from .dialects import hw
    if not isinstance(x, dict):
      raise ValueError(
          f"Structs can only be created from dicts, not '{type(x)}'")
    elem_name_values = []
    for (fname, ftype) in self.fields:
      if fname not in x:
        raise ValueError(f"Could not find expected field: {fname}")
      v = ftype._from_obj_or_sig(x[fname])
      elem_name_values.append((fname, v))
      x.pop(fname)
    if len(x) > 0:
      raise ValueError(f"Extra fields specified: {x}")

    result_type = self if alias is None else alias
    with get_user_loc():
      return hw.StructCreateOp(elem_name_values, result_type=result_type._type)

  def __repr__(self) -> str:
    ret = "struct { "
    first = True
    for field in self.fields:
      if first:
        first = False
      else:
        ret += ", "
      ret += f"{field[0]}: {field[1]}"
    ret += "}"
    return ret


class RegisteredStruct(TypeAlias):
  """Represents a named struct with a custom signal class. Primarily used by
  `value.Struct`."""

  def __new__(cls, fields: typing.List[typing.Tuple[str, Type]], name: str,
              value_class):
    inner_type = StructType(fields)
    inst = super().__new__(cls, inner_type, name)
    inst._value_class = value_class
    return inst

  def __call__(self, **kwargs):
    return self._from_obj_or_sig(kwargs)

  @property
  def fields(self):
    return self.inner_type.fields

  def _get_value_class(self):
    return self._value_class

  @property
  def is_hw_type(self) -> bool:
    return True


class BitVectorType(Type):

  @property
  def width(self):
    return self._type.width

  @property
  def is_hw_type(self) -> bool:
    return True

  def _from_obj_check(self, x):
    """This functionality can be shared by all the int types."""
    if not isinstance(x, int):
      raise ValueError(f"{type(self).__name__} can only be created from ints, "
                       f"not {type(x).__name__}")
    signed_bit = 1 if isinstance(self, SInt) else 0
    if x.bit_length() + signed_bit > self.width:
      raise ValueError(f"{x} overflows type {self}")

  def __repr__(self) -> str:
    return f"{type(self).__name__}<{self.width}>"


class Bits(BitVectorType):

  def __new__(cls, width: int):
    if width < 0:
      raise ValueError("Bits width must be non-negative")
    return super(Bits, cls).__new__(
        cls,
        ir.IntegerType.get_signless(width),
    )

  def _get_value_class(self):
    from .signals import BitsSignal
    return BitsSignal

  def _from_obj(self, x: int, alias: typing.Optional[TypeAlias] = None):
    # The MLIR python bindings don't support IntegerAttributes with unsigned
    # 64-bit values since the bindings accept signed integers so an unsigned
    # 64-bit integer overflows that.
    # https://github.com/llvm/llvm-project/issues/128072
    # So we need to work around that here using concats.
    # TODO: Adapt this to UInt and SInt.
    from .dialects import hw
    self._from_obj_check(x)
    circt_type = self if alias is None else alias
    if x.bit_length() > 63:
      # Split the int into an array of 32-bit ints.
      chunks = [(x >> i) & 0xFFFFFFFF for i in range(0, x.bit_length(), 32)]
      last_bitwidth = self.bitwidth % 32
      if last_bitwidth == 0:
        last_bitwidth = 32
      chunk_consts = [
          hw.ConstantOp(ir.IntegerType.get_signless(32), c) for c in chunks[:-1]
      ]
      chunk_consts.append(
          hw.ConstantOp(ir.IntegerType.get_signless(last_bitwidth), chunks[-1]))

      from .signals import BitsSignal
      return BitsSignal.concat(chunk_consts).bitcast(circt_type)
    return hw.ConstantOp(circt_type, x)


# A single bit is common enough to provide an alias to save 4 key strokes.
Bit = Bits(1)


class SInt(BitVectorType):

  def __new__(cls, width: int):
    return super(SInt, cls).__new__(
        cls,
        ir.IntegerType.get_signed(width),
    )

  def _get_value_class(self):
    from .signals import SIntSignal
    return SIntSignal

  def _from_obj(self, x: int, alias: typing.Optional[TypeAlias] = None):
    from .dialects import hwarith
    self._from_obj_check(x)
    circt_type = self if alias is None else alias
    return hwarith.ConstantOp(circt_type, x)


class UInt(BitVectorType):

  def __new__(cls, width: int):
    return super(UInt, cls).__new__(
        cls,
        ir.IntegerType.get_unsigned(width),
    )

  def _get_value_class(self):
    from .signals import UIntSignal
    return UIntSignal

  def _from_obj(self, x: int, alias: typing.Optional[TypeAlias] = None):
    from .dialects import hwarith
    self._from_obj_check(x)
    if x < 0:
      raise ValueError(f"UInt can only store positive numbers, not {x}")
    circt_type = self if alias is None else alias
    return hwarith.ConstantOp(circt_type, x)


class ClockType(Type):
  """A special single bit to represent a clock. Can't do any special operations
  on it, except enter it as a implicit clock block."""

  def __new__(cls):
    return super(ClockType, cls).__new__(cls, seq.ClockType.get())

  @property
  def is_hw_type(self) -> bool:
    return False

  def _get_value_class(self):
    from .signals import ClockSignal
    return ClockSignal

  def __repr__(self):
    return "Clk"


class Any(Type):

  def __new__(cls):
    return super(Any, cls).__new__(cls, esi.AnyType.get())

  @property
  def is_hw_type(self) -> bool:
    return False

  def _from_obj_or_sig(self,
                       obj,
                       alias: typing.Optional["TypeAlias"] = None) -> "Signal":
    """Any signal can be any type. Skip the type check."""

    from .signals import Signal
    if isinstance(obj, Signal):
      return obj
    return self._from_obj(obj, alias)


class Channel(Type):
  """An ESI channel type."""

  SignalingNames = {
      ChannelSignaling.ValidReady: "ValidReady",
      ChannelSignaling.FIFO: "FIFO"
  }

  def __new__(cls,
              inner_type: Type,
              signaling: int = ChannelSignaling.ValidReady,
              data_delay: int = 0):
    return super(Channel, cls).__new__(
        cls, esi.ChannelType.get(inner_type._type, signaling, data_delay))

  @property
  def inner_type(self):
    return _FromCirctType(self._type.inner)

  @property
  def is_hw_type(self) -> bool:
    return False

  @property
  def signaling(self):
    return self._type.signaling

  @property
  def data_delay(self):
    return self._type.data_delay

  def _get_value_class(self):
    from .signals import ChannelSignal
    return ChannelSignal

  def __repr__(self):
    signaling = Channel.SignalingNames[self.signaling]
    if self.data_delay == 0:
      return f"Channel<{self.inner_type}, {signaling}>"
    return f"Channel<{self.inner_type}, {signaling}({self.data_delay})>"

  @property
  def inner(self):
    return self.inner_type

  def wrap(self, value,
           valid_or_empty) -> typing.Tuple["ChannelSignal", "BitsSignal"]:
    """Wrap a data signal and valid signal into a data channel signal and a
    ready signal."""

    # Instead of implementing __call__(), we require users to call this method
    # instead. In addition to being clearer, the type signature isn't the same
    # -- this returns a tuple of Signals (data, ready) -- rather than a single
    # one.

    from .dialects import esi
    from .signals import Signal
    signaling = self.signaling
    if signaling == ChannelSignaling.ValidReady:
      if not isinstance(value, Signal):
        value = self.inner_type(value)
      elif value.type != self.inner_type:
        raise TypeError(
            f"Expected signal of type {self.inner_type}, got {value.type}")
      valid = Bits(1)(valid_or_empty)
      wrap_op = esi.WrapValidReadyOp(self._type, types.i1, value.value,
                                     valid.value)
      return wrap_op[0], wrap_op[1]
    elif signaling == ChannelSignaling.FIFO:
      value = self.inner_type(value)
      empty = Bits(1)(valid_or_empty)
      wrap_op = esi.WrapFIFOOp(self._type, types.i1, value.value, empty.value)
      return wrap_op[0], wrap_op[1]
    else:
      raise TypeError("Unknown signaling standard")

  def _join(self, a: "ChannelSignal", b: "ChannelSignal") -> "ChannelSignal":
    """Join two channels into a single channel. The resulting type is a struct
    with two fields, 'a' and 'b' wherein 'a' is the data from channel a and 'b'
    is the data from channel b."""

    from .constructs import Wire
    both_ready = Wire(Bits(1))
    a_data, a_valid = a.unwrap(both_ready)
    b_data, b_valid = b.unwrap(both_ready)
    both_valid = a_valid & b_valid
    result_data = self.inner_type({"a": a_data, "b": b_data})
    result_chan, result_ready = self.wrap(result_data, both_valid)
    both_ready.assign(result_ready & both_valid)
    return result_chan

  @staticmethod
  def join(a: "ChannelSignal", b: "ChannelSignal") -> "ChannelSignal":
    """Join two channels into a single channel. The resulting type is a struct
    with two fields, 'a' and 'b' wherein 'a' is the data from channel a and 'b'
    is the data from channel b."""

    from .types import Channel, StructType
    return Channel(
        StructType([("a", a.type.inner_type),
                    ("b", b.type.inner_type)]))._join(a, b)

  def merge(self, a: "ChannelSignal", b: "ChannelSignal") -> "ChannelSignal":
    """Merge two channels into a single channel, selecting a message from either
    one. May implement any sort of fairness policy. Both channels must be of the
    same type. Returns both the merged channel."""

    from .constructs import Mux, Wire
    a_ready = Wire(Bits(1))
    b_ready = Wire(Bits(1))
    a_data, a_valid = a.unwrap(a_ready)
    b_data, b_valid = b.unwrap(b_ready)

    sel_a = a_valid
    sel_b = ~sel_a
    out_ready = Wire(Bits(1))
    a_ready.assign(sel_a & out_ready)
    b_ready.assign(sel_b & out_ready)

    valid = (sel_a & a_valid) | (sel_b & b_valid)
    data = Mux(sel_a, b_data, a_data)
    chan, ready = self.wrap(data, valid)
    out_ready.assign(ready)
    return chan


@dataclass
class BundledChannel:
  """A named, directed channel for inclusion in a bundle."""
  name: str
  direction: ChannelDirection
  channel: Type

  def __repr__(self) -> str:
    return f"('{self.name}', {str(self.direction)}, {self.channel})"


class Bundle(Type):
  """A group of named, directed channels. Typically used in a service."""

  def __new__(cls, channels: typing.List[BundledChannel]) -> Bundle:

    def wrap_in_channel(ty: Type):
      if isinstance(ty, Channel):
        return ty
      return Channel(ty)

    type = esi.BundleType.get(
        [(bc.name, bc.direction, wrap_in_channel(bc.channel)._type)
         for bc in channels], False)
    return super(Bundle, cls).__new__(cls, type)

  def _get_value_class(self):
    from .signals import BundleSignal
    return BundleSignal

  @property
  def is_hw_type(self) -> bool:
    return False

  @property
  def channels(self) -> typing.List[BundledChannel]:
    return [
        BundledChannel(name, dir, _FromCirctType(type))
        for (name, dir, type) in self._type.channels
    ]

  def castable(self, _) -> bool:
    raise TypeError("Cannot check cast-ablity to a bundle")

  def inverted(self) -> "Bundle":
    """Return a new bundle with all the channels direction inverted."""
    return Bundle([
        BundledChannel(
            name, ChannelDirection.TO
            if dir == ChannelDirection.FROM else ChannelDirection.FROM,
            _FromCirctType(ty)) for (name, dir, ty) in self._type.channels
    ])

  def get_to_from(
      self
  ) -> typing.Tuple[typing.Optional[BundledChannel],
                    typing.Optional[BundledChannel]]:
    """In a bidirectional, one or two channel bundle, it is often desirable to
    easily have access to the from and to channels."""

    bundle_channels = self.channels
    if len(bundle_channels) > 2:
      raise ValueError("Bundle must have at most two channels")

    # Return vars.
    to_channel_bc: typing.Optional[BundledChannel] = None
    from_channel_bc: typing.Optional[BundledChannel] = None

    # Look at the first channel.
    if bundle_channels[0].direction == ChannelDirection.TO:
      to_channel_bc = bundle_channels[0]
    else:
      from_channel_bc = bundle_channels[0]

    if len(bundle_channels) == 2:
      # Look at the second channel.
      if bundle_channels[1].direction == ChannelDirection.TO:
        to_channel_bc = bundle_channels[1]
      else:
        from_channel_bc = bundle_channels[1]

    # Check and return.
    if len(bundle_channels) == 2 and (to_channel_bc is None or
                                      from_channel_bc is None):
      raise ValueError("Bundle must have one channel in each direction.")
    return to_channel_bc, from_channel_bc

  def create_uturn(self) -> typing.Tuple["BundleSignal", "BundleSignal"]:
    """Creates two bundle signals which talk to each other. The types of them
    are the inverse of each other, the first one matching this type. E.g.
    anything which is sent on the TO channel 'foo' on the first channel will be
    transmitted to the FROM channel 'foo' on the second bundle."""

    b_type = self.inverted()
    from .constructs import Wire
    to_channel_wires = {
        bc.name: Wire(bc.channel)
        for bc in self.channels
        if bc.direction == ChannelDirection.TO
    }
    a_bundle, a_froms = self.pack(**to_channel_wires)
    b_bundle, b_froms = b_type.pack(**a_froms.from_channels)
    for name, wire in to_channel_wires.items():
      wire.assign(b_froms[name])
    return a_bundle, b_bundle

  # Easy accessor for channel types by name.
  def __getattr__(self, attrname: str):
    for channel in self.channels:
      if channel.name == attrname:
        return channel.channel
    return super().__getattribute__(attrname)

  def __repr__(self):
    return f"Bundle<{self.channels}>"

  class PackSignalResults:
    """Access the FROM channels of a packed bundle in a convenient way."""

    def __init__(self, results: typing.List[ChannelSignal],
                 bundle_type: Bundle):
      self.results = results
      self.bundle_type = bundle_type

      self.from_channels = {
          name: result for (name, result) in zip([
              c.name
              for c in self.bundle_type.channels
              if c.direction == ChannelDirection.FROM
          ], results)
      }

      from_channels_idx = [
          c.name
          for c in self.bundle_type.channels
          if c.direction == ChannelDirection.FROM
      ]
      self._from_channels_idx = {
          name: idx for idx, name in enumerate(from_channels_idx)
      }

    def __contains__(self, name: str) -> bool:
      return name in self._from_channels_idx

    @singledispatchmethod
    def __getitem__(self, name: str) -> ChannelSignal:
      return self.results[self._from_channels_idx[name]]

    @__getitem__.register(int)
    def __getitem_int(self, idx: int) -> ChannelSignal:
      return self.results[idx]

    def __getattr__(self, attrname: str):
      if attrname in self._from_channels_idx:
        return self.results[self._from_channels_idx[attrname]]
      return super().__getattribute__(attrname)

    def __iter__(self):
      return iter(self.from_channels.items())

    def __len__(self):
      return len(self.from_channels)

  def pack(
      self, **kwargs: typing.Dict[str, "ChannelSignal"]
  ) -> ("BundleSignal", typing.Dict[str, "ChannelSignal"]):
    """Pack a dict of TO channels into a bundle. Returns the bundle AND a dict
    of all the FROM channels."""

    from .signals import BundleSignal, _FromCirctValue
    to_channels = {
        bc.name: (idx, bc) for idx, bc in enumerate(
            filter(lambda c: c.direction == ChannelDirection.TO, self.channels))
    }
    from_channels = [
        c for c in self.channels if c.direction == ChannelDirection.FROM
    ]

    operands = [None] * len(to_channels)
    for name, value in kwargs.items():
      if name not in to_channels:
        raise ValueError(f"Unknown channel name '{name}'")
      idx, bc = to_channels[name]
      if value.type != bc.channel:
        raise TypeError(f"Expected channel type {bc.channel}, got {value.type} "
                        f"on channel '{name}'")
      operands[idx] = value.value
      del to_channels[name]
    if len(to_channels) > 0:
      raise ValueError(f"Missing channels: {', '.join(to_channels.keys())}")

    with get_user_loc():
      pack_op = esi.PackBundleOp(self._type,
                                 [bc.channel._type for bc in from_channels],
                                 operands)

    return BundleSignal(pack_op.bundle, self), Bundle.PackSignalResults(
        [_FromCirctValue(c) for c in pack_op.fromChannels], self)


class List(Type):
  """An ESI list type represents variable length data. Just like a Python list."""

  def __new__(cls, element_type: Type):
    return super(List, cls).__new__(cls, esi.ListType.get(element_type._type))

  @property
  def element_type(self):
    return _FromCirctType(self._type.element_type)

  @property
  def is_hw_type(self) -> bool:
    return False

  def _get_value_class(self):
    from .signals import ListSignal
    return ListSignal

  def __repr__(self):
    return f"List<{self.element_type}>"

  @property
  def inner(self):
    return self.inner_type


class Window(Type):
  """An ESI data window type.

  Construct with a name (string), an 'into' type (typically a StructType), and
  a list of Window.Frame objects. Each frame spec contains a name and a list of
  field specifications. A field specification is either a field name string or a
  tuple (field_name, num_items). 'num_items' is only allowed for fields with
  array or list types in the underlying struct and indicates how many items are
  accessible in the frame.

  Example:
    Window("pkt", my_struct_type,
           [
             Window.Frame("header", ["hdr", ("payload_array", 4)]),
             Window.Frame("tail", ["tail"])
           ])
  """

  @dataclass
  class Frame:
    """Represents a frame specification within a Window type."""
    name: str
    members: typing.List[typing.Union[str, typing.Tuple[str,
                                                        typing.Optional[int]]]]

    def __post_init__(self):
      """Validate frame specification after initialization."""
      if not isinstance(self.name, str):
        raise TypeError(
            f"Frame name must be a string, got {type(self.name).__name__}")
      if not isinstance(self.members, (list, tuple)):
        raise TypeError(
            f"Frame members must be a list, got {type(self.members).__name__}")

    def __repr__(self) -> str:
      return f"Frame('{self.name}', {self.members})"

  def __new__(cls, name: str, into: StructType | type["Struct"],
              frames: typing.List["Window.Frame"]):
    # Convert Window.Frame specs into underlying CIRCT types.
    # Get struct fields for validation
    struct_fields = {
        field_name: field_type for field_name, field_type in into.fields
    }

    frame_types = []
    for frame_spec in frames:
      if not isinstance(frame_spec, Window.Frame):
        raise TypeError(
            f"Frame spec must be a Window.Frame object, got {type(frame_spec).__name__}"
        )

      field_types = []
      for m in frame_spec.members:
        if isinstance(m, tuple):
          field_name, num_items = m
          # Validate that num_items is only used on array/list fields
          if field_name not in struct_fields:
            raise ValueError(f"Field '{field_name}' not found in struct type")
          field_type = struct_fields[field_name]
          if num_items is not None:
            # Check if field is an array or list type
            if not isinstance(field_type, (Array, List)):
              raise ValueError(
                  f"num_items can only be specified for array or list fields. "
                  f"Field '{field_name}' has type {field_type}")
          # Convert field name to StringAttr and keep num_items as optional int
          field_name_attr = ir.StringAttr.get(field_name)
          field_types.append(esi.WindowFieldType.get(field_name_attr,
                                                     num_items))
        else:
          # Convert field name to StringAttr
          field_name_attr = ir.StringAttr.get(m)
          field_types.append(esi.WindowFieldType.get(field_name_attr))
      # Convert frame name to StringAttr
      frame_name_attr = ir.StringAttr.get(frame_spec.name)
      frame_types.append(esi.WindowFrameType.get(frame_name_attr, field_types))
    # Convert window name to StringAttr
    window_name_attr = ir.StringAttr.get(name)
    window_ty = esi.WindowType.get(window_name_attr, into._type, frame_types)
    return super(Window, cls).__new__(cls, window_ty)

  def _get_value_class(self):
    from .signals import WindowSignal
    return WindowSignal

  @property
  def is_hw_type(self) -> bool:
    return False

  @property
  def name(self) -> str:
    return self._type.name

  @property
  def into(self) -> Type:
    return _FromCirctType(self._type.into)

  @property
  def frames(self) -> typing.List[Frame]:
    # Return a list of Window.Frame objects with python-friendly representation
    ret = []
    for f in self._type.frames:
      members = []
      frame = support.type_to_pytype(f)
      for m in frame.members:
        member = support.type_to_pytype(m)
        num_items = member.num_items
        members.append(
            (member.field_name.value, num_items if num_items > 0 else None))
      ret.append(Window.Frame(frame.name.value, members))
    return ret

  def __repr__(self):
    return f"Window<{self.name}, {self.into}, frames={self.frames}>"

  def wrap(self, signal):
    """Wrap a signal (struct or union) into a WindowSignal using esi.WindowWrapOp.
    
    Args:
      signal: A Signal with a type that matches the 'into' type of this Window.
    
    Returns:
      A WindowSignal wrapping the input signal.
    """
    from .signals import Signal as SignalBase, _FromCirctValue, WindowSignal

    if not isinstance(signal, SignalBase):
      raise TypeError(f"Expected a Signal, got {type(signal).__name__}")

    if signal.type != self.into:
      raise TypeError(
          f"Signal type {signal.type} does not match Window 'into' type {self.into}"
      )

    with get_user_loc():
      wrap_op = esi.WindowWrapOp(self._type, signal.value)
      return WindowSignal(wrap_op, self)

  # Windows are not directly constructible from python literals.
  def _from_obj(self, obj, alias: typing.Optional[TypeAlias] = None):
    raise TypeError("Cannot create Window values from Python objects directly")


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
