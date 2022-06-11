#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .support import get_user_loc, _obj_to_value_infer_type

from circt.dialects import esi
import circt.support as support

import mlir.ir as ir

from functools import singledispatchmethod
from typing import Union
import re


class Value:

  # Dummy __init__ as everything is done in __new__.
  def __init__(self, value, type=None):
    pass

  def __new__(cls, value, type=None):
    from .pycde_types import Type

    if value is None or isinstance(value, Value):
      return value
    resvalue = support.get_value(value)

    if resvalue is None:
      return _obj_to_value_infer_type(value)

    if type is None:
      type = resvalue.type
    type = Type(type)
    v = super().__new__(type._get_value_class())
    v.value = resvalue
    v.type = type
    return v

  _reg_name = re.compile(r"^(.*)__reg(\d+)$")

  def reg(self, clk, rst=None, name=None, cycles=1):
    """Register this value, returning the delayed value.
    `clk`, `rst`: the clock and reset signals.
    `name`: name this register explicitly.
    `cycles`: number of registers to add."""
    from .dialects import seq
    if name is None:
      basename = None
      if self.name is not None:
        m = Value._reg_name.match(self.name)
        if m:
          basename = m.group(1)
          reg_num = m.group(2)
          if reg_num.isdigit():
            starting_reg = int(reg_num) + 1
          else:
            basename = self.name
        else:
          basename = self.name
          starting_reg = 1
    with get_user_loc():
      reg = self.value
      for i in range(cycles):
        give_name = name
        if give_name is None and basename is not None:
          give_name = f"{basename}__reg{i+starting_reg}"
        reg = seq.CompRegOp(self.value.type,
                            input=reg,
                            clk=clk,
                            reset=rst,
                            name=give_name,
                            sym_name=give_name)
      return reg

  @property
  def _namehint_attrname(self):
    if self.value.owner.name == "seq.compreg":
      return "name"
    return "sv.namehint"

  @property
  def name(self):
    owner = self.value.owner
    if hasattr(owner,
               "attributes") and self._namehint_attrname in owner.attributes:
      return ir.StringAttr(owner.attributes[self._namehint_attrname]).value
    from circt.dialects import msft
    if isinstance(owner, ir.Block) and isinstance(owner.owner,
                                                  msft.MSFTModuleOp):
      mod = owner.owner
      return ir.StringAttr(
          ir.ArrayAttr(mod.attributes["argNames"])[self.value.arg_number]).value
    if hasattr(self, "_name"):
      return self._name

  @name.setter
  def name(self, new: str):
    owner = self.value.owner
    if hasattr(owner, "attributes"):
      owner.attributes[self._namehint_attrname] = ir.StringAttr.get(new)
    else:
      self._name = new


class RegularValue(Value):
  pass


def _validate_idx(size: int, idx: Union[int, BitVectorValue]):
  """Validate that `idx` is a valid index into a bitvector or array."""
  if isinstance(idx, int):
    if idx >= size:
      raise ValueError("Subscript out-of-bounds")
  else:
    idx = support.get_value(idx)
    if idx is None or not isinstance(support.type_to_pytype(idx.type),
                                     ir.IntegerType):
      raise TypeError("Subscript on array must be either int or MLIR int"
                      f" Value, not {type(idx)}.")


class BitVectorValue(Value):

  @singledispatchmethod
  def __getitem__(self, idxOrSlice: Union[int, slice]) -> BitVectorValue:
    if isinstance(idxOrSlice, int):
      s = slice(idxOrSlice, idxOrSlice + 1)
    elif isinstance(idxOrSlice, slice):
      s = idxOrSlice
    else:
      raise TypeError("Expected int or slice")
    idxs = s.indices(len(self))
    if idxs[2] != 1:
      raise ValueError("Integer / bitvector slices do not support steps")

    from .pycde_types import types
    from .dialects import comb
    ret_type = types.int(idxs[1] - idxs[0])

    with get_user_loc():
      ret = comb.ExtractOp(idxs[0], ret_type, self.value)
      if self.name is not None:
        ret.name = f"{self.name}_{idxs[0]}upto{idxs[1]}"
      return ret

  @__getitem__.register(Value)
  def __get_item__value(self, idx: BitVectorValue) -> BitVectorValue:
    """Get the single bit at `idx`."""
    return self.slice(idx, 1)

  def slice(self, low_bit: BitVectorValue, num_bits: int):
    """Get a constant-width slice starting at `low_bit` and ending at `low_bit +
    num_bits`."""
    _validate_idx(self.type.width, low_bit)

    from .dialects import comb
    # comb.extract only supports constant lowBits. Shift the bits right, then
    # extract the correct number from the 0th bit.
    with get_user_loc():
      # comb.shru's rhs and lhs must be the same width.
      low_bit = low_bit.pad_or_truncate(self.type.width)
      shifted = comb.ShrUOp(self.value, low_bit)
      ret = comb.ExtractOp(0, ir.IntegerType.get_signless(num_bits), shifted)
      return ret

  def pad_or_truncate(self, num_bits: int):
    """Make value exactly `num_bits` width by either adding zeros to or lopping
    off the MSB."""
    pad_width = num_bits - self.type.width

    from .dialects import comb, hw
    if pad_width < 0:
      return comb.ExtractOp(0, ir.IntegerType.get_signless(num_bits),
                            self.value)
    if pad_width == 0:
      return self
    pad = hw.ConstantOp(ir.IntegerType.get_signless(pad_width), 0)
    return comb.ConcatOp(pad.value, self.value)

  def __len__(self):
    return self.type.width


class ListValue(Value):

  @singledispatchmethod
  def __getitem__(self, idx: Union[int, BitVectorValue]) -> Value:
    _validate_idx(self.type.size, idx)
    from .dialects import hw
    with get_user_loc():
      v = hw.ArrayGetOp(self.value, idx)
      if self.name and isinstance(idx, int):
        v.name = self.name + f"__{idx}"
      return v

  @__getitem__.register(slice)
  def __get_item__slice(self, s: slice):
    idxs = s.indices(len(self))
    if idxs[2] != 1:
      raise ValueError("Array slices do not support steps")

    from .pycde_types import types
    from .dialects import hw
    ret_type = types.array(self.type.element_type, idxs[1] - idxs[0])

    with get_user_loc():
      ret = hw.ArraySliceOp(self.value, idxs[0], ret_type)
      if self.name is not None:
        ret.name = f"{self.name}_{idxs[0]}upto{idxs[1]}"
      return ret

  def slice(self, low_idx: Union[int, BitVectorValue], num_elems: int) -> Value:
    """Get an array slice starting at `low_idx` and ending at `low_idx +
    num_elems`."""
    _validate_idx(self.type.size, low_idx)
    if num_elems > self.type.size:
      raise ValueError(
          f"num_bits ({num_elems}) must be <= value width ({len(self)})")
    if isinstance(low_idx, BitVectorValue):
      low_idx = low_idx.pad_or_truncate(self.type.size.bit_length())

    from .dialects import hw
    with get_user_loc():
      v = hw.ArraySliceOp(self.value, low_idx,
                          hw.ArrayType.get(self.type.element_type, num_elems))
      if self.name and isinstance(low_idx, int):
        v.name = self.name + f"__{low_idx}upto{low_idx+num_elems}"
      return v

  def __len__(self):
    return self.type.strip.size


class StructValue(Value):

  def __getitem__(self, sub):
    if sub not in [name for name, _ in self.type.strip.fields]:
      raise ValueError(f"Struct field '{sub}' not found in {self.type}")
    from .dialects import hw
    with get_user_loc():
      return hw.StructExtractOp(self.value, sub)

  def __getattr__(self, attr):
    ty = self.type.strip
    if attr in [name for name, _ in ty.fields]:
      from .dialects import hw
      with get_user_loc():
        v = hw.StructExtractOp(self.value, attr)
        if self.name:
          v.name = f"{self.name}__{attr}"
        return v
    raise AttributeError(f"'Value' object has no attribute '{attr}'")


class ChannelValue(Value):

  def reg(self, clk, rst=None, name=None):
    raise TypeError("Cannot register a channel")

  def unwrap(self, ready):
    from .pycde_types import types
    from .support import _obj_to_value
    ready = _obj_to_value(ready, types.i1)
    unwrap_op = esi.UnwrapValidReady(self.type.inner_type, types.i1, self.value,
                                     ready.value)
    return Value(unwrap_op.rawOutput), Value(unwrap_op.valid)


def wrap_opviews_with_values(dialect, module_name):
  """Wraps all of a dialect's OpView classes to have their create method return
     a PyCDE Value instead of an OpView. The wrapped classes are inserted into
     the provided module."""
  import sys
  module = sys.modules[module_name]

  for attr in dir(dialect):
    cls = getattr(dialect, attr)

    if isinstance(cls, type) and issubclass(cls, ir.OpView):

      def specialize_create(cls):

        def create(*args, **kwargs):
          # If any of the arguments are Value objects, we need to convert them.
          args = [v.value if isinstance(v, Value) else v for v in args]
          kwargs = {
              k: v.value if isinstance(v, Value) else v
              for k, v in kwargs.items()
          }
          # Create the OpView.
          created = cls.create(*args, **kwargs)
          if isinstance(created, support.NamedValueOpView):
            created = created.opview

          # Return the wrapped values, if any.
          converted_results = tuple(Value(res) for res in created.results)
          return converted_results[0] if len(
              converted_results) == 1 else converted_results

        return create

      wrapped_class = specialize_create(cls)
      setattr(module, attr, wrapped_class)
    else:
      setattr(module, attr, cls)
