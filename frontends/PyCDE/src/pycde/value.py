#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .support import get_user_loc, _obj_to_value_infer_type

from circt.dialects import sv, esi
import circt.support as support

import mlir.ir as ir

from contextvars import ContextVar
from functools import singledispatchmethod
from typing import Optional, Union
import re
import numpy as np


def Value(value, type=None):
  from .pycde_types import Type

  if isinstance(value, PyCDEValue):
    return value

  resvalue = support.get_value(value)
  if resvalue is None:
    return _obj_to_value_infer_type(value)

  if type is None:
    type = resvalue.type
  type = Type(type)

  return type._get_value_class()(resvalue, type)


class PyCDEValue:
  """Root of the PyCDE value (signal, in RTL terms) hierarchy."""

  def __init__(self, value, type=None):
    from .pycde_types import Type

    if isinstance(value, ir.Value):
      self.value = value
    else:
      self.value = support.get_value(value)
      if self.value is None:
        self.value = _obj_to_value_infer_type(value).value

    if type is not None:
      self.type = type
    else:
      self.type = Type(self.value.type)

  _reg_name = re.compile(r"^(.*)__reg(\d+)$")

  def reg(self,
          clk=None,
          rst=None,
          name=None,
          cycles=1,
          sv_attributes=None,
          appid=None):
    """Register this value, returning the delayed value.
    `clk`, `rst`: the clock and reset signals.
    `name`: name this register explicitly.
    `cycles`: number of registers to add."""

    if clk is None:
      clk = ClockValue._get_current_clock_block()
      if clk is None:
        raise ValueError("If 'clk' not specified, must be in clock block")

    from .dialects import seq
    if name is None:
      basename = None
      if self.name is not None:
        m = PyCDEValue._reg_name.match(self.name)
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
      if sv_attributes is not None:
        reg.value.owner.attributes["sv.attributes"] = sv.SVAttributesAttr.get(
            ir.ArrayAttr.get(
                [sv.SVAttributeAttr.get(attr) for attr in sv_attributes]))
      if appid is not None:
        reg.appid = appid
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
      block_arg = ir.BlockArgument(self.value)
      mod = owner.owner
      return ir.StringAttr(
          ir.ArrayAttr(mod.attributes["argNames"])[block_arg.arg_number]).value
    if hasattr(self, "_name"):
      return self._name

  @name.setter
  def name(self, new: str):
    owner = self.value.owner
    if hasattr(owner, "attributes"):
      owner.attributes[self._namehint_attrname] = ir.StringAttr.get(new)
    else:
      self._name = new

  @property
  def appid(self) -> Optional[object]:  # Optional AppID.
    from .module import AppID
    owner = self.value.owner
    if AppID.AttributeName in owner.attributes:
      return AppID(owner.attributes[AppID.AttributeName])
    return None

  @appid.setter
  def appid(self, appid) -> None:
    if "sym_name" not in self.value.owner.attributes:
      raise ValueError("AppIDs can only be attached to ops with symbols")
    from .module import AppID
    self.value.owner.attributes[AppID.AttributeName] = appid._appid


class RegularValue(PyCDEValue):
  pass


_current_clock_context = ContextVar("current_clock_context")


class ClockValue(PyCDEValue):
  """A clock signal."""

  __slots__ = ["_old_token"]

  def __enter__(self):
    self._old_token = _current_clock_context.set(self)

  def __exit__(self, exc_type, exc_value, traceback):
    if exc_value is not None:
      return
    _current_clock_context.reset(self._old_token)

  @staticmethod
  def _get_current_clock_block():
    return _current_clock_context.get(None)


class InOutValue(PyCDEValue):
  # Maintain a caching of the read value.
  read_value = None

  @property
  def read(self):
    if self.read_value is None:
      self.read_value = Value(sv.ReadInOutOp.create(self).results[0])
    return self.read_value


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


def get_slice_bounds(size, idxOrSlice: Union[int, slice]):
  if isinstance(idxOrSlice, int):
    s = slice(idxOrSlice, idxOrSlice + 1)
  elif isinstance(idxOrSlice, slice):
    if idxOrSlice.stop and idxOrSlice.stop > size:
      raise ValueError("Slice out-of-bounds")
    s = idxOrSlice
  else:
    raise TypeError("Expected int or slice")

  idxs = s.indices(size)
  if idxs[2] != 1:
    raise ValueError("Integer / bitvector slices do not support steps")
  return idxs[0], idxs[1]


class BitVectorValue(PyCDEValue):

  @singledispatchmethod
  def __getitem__(self, idxOrSlice: Union[int, slice]) -> BitVectorValue:
    lo, hi = get_slice_bounds(len(self), idxOrSlice)
    from .pycde_types import types
    from .dialects import comb
    ret_type = types.int(hi - lo)

    with get_user_loc():
      ret = comb.ExtractOp(lo, ret_type, self.value)
      if self.name is not None:
        ret.name = f"{self.name}_{lo}upto{hi}"
      return ret

  @__getitem__.register(PyCDEValue)
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

  #  === Casting ===

  def _exec_cast(self, targetValueType, type_getter, width: int = None):

    from .dialects import hwarith
    if width is None:
      width = self.type.width

    if type(self) is targetValueType and width == self.type.width:
      return self
    return hwarith.CastOp(self.value, type_getter(width))

  def as_int(self, width: int = None):
    """
    Returns this value as a signless integer. If 'width' is provided, this value
    will be truncated to that width.
    """
    return self._exec_cast(BitVectorValue, ir.IntegerType.get_signless, width)

  def as_sint(self, width: int = None):
    """
    Returns this value as a a signed integer. If 'width' is provided, this value
    will be truncated or sign-extended to that width.
    """
    return self._exec_cast(SignedBitVectorValue, ir.IntegerType.get_signed,
                           width)

  def as_uint(self, width: int = None):
    """
    Returns this value as an unsigned integer. If 'width' is provided, this value
    will be truncated or zero-padded to that width.
    """
    return self._exec_cast(UnsignedBitVectorValue, ir.IntegerType.get_unsigned,
                           width)

  #  === Infix operators ===

  # Signless operations. These will all return signless values - a user is
  # expected to reapply signedness semantics if needed.

  # Generalized function for executing signless binary operations. Performs
  # a check to ensure that the operands have signless semantics and are of
  # identical width, and then calls the provided operator.
  def __exec_signless_binop__(self, other, op, op_name: str):
    w = max(self.type.width, other.type.width)
    ret = op(self.as_int(w), other.as_int(w))
    if self.name is not None and other.name is not None:
      ret.name = f"{self.name}_{op_name}_{other.name}"
    return ret

  def __exec_signless_binop_nocast__(self, other, op, op_symbol: str,
                                     op_name: str):
    if not isinstance(other, PyCDEValue):
      # Fall back to the default implementation in cases where we're not dealing
      # with PyCDE value comparison.
      return super().__eq__(other)

    signednessOperand = None
    if type(self) is not BitVectorValue:
      signednessOperand = "LHS"
    elif type(other) is not BitVectorValue:
      signednessOperand = "RHS"

    if signednessOperand is not None:
      raise TypeError(
          f"Operator '{op_symbol}' requires {signednessOperand} to be cast .as_int()."
      )

    w = max(self.type.width, other.type.width)
    ret = op(self.as_int(w), other.as_int(w))
    if self.name is not None and other.name is not None:
      ret.name = f"{self.name}_{op_name}_{other.name}"
    return ret

  def __eq__(self, other):
    from .dialects import comb
    return self.__exec_signless_binop_nocast__(other, comb.EqOp, "==", "eq")

  def __ne__(self, other):
    from .dialects import comb
    return self.__exec_signless_binop_nocast__(other, comb.NeOp, "!=", "neq")

  def __and__(self, other):
    from .dialects import comb
    return self.__exec_signless_binop__(other, comb.AndOp, "and")

  def __or__(self, other):
    from .dialects import comb
    return self.__exec_signless_binop__(other, comb.OrOp, "or")

  def __xor__(self, other):
    from .dialects import comb
    return self.__exec_signless_binop__(other, comb.XorOp, "xor")

  def __invert__(self):
    from .pycde_types import types
    ret = self.as_int() ^ types.int(self.type.width)(-1)
    if self.name is not None:
      ret.name = f"neg_{self.name}"
    return ret

  # Generalized function for executing sign-aware binary operations. Performs
  # a check to ensure that the operands have signedness semantics, and then calls
  # the provided operator.
  def __exec_signedness_binop__(self, other, op, op_symbol: str, op_name: str):
    signlessOperand = None
    if type(self) is BitVectorValue:
      signlessOperand = "LHS"
    elif type(other) is BitVectorValue:
      signlessOperand = "RHS"

    if signlessOperand is not None:
      raise TypeError(
          f"Operator '{op_symbol}' is not supported on signless values. {signlessOperand} operand should be cast .as_sint()/.as_uint()."
      )

    ret = op(self, other)
    if self.name is not None and other.name is not None:
      ret.name = f"{self.name}_{op_name}_{other.name}"
    return ret

  def __add__(self, other):
    from .dialects import hwarith
    return self.__exec_signedness_binop__(other, hwarith.AddOp, "+", "plus")

  def __sub__(self, other):
    from .dialects import hwarith
    return self.__exec_signedness_binop__(other, hwarith.SubOp, "-", "minus")

  def __mul__(self, other):
    from .dialects import hwarith
    return self.__exec_signedness_binop__(other, hwarith.MulOp, "*", "mul")

  def __truediv__(self, other):
    from .dialects import hwarith
    return self.__exec_signedness_binop__(other, hwarith.DivOp, "/", "div")


class WidthExtendingBitVectorValue(BitVectorValue):
  # TODO: This class will contain comparison operators (<, >, <=, >=)
  pass

  def __lt__(self, other):
    assert False, "Unimplemented"

  def __le__(self, other):
    assert False, "Unimplemented"

  def __ge__(self, other):
    assert False, "Unimplemented"


class UnsignedBitVectorValue(WidthExtendingBitVectorValue):
  pass


class SignedBitVectorValue(WidthExtendingBitVectorValue):

  def __neg__(self):
    from .dialects import comb
    from .pycde_types import types
    return self * types.int(self.type.width)(-1).as_sint()


class ListValue(PyCDEValue):

  @singledispatchmethod
  def __getitem__(self, idx: Union[int, BitVectorValue]) -> PyCDEValue:
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

  def slice(self, low_idx: Union[int, BitVectorValue],
            num_elems: int) -> PyCDEValue:
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

  """
  Add a curated set of Numpy functions through the Matrix class.
  This allows for directly manipulating the ListValues with numpy functionality.
  Power-users who use the Matrix directly have access to all numpy functions.
  In reality, it will only be a subset of the numpy array functions which are
  safe to be used in the PyCDE context. Curating access at the level of
  ListValues seems like a safe starting point.
  """

  def transpose(self, *args, **kwargs):
    from .ndarray import NDArray
    return NDArray(from_value=self).transpose(*args, **kwargs).to_circt()

  def reshape(self, *args, **kwargs):
    from .ndarray import NDArray
    return NDArray(from_value=self).reshape(*args, **kwargs).to_circt()

  def flatten(self, *args, **kwargs):
    from .ndarray import NDArray
    return NDArray(from_value=self).flatten(*args, **kwargs).to_circt()

  def moveaxis(self, *args, **kwargs):
    from .ndarray import NDArray
    return NDArray(from_value=self).moveaxis(*args, **kwargs).to_circt()

  def rollaxis(self, *args, **kwargs):
    from .ndarray import NDArray
    return NDArray(from_value=self).rollaxis(*args, **kwargs).to_circt()

  def swapaxes(self, *args, **kwargs):
    from .ndarray import NDArray
    return NDArray(from_value=self).swapaxes(*args, **kwargs).to_circt()

  def concatenate(self, arrays, axis=0):
    from .ndarray import NDArray
    return NDArray(from_value=np.concatenate(
        NDArray.to_ndarrays([self] + list(arrays)), axis=axis)).to_circt()

  def roll(self, shift, axis=None):
    from .ndarray import NDArray
    return np.roll(NDArray(from_value=self), shift=shift, axis=axis).to_circt()


class StructValue(PyCDEValue):

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


class ChannelValue(PyCDEValue):

  def reg(self, clk, rst=None, name=None):
    raise TypeError("Cannot register a channel")

  def unwrap(self, ready):
    from .pycde_types import types
    from .support import _obj_to_value
    ready = _obj_to_value(ready, types.i1)
    unwrap_op = esi.UnwrapValidReadyOp(self.type.inner_type, types.i1,
                                       self.value, ready.value)
    return Value(unwrap_op.rawOutput), Value(unwrap_op.valid)


def wrap_opviews_with_values(dialect, module_name, excluded=[]):
  """Wraps all of a dialect's OpView classes to have their create method return
     a PyCDE Value instead of an OpView. The wrapped classes are inserted into
     the provided module."""
  import sys
  module = sys.modules[module_name]

  for attr in dir(dialect):
    cls = getattr(dialect, attr)

    if attr not in excluded and isinstance(cls, type) and issubclass(
        cls, ir.OpView):

      def specialize_create(cls):

        def create(*args, **kwargs):
          # If any of the arguments are Value objects, we need to convert them.
          args = [v.value if isinstance(v, PyCDEValue) else v for v in args]
          kwargs = {
              k: v.value if isinstance(v, PyCDEValue) else v
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
