#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .base import ir
from .core import Value, Type
from .index import index
from .rtg import rtg
from .strings import String
from .support import _create

from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
  from .immediates import Immediate


class Integer(Value):
  """
  This represents an integer with the same number of bits as a 'size_t' in C.
  It is used to provide parameter values to meta-level constructs such as the
  multiple of an element in a Bag. These integers will be fully constant folded
  away during randomization.
  """

  def __init__(self, value: Union[ir.Value, int]) -> Integer:
    """
    Use this constructor to create an Integer from a builtin Python int.
    """

    self._value = value

  def random(lower_bound: Union[int, Integer],
             upper_bound: Union[int, Integer]) -> Integer:
    """
    Get a random number in the given range (lower and upper inclusive).
    """

    if isinstance(lower_bound, int):
      lower_bound = Integer(lower_bound)
    if isinstance(upper_bound, int):
      upper_bound = Integer(upper_bound)

    op = _create(rtg.RandomNumberInRangeOp, lower_bound, upper_bound)
    return Integer(op.result)

  def __add__(self, other: Integer) -> Integer:
    return Integer(_create(index.AddOp, self, other).result)

  def __sub__(self, other: Integer) -> Integer:
    return Integer(_create(index.SubOp, self, other).result)

  def __mul__(self, other: Integer) -> Integer:
    return Integer(_create(index.MulOp, self, other).result)

  def __floordiv__(self, other: Integer) -> Integer:
    return Integer(_create(index.DivUOp, self, other).result)

  def __truediv__(self, other: Integer) -> Integer:
    return Integer(_create(index.CeilDivUOp, self, other).result)

  def __mod__(self, other: Integer) -> Integer:
    return Integer(_create(index.RemUOp, self, other).result)

  def __lshift__(self, other: Integer) -> Integer:
    return Integer(_create(index.ShlOp, self, other).result)

  def __rshift__(self, other: Integer) -> Integer:
    return Integer(_create(index.ShrUOp, self, other).result)

  def __and__(self, other: Integer) -> Integer:
    return Integer(_create(index.AndOp, self, other).result)

  def __or__(self, other: Integer) -> Integer:
    return Integer(_create(index.OrOp, self, other).result)

  def __xor__(self, other: Integer) -> Integer:
    return Integer(_create(index.XOrOp, self, other).result)

  def __eq__(self, other: Integer) -> Immediate:
    from .immediates import Immediate
    return Immediate(1, _create(index.CmpOp, "eq", self, other).result)

  def __ne__(self, other: Integer) -> Immediate:
    from .immediates import Immediate
    return Immediate(1, _create(index.CmpOp, "ne", self, other).result)

  def __lt__(self, other: Integer) -> Immediate:
    from .immediates import Immediate
    return Immediate(1, _create(index.CmpOp, "ult", self, other).result)

  def __le__(self, other: Integer) -> Immediate:
    from .immediates import Immediate
    return Immediate(1, _create(index.CmpOp, "ule", self, other).result)

  def __gt__(self, other: Integer) -> Immediate:
    from .immediates import Immediate
    return Immediate(1, _create(index.CmpOp, "ugt", self, other).result)

  def __ge__(self, other: Integer) -> Immediate:
    from .immediates import Immediate
    return Immediate(1, _create(index.CmpOp, "uge", self, other).result)

  def max(self, other: Integer) -> Integer:
    return Integer(_create(index.MaxUOp, self, other).result)

  def min(self, other: Integer) -> Integer:
    return Integer(_create(index.MinUOp, self, other).result)

  def to_string(self) -> String:
    """
    Format this integer as a string in unsigned decimal.
    """

    return String(_create(rtg.IntFormatOp, self).result)

  def get_type(self) -> Type:
    return IntegerType()

  def _get_ssa_value(self) -> ir.Value:
    if isinstance(self._value, int):
      # The MLIR index type (more precisely the attribute) does not support integers with more than 64 bits.
      if self._value >= 2**64:
        raise ValueError("Integer value out of range")

      # The Python Bindings use an 'int64_t' and the bounds of this type are
      # checked and enforced. However, that integer ends up in a signless APInt
      # anyway, so we can compute the signed integer that matches the bitvector
      # of the unsigned integer and use that.
      if self._value >= 2**63:
        return _create(index.ConstantOp, self._value - 2**64).result
      else:
        return _create(index.ConstantOp, self._value).result

    return self._value


class IntegerType(Type):
  """
  Represents the type of integer values.
  """

  def __eq__(self, other) -> bool:
    return isinstance(other, IntegerType)

  def _codegen(self) -> ir.Type:
    return ir.IndexType.get()

  def _wrap(self, value: ir.Value) -> Integer:
    return Integer(value)
