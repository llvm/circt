#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .rtg import rtg
from .core import Value, Type
from .base import ir
from .integers import Integer
from .strings import String
from .arith import arith

from typing import Union


class Immediate(Value):

  def __init__(self, width: int, value: Union[ir.Value, int,
                                              Integer]) -> Immediate:
    if width < 0:
      raise ValueError(f"width must be non-negative, got {width}")

    if isinstance(value, int):
      # Note: it's valid to pass in a negative value here, but also a positive
      # value that is small enough for a unsigned representation
      max = (1 << width) - 1
      min = -(1 << (width - 1))

      if value < min or value > max:
        raise ValueError(
            f"Value {value} does not fit in {width}-bit representation "
            f"(valid range: [{min}, {max}])")

      # Convert to signed representation if the high bit is set because MLIR
      # built-in integer attributes are constructed with signed integers.
      if value >= (1 << (width - 1)):
        value = value - (1 << width)

    self._width = width
    self._value = value

  @staticmethod
  def random(width: int) -> Immediate:
    """
    An immediate of the provided width of a random value from 0 to the maximum
    unsigned number the immediate can hold (all bits set).
    """

    # Note that the upper limit is exclusive
    return Immediate(width, Integer.random(0, 2**width - 1))

  def concat(*args: Immediate) -> Immediate:
    """
    Concatenates this immediate with the provided immediates. The operands are
    concatenated in order, with this immediate becoming the most significant
    bits of the result.
    """

    if len(args) == 0:
      raise ValueError("At least one immediate required")

    return rtg.ConcatImmediateOp(list(args))

  def replicate(self, count: int) -> Immediate:
    """
    Replicates this immediate the provided number of times. The result is an
    immediate of a width equal to the width of this immediate multiplied by
    `count` and containing this immediate concatenated with itself `count`
    times.
    """

    if count < 0:
      raise ValueError("replicate count must be non-negative")

    return Immediate.concat(*([self] * count))

  def __getitem__(self, slice_range) -> Immediate:
    """
    Extracts bits from the immediate using Python slice notation.
    The least significant bit has index 0.
    """

    if isinstance(slice_range, slice):
      start = slice_range.start if slice_range.start is not None else 0
      stop = slice_range.stop if slice_range.stop is not None else self._width
      if slice_range.step is not None and slice_range.step != 1:
        raise ValueError("Step value other than 1 is not supported")
      if start < 0 or stop > self._width or start >= stop:
        raise ValueError(
            f"Invalid slice range [{start}:{stop}] for width {self._width}")
      return rtg.SliceImmediateOp(ImmediateType(stop - start), self, start)

    if isinstance(slice_range, int):
      if slice_range < 0 or slice_range >= self._width:
        raise ValueError(
            f"Index {slice_range} out of range for width {self._width}")
      return rtg.SliceImmediateOp(ImmediateType(1), self, slice_range)

    raise TypeError("Slice must be an integer or slice object")

  @staticmethod
  def umax(width: int) -> Immediate:
    """
    An immediate of the provided width with the maximum unsigned value it can
    hold.
    """

    return Immediate(width, 2**width - 1)

  @staticmethod
  def smax(width: int) -> Immediate:
    """
    An immediate of the provided width with the maximum signed value it can
    hold.
    """

    return Immediate(width, 2**(width - 1) - 1)

  @staticmethod
  def smin(width: int) -> Immediate:
    """
    An immediate of the provided width with the minimum signed value it can
    hold.
    """

    return Immediate(width, 1 << (width - 1))

  def to_string(self) -> String:
    """
    Formats this immediate as a string.
    """

    return rtg.ImmediateFormatOp(self)

  def __add__(self, other: Immediate) -> Immediate:
    return arith.AddIOp(self._get_ssa_value(), other._get_ssa_value())

  def __sub__(self, other: Immediate) -> Immediate:
    return arith.SubIOp(self._get_ssa_value(), other._get_ssa_value())

  def __mul__(self, other: Immediate) -> Immediate:
    return arith.MulIOp(self._get_ssa_value(), other._get_ssa_value())

  def __lshift__(self, other: Immediate) -> Immediate:
    return arith.ShLIOp(self._get_ssa_value(), other._get_ssa_value())

  def __rshift__(self, other: Immediate) -> Immediate:
    return arith.ShRUIOp(self._get_ssa_value(), other._get_ssa_value())

  def __and__(self, other: Immediate) -> Immediate:
    return arith.AndIOp(self._get_ssa_value(), other._get_ssa_value())

  def __or__(self, other: Immediate) -> Immediate:
    return arith.OrIOp(self._get_ssa_value(), other._get_ssa_value())

  def __xor__(self, other: Immediate) -> Immediate:
    return arith.XOrIOp(self._get_ssa_value(), other._get_ssa_value())

  def eq(self, other: Immediate) -> Value:
    """
    Equality comparison.
    """
    return arith.CmpIOp(arith.CmpIPredicate.eq, self._get_ssa_value(), other._get_ssa_value())

  def ne(self, other: Immediate) -> Value:
    """
    Inequality comparison.
    """
    return arith.CmpIOp(arith.CmpIPredicate.ne, self._get_ssa_value(), other._get_ssa_value())

  def ult(self, other: Immediate) -> Value:
    """
    Unsigned less than comparison.
    """
    return arith.CmpIOp(arith.CmpIPredicate.ult, self._get_ssa_value(), other._get_ssa_value())

  def ule(self, other: Immediate) -> Value:
    """
    Unsigned less than or equal comparison.
    """
    return arith.CmpIOp(arith.CmpIPredicate.ule, self._get_ssa_value(), other._get_ssa_value())

  def ugt(self, other: Immediate) -> Value:
    """
    Unsigned greater than comparison.
    """
    return arith.CmpIOp(arith.CmpIPredicate.ugt, self._get_ssa_value(), other._get_ssa_value())

  def uge(self, other: Immediate) -> Value:
    """
    Unsigned greater than or equal comparison.
    """
    return arith.CmpIOp(arith.CmpIPredicate.uge, self._get_ssa_value(), other._get_ssa_value())

  def slt(self, other: Immediate) -> Value:
    """
    Signed less than comparison.
    """
    return arith.CmpIOp(arith.CmpIPredicate.slt, self._get_ssa_value(), other._get_ssa_value())

  def sle(self, other: Immediate) -> Value:
    """
    Signed less than or equal comparison.
    """
    return arith.CmpIOp(arith.CmpIPredicate.sle, self._get_ssa_value(), other._get_ssa_value())

  def sgt(self, other: Immediate) -> Value:
    """
    Signed greater than comparison.
    """
    return arith.CmpIOp(arith.CmpIPredicate.sgt, self._get_ssa_value(), other._get_ssa_value())

  def sge(self, other: Immediate) -> Value:
    """
    Signed greater than or equal comparison.
    """
    return arith.CmpIOp(arith.CmpIPredicate.sge, self._get_ssa_value(), other._get_ssa_value())

  # def umax(self, other: Immediate) -> Immediate:
  #   """
  #   Unsigned maximum of this immediate and another.
  #   """
  #   return arith.MaxUIOp(self._get_ssa_value(), other._get_ssa_value())

  # def umin(self, other: Immediate) -> Immediate:
  #   """
  #   Unsigned minimum of this immediate and another.
  #   """
  #   return arith.MinUIOp(self._get_ssa_value(), other._get_ssa_value())

  # def smax(self, other: Immediate) -> Immediate:
  #   """
  #   Signed maximum of this immediate and another.
  #   """
  #   return arith.MaxSIOp(self._get_ssa_value(), other._get_ssa_value())

  # def smin(self, other: Immediate) -> Immediate:
  #   """
  #   Signed minimum of this immediate and another.
  #   """
  #   return arith.MinSIOp(self._get_ssa_value(), other._get_ssa_value())

  def zext(self, target_width: int) -> Immediate:
    """
    Zero extension (unsigned extension) to a wider bit width.
    The top-most bits are filled with zeros.
    """
    if target_width < self._width:
      raise ValueError(
          f"Zero extension target width ({target_width}) must be >= "
          f"current width ({self._width})")

    if target_width == self._width:
      return self

    return arith.ExtUIOp(ir.IntegerType.get_signless(target_width),
                         self._get_ssa_value())

  def sext(self, target_width: int) -> Immediate:
    """
    Sign extension to a wider bit width.
    The top-most bits are filled with copies of the most significant bit.
    """
    if target_width < self._width:
      raise ValueError(
          f"Sign extension target width ({target_width}) must be >= "
          f"current width ({self._width})")

    if target_width == self._width:
      return self

    return arith.ExtSIOp(ir.IntegerType.get_signless(target_width),
                         self._get_ssa_value())

  def __repr__(self) -> str:
    return f"Immediate<{self._width}, {self._value}>"

  def _get_ssa_value(self) -> ir.Value:
    if isinstance(self._value, int):
      self = rtg.ConstantOp(
          ir.IntegerAttr.get(ir.IntegerType.get_signless(self._width),
                             self._value))
    if isinstance(self._value, Integer):
      self = rtg.IntToImmediateOp(ir.IntegerType.get_signless(self._width),
                                  self._value)
    return self._value

  def get_type(self) -> Type:
    return ImmediateType(self._width)


class ImmediateType(Type):
  """
  Represents the type of immediate values with a specific bit width.

  Fields:
    width: int - The bit width of the immediate value
  """

  def __init__(self, width: int):
    self.width = width

  def __eq__(self, other) -> bool:
    return isinstance(other, ImmediateType) and self.width == other.width

  def __repr__(self) -> str:
    return f"ImmediateType<{self.width}>"

  def _codegen(self) -> ir.Type:
    return ir.IntegerType.get(self.width)
