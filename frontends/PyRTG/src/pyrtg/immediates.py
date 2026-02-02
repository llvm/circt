#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .rtg import rtg
from .core import Value, Type
from .base import ir
from .integers import Integer

from typing import Union


class Immediate(Value):

  def __init__(self, width: int, value: Union[ir.Value, int,
                                              Integer]) -> Immediate:
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

  def concat(self, *others: Immediate) -> Immediate:
    """
    Concatenates this immediate with the provided immediates. The operands are
    concatenated in order, with this immediate becoming the most significant
    bits of the result.
    """

    return rtg.ConcatImmediateOp([self, *others])

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

  def __repr__(self) -> str:
    return f"Immediate<{self._width}, {self._value}>"

  def _get_ssa_value(self) -> ir.Value:
    if isinstance(self._value, int):
      self = rtg.ConstantOp(rtg.ImmediateAttr.get(self._width, self._value))
    if isinstance(self._value, Integer):
      self = rtg.IntToImmediateOp(rtg.ImmediateType.get(self._width),
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
    return rtg.ImmediateType.get(self.width)
