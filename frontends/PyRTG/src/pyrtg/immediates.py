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

  @staticmethod
  def umax(width: int) -> Immediate:
    """
    An immediate of the provided width with the maximum unsigned value it can
    hold.
    """

    return Immediate(width, 2**width - 1)

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
