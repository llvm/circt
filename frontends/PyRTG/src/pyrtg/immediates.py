#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .rtg import rtg
from .core import Value
from .base import ir
from .integers import Integer

from typing import Union


class Immediate(Value):

  def __init__(self, width: int, value: Union[ir.Value, int,
                                              Integer]) -> Immediate:
    self._width = width
    self._value = value

  def _get_ssa_value(self) -> ir.Value:
    if isinstance(self._value, int):
      self = rtg.ConstantOp(rtg.ImmediateAttr.get(self._width, self._value))
    if isinstance(self._value, Integer):
      self = rtg.IntToImmediateOp(rtg.ImmediateType.get(self._width),
                                  self._value)
    return self._value

  def get_type(self) -> ir.Type:
    return rtg.ImmediateType.get(self._width)

  @staticmethod
  def ty(width: int) -> ir.Type:
    return rtg.ImmediateType.get(width)
