#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .base import ir
from .core import Type
from .rtg import rtg
from .strings import String
from .immediates import Immediate, ImmediateType

from typing import Union


class Bool(Immediate):
  """
  This represents a boolean value. It is used to provide boolean parameter
  values to meta-level constructs. These booleans will be fully constant folded
  away during randomization.
  """

  def __init__(self, value: Union[ir.Value, bool]) -> Bool:
    """
    Use this constructor to create a Bool from a builtin Python bool.
    """

    super().__init__(1, value)

  def to_string(self) -> String:
    """
    Format this integer as a string in unsigned decimal.
    """

    return rtg.BoolFormatOp(self)

  def get_type(self) -> Type:
    return BoolType()

  def _get_ssa_value(self) -> ir.Value:
    if isinstance(self._value, bool):
      self = rtg.ConstantOp(ir.BoolAttr.get(self._value))

    return self._value


class BoolType(ImmediateType):
  """
  Represents the type of boolean values.
  """

  def __eq__(self, other) -> bool:
    return isinstance(other, BoolType)

  def _codegen(self) -> ir.Type:
    return ir.IntegerType.get_signless(1)
