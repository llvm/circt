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

    if isinstance(value, bool):
      value = 1 if value else 0

    self._value = value
    self._width = 1

  def to_string(self) -> String:
    """
    Format this integer as a string in unsigned decimal.
    """

    return rtg.BoolFormatOp(self)

  def get_type(self) -> Type:
    return BoolType()


class BoolType(ImmediateType):
  """
  Represents the type of boolean values.
  """

  def __init__(self):
    self.width = 1

  def __eq__(self, other) -> bool:
    return isinstance(other, BoolType)
