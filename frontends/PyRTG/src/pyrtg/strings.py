#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .base import ir
from .core import Value, Type
from .rtg import rtg

from typing import Union


class String(Value):
  """
  Represents an immutable string value.
  """

  def __init__(self, value: Union[ir.Value, str]):
    if isinstance(value, str):
      self._value = rtg.ConstantOp(
          ir.StringAttr.get_typed(rtg.StringType.get(),
                                  value))._get_ssa_value()
    else:
      self._value = value

  def __add__(self, other: String) -> String:
    """
    String concatenation.
    """

    return rtg.StringConcatOp([self, other])

  def get_type(self) -> Type:
    return StringType()

  def _get_ssa_value(self) -> ir.Value:
    return self._value


class StringType(Type):
  """
  Represents the type of string values.
  """

  def __eq__(self, other) -> bool:
    return isinstance(other, StringType)

  def _codegen(self) -> ir.Type:
    return rtg.StringType.get()
