#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .circt import ir
from .core import Value
from .index import index

import typing


class Integer(Value):
  """
  This represents an integer with the same number of bits as a 'size_t' in C.
  It is used to provide parameter values to meta-level constructs such as the
  multiple of an element in a Bag. These integers will be fully constant folded
  away during randomization.
  """

  def __init__(self, value: typing.Union[ir.Value, int]) -> Integer:
    """
    Use this constructor to create an Integer from a builtin Python int.
    """

    self._value = value

  def __add__(self, other: Integer) -> Integer:
    return index.AddOp(self._get_ssa_value(), other._get_ssa_value())

  def __sub__(self, other: Integer) -> Integer:
    return index.SubOp(self._get_ssa_value(), other._get_ssa_value())

  def __and__(self, other: Integer) -> Integer:
    return index.AndOp(self._get_ssa_value(), other._get_ssa_value())

  def __or__(self, other: Integer) -> Integer:
    return index.OrOp(self._get_ssa_value(), other._get_ssa_value())

  def __xor__(self, other: Integer) -> Integer:
    return index.XOrOp(self._get_ssa_value(), other._get_ssa_value())

  def get_type(self) -> ir.Type:
    return ir.IndexType.get()

  def _get_ssa_value(self) -> ir.Value:
    if isinstance(self._value, int):
      self = index.ConstantOp(self._value)

    return self._value
