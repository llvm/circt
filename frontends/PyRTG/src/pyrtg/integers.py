#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .base import ir
from .core import Value
from .index import index
from .rtg import rtg

from typing import Union


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
    Get a random number in the given range (lower inclusive, upper exclusive).
    """

    if isinstance(lower_bound, int):
      lower_bound = Integer(lower_bound)
    if isinstance(upper_bound, int):
      upper_bound = Integer(upper_bound)

    return rtg.RandomNumberInRangeOp(lower_bound, upper_bound)

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

  def __eq__(self, other: Integer) -> Bool:
    return index.CmpOp("eq", self._get_ssa_value(), other._get_ssa_value())

  def __ne__(self, other: Integer) -> Bool:
    return index.CmpOp("ne", self._get_ssa_value(), other._get_ssa_value())

  def __lt__(self, other: Integer) -> Bool:
    return index.CmpOp("ult", self._get_ssa_value(), other._get_ssa_value())

  def __le__(self, other: Integer) -> Bool:
    return index.CmpOp("ule", self._get_ssa_value(), other._get_ssa_value())

  def __gt__(self, other: Integer) -> Bool:
    return index.CmpOp("ugt", self._get_ssa_value(), other._get_ssa_value())

  def __ge__(self, other: Integer) -> Bool:
    return index.CmpOp("uge", self._get_ssa_value(), other._get_ssa_value())

  def get_type(self) -> ir.Type:
    return ir.IndexType.get()

  def _get_ssa_value(self) -> ir.Value:
    if isinstance(self._value, int):
      self = index.ConstantOp(self._value)

    return self._value

  @staticmethod
  def ty() -> ir.Type:
    """
    Returns the index type.
    """

    return ir.IndexType.get()


class Bool(Value):
  """
  This represents a boolean value. It is used to provide boolean parameter
  values to meta-level constructs. These booleans will be fully constant folded
  away during randomization.
  """

  def __init__(self, value: Union[ir.Value, bool]) -> Bool:
    """
    Use this constructor to create a Bool from a builtin Python bool.
    """

    self._value = value

  def get_type(self) -> ir.Type:
    return ir.IntegerType.get_signless(1)

  def _get_ssa_value(self) -> ir.Value:
    if isinstance(self._value, bool):
      self = index.BoolConstantOp(self._value)

    return self._value

  @staticmethod
  def ty() -> ir.Type:
    """
    Returns the 'i1' type representing a boolean.
    """

    return ir.IntegerType.get_signless(1)
