#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .base import ir
from .rtg import rtg
from .core import Value, Type
from .support import _FromCirctType


class Tuple(Value):
  """
  Represents a statically-typed immutable tuple. Each tuple has a fixed number
  of elements of potentially different types.
  """

  def __init__(self, value: ir.Value) -> Tuple:
    """
    Intended for library internal usage only.
    """

    self._value = value

  def create(*elements: Value) -> Tuple:
    """
    Create a tuple containing the provided values. At least one
    element must be provided. Each element can be of a different type.
    """

    if len(elements) == 0:
      raise ValueError("at least one element must be present")

    return rtg.TupleCreateOp(elements)

  def __getitem__(self, i) -> Value:
    """
    Access an element in the tuple at the specified index (read-only).
    """

    if not isinstance(i, int):
      raise TypeError("index must be a python int")

    return rtg.TupleExtractOp(self, i)

  def _get_ssa_value(self) -> ir.Value:
    return self._value

  def get_type(self) -> Type:
    return _FromCirctType(self._value.type)


class TupleType(Type):
  """
  Represents the type of statically typed tuples.

  Fields:
    element_types: list[Type]
  """

  def __init__(self, element_types: list[Type]):
    self.element_types = element_types

  def __eq__(self, other) -> bool:
    return isinstance(other,
                      TupleType) and self.element_types == other.element_types

  def _codegen(self) -> ir.Type:
    return ir.TupleType.get_tuple([ty._codegen() for ty in self.element_types])
