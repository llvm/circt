#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .base import ir
from .rtg import rtg
from .core import Value, Type
from .support import _FromCirctType, _create


class Tuple(Value):
  """
  Represents a statically-typed immutable tuple. Each tuple has a fixed number
  of elements of potentially different types.
  """

  def __init__(self, value: ir.Value, type: TupleType = None) -> Tuple:
    """
    Intended for library internal usage only.
    """

    self._value = value
    self._type = type

  def create(*elements: Value) -> Tuple:
    """
    Create a tuple containing the provided values. At least one
    element must be provided. Each element can be of a different type.
    """

    op = _create(rtg.TupleCreateOp, elements)
    return Tuple(op.result, TupleType([element.get_type() for element in elements]))

  def __getitem__(self, i) -> Value:
    """
    Access an element in the tuple at the specified index (read-only).
    """

    if not isinstance(i, int):
      raise TypeError("index must be a python int")

    op = _create(rtg.TupleExtractOp, self, i)
    return self.get_type().element_types[i]._wrap(op.result)

  def _get_ssa_value(self) -> ir.Value:
    return self._value

  def get_type(self) -> Type:
    if self._type is not None:
      return self._type
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
    return rtg.TupleType.get([ty._codegen() for ty in self.element_types])

  def _wrap(self, value: ir.Value) -> Tuple:
    return Tuple(value, self)
