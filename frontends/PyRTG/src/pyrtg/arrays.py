#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .base import ir
from .rtg import rtg
from .index import index
from .core import Value, Type
from .integers import Integer
from .support import _FromCirctType

from typing import Union


class Array(Value):
  """
  Represents a statically typed array for any kind of values.
  """

  def __init__(self, value: ir.Value) -> Array:
    """
    Intended for library internal usage only.
    """

    self._value = value

  def create(elements: list[Value], element_type: Type) -> Array:
    """
    Create an array containing the provided values. All elements must have the
    same type.
    """

    if not all([e.get_type() == element_type for e in elements]):
      raise TypeError(
          "all elements of an RTG array must be of the specified element type")

    return rtg.ArrayCreateOp(rtg.ArrayType.get(element_type._codegen()),
                             elements)

  def __getitem__(self, i) -> Value:
    """
    Access an element in the array at the specified index (read-only).
    """

    assert isinstance(i, (int, Integer)), "slicing not supported yet"

    idx = i
    if isinstance(i, int):
      idx = index.ConstantOp(i)

    return rtg.ArrayExtractOp(self._value, idx)

  def set(self, index: Union[int, Integer], value: Value) -> Array:
    """
    Set an element at the specified index in the array.
    """

    index = index if isinstance(index, Integer) else Integer(index)
    return rtg.ArrayInjectOp(self._value, index, value)

  def size(self) -> Integer:
    """
    Get the number of elements in the array.
    """

    return rtg.ArraySizeOp(self._value)

  def append(self, element: Value) -> Array:
    """
    Append an element to the end of the array.
    """

    return rtg.ArrayAppendOp(self._value, element)

  def __add__(self, other: Value) -> Array:
    return self.append(other)

  def _get_ssa_value(self) -> ir.Value:
    return self._value

  def get_type(self) -> Type:
    return _FromCirctType(self._value.type)


class ArrayType(Type):
  """
  Represents the type of statically typed arrays.

  Fields:
    element_type: Type
  """

  def __init__(self, element_type: Type):
    self.element_type = element_type

  def __eq__(self, other) -> bool:
    return isinstance(other,
                      ArrayType) and self.element_type == other.element_type

  def _codegen(self) -> ir.Type:
    return rtg.ArrayType.get(self.element_type._codegen())
