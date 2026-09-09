#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .base import ir
from .rtg import rtg
from .index import index
from .core import Value, Type
from .integers import Integer
from .support import _FromCirctType, _create

from typing import Union


class Array(Value):
  """
  Represents a statically typed array for any kind of values.
  """

  def __init__(self, value: ir.Value, type: ArrayType = None) -> Array:
    """
    Intended for library internal usage only.
    """

    self._value = value
    self._type = type

  def create(elements: list[Value], element_type: Type) -> Array:
    """
    Create an array containing the provided values. All elements must have the
    same type.
    """

    if not all([e.get_type() == element_type for e in elements]):
      raise TypeError(
          "all elements of an RTG array must be of the specified element type")

    op = _create(rtg.ArrayCreateOp, rtg.ArrayType.get(element_type._codegen()),
                 elements)
    return Array(op.result, ArrayType(element_type))

  def __getitem__(self, i) -> Value:
    """
    Access an element in the array at the specified index (read-only).
    """

    assert isinstance(i, (int, Integer)), "slicing not supported yet"

    idx = i
    if isinstance(i, int):
      idx = _create(index.ConstantOp, i).result

    op = _create(rtg.ArrayExtractOp, self._value, idx)
    return self.get_type().element_type._wrap(op.result)

  def set(self, index: Union[int, Integer], value: Value) -> Array:
    """
    Set an element at the specified index in the array.
    """

    index = index if isinstance(index, Integer) else Integer(index)
    op = _create(rtg.ArrayInjectOp, self._value, index, value)
    return Array(op.result, self.get_type())

  def size(self) -> Integer:
    """
    Get the number of elements in the array.
    """

    return Integer(_create(rtg.ArraySizeOp, self._value).result)

  def append(self, element: Value) -> Array:
    """
    Append an element to the end of the array.
    """

    op = _create(rtg.ArrayAppendOp, self._value, element)
    return Array(op.result, self.get_type())

  def __add__(self, other: Value) -> Array:
    return self.append(other)

  def _get_ssa_value(self) -> ir.Value:
    return self._value

  def get_type(self) -> Type:
    if self._type is not None:
      return self._type
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

  def _wrap(self, value: ir.Value) -> Array:
    return Array(value, self)
