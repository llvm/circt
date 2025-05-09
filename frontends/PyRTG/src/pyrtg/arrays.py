#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .circt import ir
from .rtg import rtg
from .index import index
from .core import Value
from .integers import Integer

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

  def create(elements: list[Value], element_type: ir.Type) -> Array:
    """
    Create an array containing the provided values. All elements must have the
    same type.
    """

    if not all([e.get_type() == element_type for e in elements]):
      raise TypeError(
          "all elements of an RTG array must be of the specified element type")

    return rtg.ArrayCreateOp(rtg.ArrayType.get(element_type), elements)

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

  def _get_ssa_value(self) -> ir.Value:
    return self._value

  def get_type(self) -> ir.Type:
    return self._value.type

  def type(element_type: ir.Type) -> ir.Type:
    """
    Returns the array type for the given element type.
    """

    return rtg.ArrayType.get(element_type)
