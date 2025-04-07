#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .base import ir
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

  def create_empty(elementType: ir.Type) -> Array:
    """
    Create an empty array that can hold elements of the provided type.
    """

    return rtg.ArrayCreateOp(rtg.ArrayType.get(elementType), [])

  def create(*elements: Value) -> Array:
    """
    Create an array containing the provided values. At least one element must
    be provided.
    """

    assert len(
        elements) > 0, "use 'create_empty' to create arrays with no elements"
    assert all([e.get_type() == elements[0].get_type() for e in elements
               ]), "all elements must have the same type"
    return rtg.ArrayCreateOp(rtg.ArrayType.get(elements[0].get_type()),
                             elements)

  def from_list(py_list) -> Array:
    """
    Create an array from a Python list. Handles arbitrarily nested lists
    by recursively converting them to RTG arrays. The list must contain at least
    one element and all elements at each nesting level must have the same type.
    """

    assert len(
        py_list) > 0, "use 'create_empty' to create arrays with no elements"

    # If the first element is a list, we have a nested structure
    if isinstance(py_list[0], list):
      assert all(isinstance(elem, list) for elem in py_list
                ), "all elements at same nesting level must be lists"

      # Recursively convert each nested list
      converted_elements = [Array.from_list(elem) for elem in py_list]
      return rtg.ArrayCreateOp(
          rtg.ArrayType.get(converted_elements[0].get_type()),
          converted_elements)

    # Base case: list of RTG Values
    assert all(isinstance(elem, Value)
               for elem in py_list), "all elements must be RTG Values"
    assert all(elem.get_type() == py_list[0].get_type()
               for elem in py_list), "all elements must have the same type"

    return rtg.ArrayCreateOp(rtg.ArrayType.get(py_list[0].get_type()), py_list)

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

  @staticmethod
  def ty(*args) -> ir.Type:
    """
    Returns the array type for the given element type.
    """

    assert len(args) == 1 and isinstance(
        args[0], ir.Type), "Array type requires exactly one element type"
    return rtg.ArrayType.get(args[0])
