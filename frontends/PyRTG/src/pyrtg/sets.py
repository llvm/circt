#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .circt import ir, support
from .rtg import rtg
from .core import Value


class Set(Value):
  """
  Represents a statically typed set for any kind of values that allows picking
  elements at random.
  """

  def __init__(self, value: ir.Value) -> Set:
    """
    Intended for library internal usage only.
    """

    self._value = value

  def create_empty(elementType: ir.Type) -> Set:
    """
    Create an empty set that can hold elements of the provided type.
    """

    return rtg.SetCreateOp(rtg.SetType.get(elementType), [])

  def create(*elements: Value) -> Set:
    """
    Create a set containing the provided values. At least one element must be
    provided.
    """

    assert len(
        elements) > 0, "use 'create_empty' to create sets with no elements"
    assert all([e.get_type() == elements[0].get_type() for e in elements
               ]), "all elements must have the same type"
    return rtg.SetCreateOp(rtg.SetType.get(elements[0].get_type()), elements)

  def __add__(self, other: Value) -> Set:
    """
    If another set is provided their types must match and a new Set will be
    returned containing all elements of both sets (set union). If a value that
    is not a Set is provided, it must match the element type of this Set. A new
    Set will be returned containing all elements of this Set plus the provided
    value.
    """

    if isinstance(other, Set):
      assert self.get_type() == other.get_type(
      ), "sets must be of the same type"
      return rtg.SetUnionOp([self._value, other._value])

    assert support.type_to_pytype(
        self.get_type()).element_type == other.get_type(
        ), "type of the provided value must match element type of the set"
    return self + Set.create(other)

  def __sub__(self, other: Value) -> Set:
    """
    If another set is provided their types must match and a new Set will be
    returned containing all elements in this Set that are not contained in the
    'other' Set. If a value that is not a Set is provided, it must match the
    element type of this Set. A new Set will be returned containing all
    elements of this Set except the provided value in that case.
    """

    if isinstance(other, Set):
      assert self.get_type() == other.get_type(
      ), "sets must be of the same type"
      return rtg.SetDifferenceOp(self._value, other._value)

    assert support.type_to_pytype(
        self.get_type()).element_type == other.get_type(
        ), "type of the provided value must match element type of the set"
    return self - Set.create(other)

  def get_random(self) -> Value:
    """
    Returns an element from the set picked uniformly at random. If the set is
    empty, calling this method is undefined behavior.
    """

    return rtg.SetSelectRandomOp(self._value)

  def get_random_and_exclude(self) -> Value:
    """
    Returns an element from the set picked uniformly at random and removes it
    from the set. If the set is empty, calling this method is undefined
    behavior.
    """

    r = self.get_random()
    self._value = (self - r)._get_ssa_value()
    return r

  def _get_ssa_value(self) -> ir.Value:
    return self._value

  def get_type(self) -> ir.Type:
    return self._value.type

  def type(*args: ir.Type) -> ir.Type:
    """
    Returns the set type for the given element type.
    """

    assert len(args) == 1, "Set type requires exactly one element type"
    return rtg.SetType.get(args[0])
