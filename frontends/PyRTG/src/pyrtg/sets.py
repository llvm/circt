#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .base import ir
from .rtg import rtg
from .core import Value, Type
from .support import _FromCirctType


class Set(Value):
  """
  Represents a statically typed set for any kind of values that allows picking
  elements at random.
  """

  def __init__(self, value: ir.Value):
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

    if len(elements) == 0:
      raise ValueError("use 'create_empty' to create sets with no elements")
    if not all([e.get_type() == elements[0].get_type() for e in elements]):
      raise TypeError("all elements must have the same type")
    return rtg.SetCreateOp(rtg.SetType.get(elements[0].get_type()._codegen()),
                           elements)

  def __add__(self, other: Value) -> Set:
    """
    If another set is provided their types must match and a new Set will be
    returned containing all elements of both sets (set union). If a value that
    is not a Set is provided, it must match the element type of this Set. A new
    Set will be returned containing all elements of this Set plus the provided
    value.
    """

    if isinstance(other, Set):
      if self.get_type() != other.get_type():
        raise TypeError("sets must be of the same type")
      return rtg.SetUnionOp([self._value, other._value])

    if self.get_type().element_type != other.get_type():
      raise TypeError(
          "type of the provided value must match element type of the set")
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
      if self.get_type() != other.get_type():
        raise TypeError("sets must be of the same type")
      return rtg.SetDifferenceOp(self._value, other._value)

    if self.get_type().element_type != other.get_type():
      raise TypeError(
          "type of the provided value must match element type of the set")
    return self - Set.create(other)

  @staticmethod
  def cartesian_product(*args: Set) -> Set:
    """
    Compute the n-ary cartesian product of the given sets.
    This means, for n input sets it computes
    `X_1 x ... x X_n = {(x_1, ..., x_n) | x_i in X_i for i  in {1, ..., n}}`.
    At least one input set has to be provided (i.e., `n > 0`).
    """

    if len(args) == 0:
      raise ValueError("at least one set must be provided")
    return rtg.SetCartesianProductOp(args)

  def to_bag(self) -> Value:
    """
    Returns this set converted to a bag. Does not modify this object.
    """

    return rtg.SetConvertToBagOp(self)

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

  def get_type(self) -> Type:
    return _FromCirctType(self._value.type)


class SetType(Type):
  """
  Represents the type of statically typed sets.

  Fields:
    element_type: Type
  """

  def __init__(self, element_type: Type):
    self.element_type = element_type

  def __eq__(self, other) -> bool:
    return isinstance(other,
                      SetType) and self.element_type == other.element_type

  def _codegen(self):
    return rtg.SetType.get(self.element_type._codegen())
