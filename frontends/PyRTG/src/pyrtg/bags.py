#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .circt import ir, support
from .rtg import rtg
from .core import Value
from .index import index

import typing


class Bag(Value):
  """
  Represents a statically typed bag or also called multiset for any kind of
  values that allows picking elements at random.
  """

  def __init__(self, value: ir.Value) -> Bag:
    """
    Intended for library internal usage only.
    """

    self._value = value

  def create_empty(elementType: ir.Type) -> Bag:
    """
    Create an empty bag that can hold elements of the provided type.
    """

    return rtg.BagCreateOp(rtg.BagType.get(elementType), [], [])

  def create(*elements: tuple[typing.Union[Value, int], Value]) -> Bag:
    """
    Create a bag containing the provided values. At least one element must be
    provided.
    """

    assert len(
        elements) > 0, "use 'create_empty' to create sets with no elements"
    assert all([e.get_type() == elements[0][1].get_type() for _, e in elements
               ]), "all elements must have the same type"
    return rtg.BagCreateOp(
        rtg.BagType.get(elements[0][1].get_type()), [x for _, x in elements],
        [(x if not isinstance(x, int) else index.ConstantOp(x))
         for x, _ in elements])

  def __add__(self, other: Value) -> Bag:
    """
    If another bag is provided their types must match and a new bag will be
    returned containing all elements of both bags (bag/multiset union). If a
    value that is not a bag is provided, it must match the element type of this
    bag. A new bag will be returned containing all elements of this bag plus
    the provided value.
    """

    if isinstance(other, Bag):
      assert self.get_type() == other.get_type(
      ), "bags must be of the same type"
      return rtg.BagUnionOp([self._value, other._value])

    assert support.type_to_pytype(
        self.get_type()).element_type == other.get_type(
        ), "type of the provided value must match element type of the bag"
    return self + Bag.create((1, other))

  def __sub__(self, other: Value) -> Bag:
    """
    If another bag is provided their types must match and a new bag will be
    returned where the number of occurences of an element in this bag is
    reduced by the number of occurences of the same element in the other bag.
    If a value that is not a bag is provided, it must match the element type of
    this bag. A new bag will be returned containing all elements of this bag
    with the number of occurences of the provided element is reduced by 1.
    """

    if isinstance(other, Bag):
      assert self.get_type() == other.get_type(
      ), "bags must be of the same type"
      return rtg.BagDifferenceOp(self._value, other._value)

    assert support.type_to_pytype(
        self.get_type()).element_type == other.get_type(
        ), "type of the provided value must match element type of the bag"
    return self - Bag.create((1, other))

  def exclude(self, other: Value) -> Value:
    """
    If another bag is provided their types must match and a new bag will be
    returned where all occurences of an element in this bag are removed if the
    same element is present at least once in the other bag. If a value that is
    not a bag is provided, it must match the element type of this bag. A new
    bag will be returned containing all elements of this bag except the
    provided value.
    """

    if isinstance(other, Bag):
      assert self.get_type() == other.get_type(
      ), "bags must be of the same type"
      return rtg.BagDifferenceOp(self._value,
                                 other._value,
                                 inf=ir.UnitAttr.get())

    assert support.type_to_pytype(
        self.get_type()).element_type == other.get_type(
        ), "type of the provided value must match element type of the bag"
    return self.exclude(Bag.create((1, other)))

  def get_random(self) -> Value:
    """
    Returns an element from the bag picked uniformly at random (i.e., the
    multiples of items effectively are the weights in the random distribution).
    If the bag is empty, calling this method is undefined behavior.
    """

    return rtg.BagSelectRandomOp(self._value)

  def get_random_and_exclude(self) -> Value:
    """
    Returns an element from the bag picked uniformly at random (i.e., the
    multiples of items effectively are the weights in the random distribution)
    and removes all occurences of that item from the bag. If the bag is empty,
    calling this method is undefined behavior.
    """

    r = self.get_random()
    self._value = self.exclude(r)._get_ssa_value()
    return r

  def _get_ssa_value(self) -> ir.Value:
    return self._value

  def get_type(self) -> ir.Type:
    return self._value.type

  def type(*args: ir.Type) -> ir.Type:
    """
    Returns the bag type for the given element type.
    """

    assert len(args) == 1, "Bag type requires exactly one element type"
    return rtg.BagType.get(args[0])
