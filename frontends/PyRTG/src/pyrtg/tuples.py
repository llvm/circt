#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .base import ir
from .rtg import rtg
from .core import Value


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

    assert len(elements) > 0, "at least one element must be present"
    return rtg.TupleCreateOp(elements)

  def __getitem__(self, i) -> Value:
    """
    Access an element in the tuple at the specified index (read-only).
    """

    assert isinstance(i, int), "index must be a python int"
    return rtg.TupleExtractOp(self, i)

  def _get_ssa_value(self) -> ir.Value:
    return self._value

  def get_type(self) -> ir.Type:
    return self._value.type

  @staticmethod
  def ty(*args) -> ir.Type:
    """
    Returns the tuple type for the given element types.
    """

    return ir.TupleType.get_tuple(args)
