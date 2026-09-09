#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .base import ir, _get_index_type, _get_signless_integer_type, _get_string_type
from .core import Value, Type
from .rtg import rtg
from .strings import String

from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
  from .immediates import Immediate


class Integer(Value):
  """
  This represents an integer with the same number of bits as a 'size_t' in C.
  It is used to provide parameter values to meta-level constructs such as the
  multiple of an element in a Bag. These integers will be fully constant folded
  away during randomization.
  """

  def __init__(self, value: Union[ir.Value, int]) -> Integer:
    """
    Use this constructor to create an Integer from a builtin Python int.
    """

    self._value = value

  def random(lower_bound: Union[int, Integer],
             upper_bound: Union[int, Integer]) -> Integer:
    """
    Get a random number in the given range (lower and upper inclusive).
    """

    if isinstance(lower_bound, int):
      lower_bound = Integer(lower_bound)
    if isinstance(upper_bound, int):
      upper_bound = Integer(upper_bound)

    op = ir.Operation.create(
        "rtg.random_number_in_range",
        operands=[lower_bound._get_ssa_value(), upper_bound._get_ssa_value()],
        results=[_get_index_type()],
        regions=0)
    return Integer(op.result)

  def __add__(self, other: Integer) -> Integer:
    return Integer(ir.Operation.create(
        "index.add",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        results=[_get_index_type()],
        regions=0).result)

  def __sub__(self, other: Integer) -> Integer:
    return Integer(ir.Operation.create(
        "index.sub",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        results=[_get_index_type()],
        regions=0).result)

  def __mul__(self, other: Integer) -> Integer:
    return Integer(ir.Operation.create(
        "index.mul",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        results=[_get_index_type()],
        regions=0).result)

  def __floordiv__(self, other: Integer) -> Integer:
    return Integer(ir.Operation.create(
        "index.divu",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        results=[_get_index_type()],
        regions=0).result)

  def __truediv__(self, other: Integer) -> Integer:
    return Integer(ir.Operation.create(
        "index.ceildivu",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        results=[_get_index_type()],
        regions=0).result)

  def __mod__(self, other: Integer) -> Integer:
    return Integer(ir.Operation.create(
        "index.remu",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        results=[_get_index_type()],
        regions=0).result)

  def __lshift__(self, other: Integer) -> Integer:
    return Integer(ir.Operation.create(
        "index.shl",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        results=[_get_index_type()],
        regions=0).result)

  def __rshift__(self, other: Integer) -> Integer:
    return Integer(ir.Operation.create(
        "index.shru",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        results=[_get_index_type()],
        regions=0).result)

  def __and__(self, other: Integer) -> Integer:
    return Integer(ir.Operation.create(
        "index.and",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        results=[_get_index_type()],
        regions=0).result)

  def __or__(self, other: Integer) -> Integer:
    return Integer(ir.Operation.create(
        "index.or",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        results=[_get_index_type()],
        regions=0).result)

  def __xor__(self, other: Integer) -> Integer:
    return Integer(ir.Operation.create(
        "index.xor",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        results=[_get_index_type()],
        regions=0).result)

  def _compare(self, predicate: str, other: Integer) -> Immediate:
    from .immediates import Immediate
    return Immediate(
        1,
        ir.Operation.create(
            "index.cmp",
            operands=[self._get_ssa_value(), other._get_ssa_value()],
            attributes={
                "pred": ir.Attribute.parse(f"#index<cmp_predicate {predicate}>")
            },
            results=[_get_signless_integer_type(1)],
            regions=0).result)

  def __eq__(self, other: Integer) -> Immediate:
    return self._compare("eq", other)

  def __ne__(self, other: Integer) -> Immediate:
    return self._compare("ne", other)

  def __lt__(self, other: Integer) -> Immediate:
    return self._compare("ult", other)

  def __le__(self, other: Integer) -> Immediate:
    return self._compare("ule", other)

  def __gt__(self, other: Integer) -> Immediate:
    return self._compare("ugt", other)

  def __ge__(self, other: Integer) -> Immediate:
    return self._compare("uge", other)

  def max(self, other: Integer) -> Integer:
    return Integer(ir.Operation.create(
        "index.maxu",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        results=[_get_index_type()],
        regions=0).result)

  def min(self, other: Integer) -> Integer:
    return Integer(ir.Operation.create(
        "index.minu",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        results=[_get_index_type()],
        regions=0).result)

  def to_string(self) -> String:
    """
    Format this integer as a string in unsigned decimal.
    """

    return String(ir.Operation.create(
        "rtg.int_format",
        operands=[self._get_ssa_value()],
        results=[_get_string_type()],
        regions=0).result)

  def get_type(self) -> Type:
    return IntegerType()

  def _get_ssa_value(self) -> ir.Value:
    if isinstance(self._value, int):
      # The MLIR index type (more precisely the attribute) does not support integers with more than 64 bits.
      if self._value >= 2**64:
        raise ValueError("Integer value out of range")

      # The Python Bindings use an 'int64_t' and the bounds of this type are
      # checked and enforced. However, that integer ends up in a signless APInt
      # anyway, so we can compute the signed integer that matches the bitvector
      # of the unsigned integer and use that.
      if self._value >= 2**63:
        return ir.Operation.create(
            "index.constant",
            attributes={
                "value": ir.IntegerAttr.get(_get_index_type(),
                                             self._value - 2**64)
            },
            results=[_get_index_type()],
            regions=0).result
      else:
        return ir.Operation.create(
            "index.constant",
            attributes={
                "value": ir.IntegerAttr.get(_get_index_type(), self._value)
            },
            results=[_get_index_type()],
            regions=0).result

    return self._value


class IntegerType(Type):
  """
  Represents the type of integer values.
  """

  def __eq__(self, other) -> bool:
    return isinstance(other, IntegerType)

  def _codegen(self) -> ir.Type:
    return _get_index_type()

  def _wrap(self, value: ir.Value) -> Integer:
    return Integer(value)
