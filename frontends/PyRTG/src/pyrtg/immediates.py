#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .rtg import rtg
from .core import Value, Type
from .base import ir
from .integers import Integer
from .strings import String
from .arith import arith

from typing import Union


class Immediate(Value):

  def __init__(self, width: int, value: Union[ir.Value, int,
                                              Integer]) -> Immediate:
    if width < 0:
      raise ValueError(f"width must be non-negative, got {width}")

    if isinstance(value, int):
      # Note: it's valid to pass in a negative value here, but also a positive
      # value that is small enough for a unsigned representation
      max = (1 << width) - 1
      min = -(1 << (width - 1))

      if value < min or value > max:
        raise ValueError(
            f"Value {value} does not fit in {width}-bit representation "
            f"(valid range: [{min}, {max}])")

      # Convert to signed representation if the high bit is set because MLIR
      # built-in integer attributes are constructed with signed integers.
      if value >= (1 << (width - 1)):
        value = value - (1 << width)

    self._width = width
    self._value = value

  @staticmethod
  def random(width: int) -> Immediate:
    """
    An immediate of the provided width of a random value from 0 to the maximum
    unsigned number the immediate can hold (all bits set).
    """

    # Note that the upper limit is exclusive
    return Immediate(width, Integer.random(0, 2**width - 1))

  @staticmethod
  def concat(*args: Immediate) -> Immediate:
    """
    Concatenates this immediate with the provided immediates. The operands are
    concatenated in order, with this immediate becoming the most significant
    bits of the result.
    """

    if len(args) == 0:
      raise ValueError("At least one immediate required")

    width = sum(arg._width for arg in args)
    return Immediate(width, ir.Operation.create(
        "rtg.isa.concat_immediate",
        operands=[arg._get_ssa_value() for arg in args],
        results=[ir.IntegerType.get_signless(width)],
        regions=0).result)

  def replicate(self, count: int) -> Immediate:
    """
    Replicates this immediate the provided number of times. The result is an
    immediate of a width equal to the width of this immediate multiplied by
    `count` and containing this immediate concatenated with itself `count`
    times.
    """

    if count < 0:
      raise ValueError("replicate count must be non-negative")

    return Immediate.concat(*([self] * count))

  def __getitem__(self, slice_range) -> Immediate:
    """
    Extracts bits from the immediate using Python slice notation.
    The least significant bit has index 0.
    """

    if isinstance(slice_range, slice):
      start = slice_range.start if slice_range.start is not None else 0
      stop = slice_range.stop if slice_range.stop is not None else self._width
      if slice_range.step is not None and slice_range.step != 1:
        raise ValueError("Step value other than 1 is not supported")
      if start < 0 or stop > self._width or start >= stop:
        raise ValueError(
            f"Invalid slice range [{start}:{stop}] for width {self._width}")
      width = stop - start
      return Immediate(width, ir.Operation.create(
          "rtg.isa.slice_immediate",
          operands=[self._get_ssa_value()],
          attributes={"lowBit": ir.IntegerAttr.get(
              ir.IntegerType.get_signless(32), start)},
          results=[ir.IntegerType.get_signless(width)],
          regions=0).result)

    if isinstance(slice_range, int):
      if slice_range < 0 or slice_range >= self._width:
        raise ValueError(
            f"Index {slice_range} out of range for width {self._width}")
      return Immediate(1, ir.Operation.create(
          "rtg.isa.slice_immediate",
          operands=[self._get_ssa_value()],
          attributes={"lowBit": ir.IntegerAttr.get(
              ir.IntegerType.get_signless(32), slice_range)},
          results=[ir.IntegerType.get_signless(1)],
          regions=0).result)

    raise TypeError("Slice must be an integer or slice object")

  @staticmethod
  def umax(width: int) -> Immediate:
    """
    An immediate of the provided width with the maximum unsigned value it can
    hold.
    """

    return Immediate(width, 2**width - 1)

  @staticmethod
  def smax(width: int) -> Immediate:
    """
    An immediate of the provided width with the maximum signed value it can
    hold.
    """

    return Immediate(width, 2**(width - 1) - 1)

  @staticmethod
  def smin(width: int) -> Immediate:
    """
    An immediate of the provided width with the minimum signed value it can
    hold.
    """

    return Immediate(width, 1 << (width - 1))

  def to_string(self) -> String:
    """
    Formats this immediate as a string.
    """

    return String(ir.Operation.create(
        "rtg.immediate_format",
        operands=[self._get_ssa_value()],
        results=[rtg.StringType.get()],
        regions=0).result)

  def __add__(self, other: Immediate) -> Immediate:
    return Immediate(self._width, ir.Operation.create(
        "arith.addi",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        results=[self.get_type()._codegen()],
        regions=0).result)

  def __sub__(self, other: Immediate) -> Immediate:
    return Immediate(self._width, ir.Operation.create(
        "arith.subi",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        results=[self.get_type()._codegen()],
        regions=0).result)

  def __mul__(self, other: Immediate) -> Immediate:
    return Immediate(self._width, ir.Operation.create(
        "arith.muli",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        results=[self.get_type()._codegen()],
        regions=0).result)

  def __lshift__(self, other: Immediate) -> Immediate:
    return Immediate(self._width, ir.Operation.create(
        "arith.shli",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        results=[self.get_type()._codegen()],
        regions=0).result)

  def __rshift__(self, other: Immediate) -> Immediate:
    return Immediate(self._width, ir.Operation.create(
        "arith.shrui",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        results=[self.get_type()._codegen()],
        regions=0).result)

  def __and__(self, other: Immediate) -> Immediate:
    return Immediate(self._width, ir.Operation.create(
        "arith.andi",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        results=[self.get_type()._codegen()],
        regions=0).result)

  def __or__(self, other: Immediate) -> Immediate:
    return Immediate(self._width, ir.Operation.create(
        "arith.ori",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        results=[self.get_type()._codegen()],
        regions=0).result)

  def __xor__(self, other: Immediate) -> Immediate:
    return Immediate(self._width, ir.Operation.create(
        "arith.xori",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        results=[self.get_type()._codegen()],
        regions=0).result)

  def __eq__(self, other: Immediate) -> Value:
    return Immediate(1, ir.Operation.create(
        "arith.cmpi",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        attributes={"predicate": ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64), int(arith.CmpIPredicate.eq))},
        results=[ir.IntegerType.get_signless(1)],
        regions=0).result)

  def __ne__(self, other: Immediate) -> Value:
    return Immediate(1, ir.Operation.create(
        "arith.cmpi",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        attributes={"predicate": ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64), int(arith.CmpIPredicate.ne))},
        results=[ir.IntegerType.get_signless(1)],
        regions=0).result)

  def ult(self, other: Immediate) -> Value:
    """
    Unsigned less than comparison.
    """
    return Immediate(1, ir.Operation.create(
        "arith.cmpi",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        attributes={"predicate": ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64), int(arith.CmpIPredicate.ult))},
        results=[ir.IntegerType.get_signless(1)],
        regions=0).result)

  def ule(self, other: Immediate) -> Value:
    """
    Unsigned less than or equal comparison.
    """
    return Immediate(1, ir.Operation.create(
        "arith.cmpi",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        attributes={"predicate": ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64), int(arith.CmpIPredicate.ule))},
        results=[ir.IntegerType.get_signless(1)], regions=0).result)

  def ugt(self, other: Immediate) -> Value:
    """
    Unsigned greater than comparison.
    """
    return Immediate(1, ir.Operation.create(
        "arith.cmpi",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        attributes={"predicate": ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64), int(arith.CmpIPredicate.ugt))},
        results=[ir.IntegerType.get_signless(1)], regions=0).result)

  def uge(self, other: Immediate) -> Value:
    """
    Unsigned greater than or equal comparison.
    """
    return Immediate(1, ir.Operation.create(
        "arith.cmpi",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        attributes={"predicate": ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64), int(arith.CmpIPredicate.uge))},
        results=[ir.IntegerType.get_signless(1)], regions=0).result)

  def slt(self, other: Immediate) -> Value:
    """
    Signed less than comparison.
    """
    return Immediate(1, ir.Operation.create(
        "arith.cmpi",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        attributes={"predicate": ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64), int(arith.CmpIPredicate.slt))},
        results=[ir.IntegerType.get_signless(1)], regions=0).result)

  def sle(self, other: Immediate) -> Value:
    """
    Signed less than or equal comparison.
    """
    return Immediate(1, ir.Operation.create(
        "arith.cmpi",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        attributes={"predicate": ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64), int(arith.CmpIPredicate.sle))},
        results=[ir.IntegerType.get_signless(1)], regions=0).result)

  def sgt(self, other: Immediate) -> Value:
    """
    Signed greater than comparison.
    """
    return Immediate(1, ir.Operation.create(
        "arith.cmpi",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        attributes={"predicate": ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64), int(arith.CmpIPredicate.sgt))},
        results=[ir.IntegerType.get_signless(1)], regions=0).result)

  def sge(self, other: Immediate) -> Value:
    """
    Signed greater than or equal comparison.
    """
    return Immediate(1, ir.Operation.create(
        "arith.cmpi",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        attributes={"predicate": ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64), int(arith.CmpIPredicate.sge))},
        results=[ir.IntegerType.get_signless(1)], regions=0).result)

  def umax_of(self, other: Immediate) -> Immediate:
    """
    Unsigned maximum of this immediate and another.
    """
    return Immediate(self._width, ir.Operation.create(
        "arith.maxui",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        results=[self.get_type()._codegen()], regions=0).result)

  def umin_of(self, other: Immediate) -> Immediate:
    """
    Unsigned minimum of this immediate and another.
    """
    return Immediate(self._width, ir.Operation.create(
        "arith.minui",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        results=[self.get_type()._codegen()], regions=0).result)

  def smax_of(self, other: Immediate) -> Immediate:
    """
    Signed maximum of this immediate and another.
    """
    return Immediate(self._width, ir.Operation.create(
        "arith.maxsi",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        results=[self.get_type()._codegen()], regions=0).result)

  def smin_of(self, other: Immediate) -> Immediate:
    """
    Signed minimum of this immediate and another.
    """
    return Immediate(self._width, ir.Operation.create(
        "arith.minsi",
        operands=[self._get_ssa_value(), other._get_ssa_value()],
        results=[self.get_type()._codegen()], regions=0).result)

  def zext(self, target_width: int) -> Immediate:
    """
    Zero extension (unsigned extension) to a wider bit width.
    The top-most bits are filled with zeros.
    """
    if target_width < self._width:
      raise ValueError(
          f"Zero extension target width ({target_width}) must be >= "
          f"current width ({self._width})")

    if target_width == self._width:
      return self

    return Immediate(target_width, ir.Operation.create(
        "arith.extui",
        operands=[self._get_ssa_value()],
        results=[ir.IntegerType.get_signless(target_width)], regions=0).result)

  def sext(self, target_width: int) -> Immediate:
    """
    Sign extension to a wider bit width.
    The top-most bits are filled with copies of the most significant bit.
    """
    if target_width < self._width:
      raise ValueError(
          f"Sign extension target width ({target_width}) must be >= "
          f"current width ({self._width})")

    if target_width == self._width:
      return self

    return Immediate(target_width, ir.Operation.create(
        "arith.extsi",
        operands=[self._get_ssa_value()],
        results=[ir.IntegerType.get_signless(target_width)], regions=0).result)

  def __repr__(self) -> str:
    return f"Immediate<{self._width}, {self._value}>"

  def _get_ssa_value(self) -> ir.Value:
    if isinstance(self._value, int):
      return ir.Operation.create(
          "rtg.constant",
          attributes={"value": ir.IntegerAttr.get(
              ir.IntegerType.get_signless(self._width), self._value)},
          results=[ir.IntegerType.get_signless(self._width)],
          regions=0).result
    if isinstance(self._value, Integer):
      return ir.Operation.create(
          "rtg.isa.int_to_immediate",
          operands=[self._value._get_ssa_value()],
          results=[ir.IntegerType.get_signless(self._width)],
          regions=0).result
    return self._value

  def get_type(self) -> Type:
    return ImmediateType(self._width)


class ImmediateType(Type):
  """
  Represents the type of immediate values with a specific bit width.

  Fields:
    width: int - The bit width of the immediate value
  """

  def __init__(self, width: int):
    self.width = width

  def __eq__(self, other) -> bool:
    return isinstance(other, ImmediateType) and self.width == other.width

  def __repr__(self) -> str:
    return f"ImmediateType<{self.width}>"

  def _codegen(self) -> ir.Type:
    return ir.IntegerType.get(self.width)

  def _wrap(self, value: ir.Value) -> Immediate:
    return Immediate(self.width, value)
