#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .rtg import rtg
from .rtgtest import rtgtest
from .core import Value, Type
from .base import ir
from .strings import String


class IntegerRegister(Value):
  """
  Represents an integer register. Use the static properties to access the
  registers. 'virtual' returns a virtual register that will be resolved to a
  concrete register in the register allocation pass after randomization.
  """

  def __init__(self, value: ir.Value) -> IntegerRegister:
    """
    For library internal use only.
    """

    self._value = value

  def virtual() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create(
        "rtg.virtual_reg",
        attributes={"allowedRegs": rtg.VirtualRegisterConfigAttr.get([
            # Choose temporaries with highest priority
            rtgtest.RegT0Attr.get(),
            rtgtest.RegT1Attr.get(),
            rtgtest.RegT2Attr.get(),
            rtgtest.RegT3Attr.get(),
            rtgtest.RegT4Attr.get(),
            rtgtest.RegT5Attr.get(),
            rtgtest.RegT6Attr.get(),
            # Function arguments in reverse order
            rtgtest.RegA7Attr.get(),
            rtgtest.RegA6Attr.get(),
            rtgtest.RegA5Attr.get(),
            rtgtest.RegA4Attr.get(),
            rtgtest.RegA3Attr.get(),
            rtgtest.RegA2Attr.get(),
            rtgtest.RegA1Attr.get(),
            rtgtest.RegA0Attr.get(),
            # Callee saved temporaries
            rtgtest.RegS1Attr.get(),
            rtgtest.RegS2Attr.get(),
            rtgtest.RegS3Attr.get(),
            rtgtest.RegS4Attr.get(),
            rtgtest.RegS5Attr.get(),
            rtgtest.RegS6Attr.get(),
            rtgtest.RegS7Attr.get(),
            rtgtest.RegS8Attr.get(),
            rtgtest.RegS9Attr.get(),
            rtgtest.RegS10Attr.get(),
            rtgtest.RegS11Attr.get(),
            # Some special registers last
            rtgtest.RegS0Attr.get(),
            rtgtest.RegRaAttr.get(),
            rtgtest.RegSpAttr.get(),
        ])},
        results=[rtgtest.IntegerRegisterType.get()],
        regions=0).result)

  def zero() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegZeroAttr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def ra() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegRaAttr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def sp() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegSpAttr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def gp() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegGpAttr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def tp() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegTpAttr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def t0() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegT0Attr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def t1() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegT1Attr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def t2() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegT2Attr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def s0() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegS0Attr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def s1() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegS1Attr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def a0() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegA0Attr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def a1() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegA1Attr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def a2() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegA2Attr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def a3() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegA3Attr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def a4() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegA4Attr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def a5() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegA5Attr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def a6() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegA6Attr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def a7() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegA7Attr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def s2() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegS2Attr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def s3() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegS3Attr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def s4() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegS4Attr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def s5() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegS5Attr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def s6() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegS6Attr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def s7() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegS7Attr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def s8() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegS8Attr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def s9() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegS9Attr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def s10() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegS10Attr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def s11() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegS11Attr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def t3() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegT3Attr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def t4() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegT4Attr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def t5() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegT5Attr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def t6() -> IntegerRegister:
    return IntegerRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegT6Attr.get()}, results=[rtgtest.IntegerRegisterType.get()], regions=0).result)

  def to_string(self) -> String:
    """
    Formats this register as a string.
    """

    return String(ir.Operation.create(
        "rtg.register_format",
        operands=[self._value],
        results=[rtg.StringType.get()], regions=0).result)

  def _get_ssa_value(self) -> ir.Value:
    return self._value

  def get_type(self) -> Type:
    return IntegerRegisterType()


class IntegerRegisterType(Type):
  """
  Represents the type of integer registers.
  """

  def __eq__(self, other) -> bool:
    return isinstance(other, IntegerRegisterType)

  def _codegen(self):
    return rtgtest.IntegerRegisterType.get()

  def _wrap(self, value: ir.Value) -> IntegerRegister:
    return IntegerRegister(value)


class FloatRegister(Value):
  """
  Represents a floating-point register. Use the static properties to access the
  registers. 'virtual' returns a virtual register that will be resolved to a
  concrete register in the register allocation pass after randomization.
  """

  def __init__(self, value: ir.Value) -> FloatRegister:
    """
    For library internal use only.
    """

    self._value = value

  def virtual() -> FloatRegister:
    return FloatRegister(ir.Operation.create(
        "rtg.virtual_reg",
        attributes={"allowedRegs": rtg.VirtualRegisterConfigAttr.get([
            rtgtest.RegF0Attr.get(),
        ])},
        results=[rtgtest.FloatRegisterType.get()],
        regions=0).result)

  def f0() -> FloatRegister:
    return FloatRegister(ir.Operation.create("rtg.constant", attributes={"value": rtgtest.RegF0Attr.get()}, results=[rtgtest.FloatRegisterType.get()], regions=0).result)

  def to_string(self) -> String:
    """
    Formats this register as a string.
    """

    return String(ir.Operation.create(
        "rtg.register_format",
        operands=[self._value],
        results=[rtg.StringType.get()], regions=0).result)

  def _get_ssa_value(self) -> ir.Value:
    return self._value

  def get_type(self) -> Type:
    return FloatRegisterType()


class FloatRegisterType(Type):
  """
  Represents the type of floating-point registers.
  """

  def __eq__(self, other) -> bool:
    return isinstance(other, FloatRegisterType)

  def _codegen(self):
    return rtgtest.FloatRegisterType.get()

  def _wrap(self, value: ir.Value) -> FloatRegister:
    return FloatRegister(value)
