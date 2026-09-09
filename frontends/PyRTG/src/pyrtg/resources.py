#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .rtg import rtg
from .rtgtest import rtgtest
from .core import Value, Type
from .base import ir
from .strings import String
from .support import _create


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
    return IntegerRegister(_create(rtg.VirtualRegisterOp,
        rtg.VirtualRegisterConfigAttr.get([
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
        ])).result)

  def zero() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegZeroAttr.get()).result)

  def ra() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegRaAttr.get()).result)

  def sp() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegSpAttr.get()).result)

  def gp() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegGpAttr.get()).result)

  def tp() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegTpAttr.get()).result)

  def t0() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegT0Attr.get()).result)

  def t1() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegT1Attr.get()).result)

  def t2() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegT2Attr.get()).result)

  def s0() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegS0Attr.get()).result)

  def s1() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegS1Attr.get()).result)

  def a0() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegA0Attr.get()).result)

  def a1() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegA1Attr.get()).result)

  def a2() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegA2Attr.get()).result)

  def a3() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegA3Attr.get()).result)

  def a4() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegA4Attr.get()).result)

  def a5() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegA5Attr.get()).result)

  def a6() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegA6Attr.get()).result)

  def a7() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegA7Attr.get()).result)

  def s2() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegS2Attr.get()).result)

  def s3() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegS3Attr.get()).result)

  def s4() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegS4Attr.get()).result)

  def s5() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegS5Attr.get()).result)

  def s6() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegS6Attr.get()).result)

  def s7() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegS7Attr.get()).result)

  def s8() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegS8Attr.get()).result)

  def s9() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegS9Attr.get()).result)

  def s10() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegS10Attr.get()).result)

  def s11() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegS11Attr.get()).result)

  def t3() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegT3Attr.get()).result)

  def t4() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegT4Attr.get()).result)

  def t5() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegT5Attr.get()).result)

  def t6() -> IntegerRegister:
    return IntegerRegister(_create(rtg.ConstantOp, rtgtest.RegT6Attr.get()).result)

  def to_string(self) -> String:
    """
    Formats this register as a string.
    """

    return String(_create(rtg.RegisterFormatOp, self).result)

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
    return FloatRegister(_create(rtg.VirtualRegisterOp,
        rtg.VirtualRegisterConfigAttr.get([
            rtgtest.RegF0Attr.get(),
        ])).result)

  def f0() -> FloatRegister:
    return FloatRegister(_create(rtg.ConstantOp, rtgtest.RegF0Attr.get()).result)

  def to_string(self) -> String:
    """
    Formats this register as a string.
    """

    return String(_create(rtg.RegisterFormatOp, self).result)

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
