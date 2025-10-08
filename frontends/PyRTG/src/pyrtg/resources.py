#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .rtg import rtg
from .rtgtest import rtgtest
from .core import Value, Type
from .base import ir


class IntegerRegister(Value):
  """
  Represents a RISC-V integer register. Use the static properties to access the
  registers. 'virtual' returns a virtual register that will be resolved to a
  concrete register in the register allocation pass after randomization.
  """

  def __init__(self, value: ir.Value) -> IntegerRegister:
    """
    For library internal use only.
    """

    self._value = value

  def virtual() -> IntegerRegister:
    return rtg.VirtualRegisterOp(
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
        ]))

  def zero() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegZeroAttr.get())

  def ra() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegRaAttr.get())

  def sp() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegSpAttr.get())

  def gp() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegGpAttr.get())

  def tp() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegTpAttr.get())

  def t0() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegT0Attr.get())

  def t1() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegT1Attr.get())

  def t2() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegT2Attr.get())

  def s0() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegS0Attr.get())

  def s1() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegS1Attr.get())

  def a0() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegA0Attr.get())

  def a1() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegA1Attr.get())

  def a2() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegA2Attr.get())

  def a3() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegA3Attr.get())

  def a4() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegA4Attr.get())

  def a5() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegA5Attr.get())

  def a6() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegA6Attr.get())

  def a7() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegA7Attr.get())

  def s2() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegS2Attr.get())

  def s3() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegS3Attr.get())

  def s4() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegS4Attr.get())

  def s5() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegS5Attr.get())

  def s6() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegS6Attr.get())

  def s7() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegS7Attr.get())

  def s8() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegS8Attr.get())

  def s9() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegS9Attr.get())

  def s10() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegS10Attr.get())

  def s11() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegS11Attr.get())

  def t3() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegT3Attr.get())

  def t4() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegT4Attr.get())

  def t5() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegT5Attr.get())

  def t6() -> IntegerRegister:
    return rtg.FixedRegisterOp(rtgtest.RegT6Attr.get())

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
