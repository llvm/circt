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
            # Function arguments in reverse order
            rtgtest.RegA5Attr.get(),
            rtgtest.RegA4Attr.get(),
            rtgtest.RegA3Attr.get(),
            rtgtest.RegA2Attr.get(),
            rtgtest.RegA1Attr.get(),
            rtgtest.RegA0Attr.get(),
            # Callee saved temporaries
            rtgtest.RegS1Attr.get(),
            # Some special registers last
            rtgtest.RegS0Attr.get(),
            rtgtest.RegRaAttr.get(),
            rtgtest.RegSpAttr.get(),
        ]))

  def zero() -> IntegerRegister:
    return rtg.ConstantOp(rtgtest.RegZeroAttr.get())

  def ra() -> IntegerRegister:
    return rtg.ConstantOp(rtgtest.RegRaAttr.get())

  def sp() -> IntegerRegister:
    return rtg.ConstantOp(rtgtest.RegSpAttr.get())

  def gp() -> IntegerRegister:
    return rtg.ConstantOp(rtgtest.RegGpAttr.get())

  def tp() -> IntegerRegister:
    return rtg.ConstantOp(rtgtest.RegTpAttr.get())

  def t0() -> IntegerRegister:
    return rtg.ConstantOp(rtgtest.RegT0Attr.get())

  def t1() -> IntegerRegister:
    return rtg.ConstantOp(rtgtest.RegT1Attr.get())

  def t2() -> IntegerRegister:
    return rtg.ConstantOp(rtgtest.RegT2Attr.get())

  def s0() -> IntegerRegister:
    return rtg.ConstantOp(rtgtest.RegS0Attr.get())

  def s1() -> IntegerRegister:
    return rtg.ConstantOp(rtgtest.RegS1Attr.get())

  def a0() -> IntegerRegister:
    return rtg.ConstantOp(rtgtest.RegA0Attr.get())

  def a1() -> IntegerRegister:
    return rtg.ConstantOp(rtgtest.RegA1Attr.get())

  def a2() -> IntegerRegister:
    return rtg.ConstantOp(rtgtest.RegA2Attr.get())

  def a3() -> IntegerRegister:
    return rtg.ConstantOp(rtgtest.RegA3Attr.get())

  def a4() -> IntegerRegister:
    return rtg.ConstantOp(rtgtest.RegA4Attr.get())

  def a5() -> IntegerRegister:
    return rtg.ConstantOp(rtgtest.RegA5Attr.get())

  def to_string(self) -> String:
    """
    Formats this register as a string.
    """

    return rtg.RegisterFormatOp(self)

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
