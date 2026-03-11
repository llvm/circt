#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This file contains the header for auto-generated RTGTest instruction wrappers.
# The instruction wrappers are generated from the RTGTest dialect TableGen definitions.

from typing import Union
from pyrtg import instruction, SideEffect, Type, Value, Immediate, ImmediateType, Label, LabelType, Memory, MemoryType, IntegerRegister, IntegerRegisterType, FloatRegister, FloatRegisterType, rtgtest


def get_virtual_reg(ty: Type) -> Value:
  if ty == IntegerRegisterType():
    return IntegerRegister.virtual()
  elif ty == FloatRegisterType():
    return FloatRegister.virtual()
  else:
    raise ValueError(f"Unsupported register type: {ty}")
