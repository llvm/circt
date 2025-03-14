#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .rtg import rtg
from .rtgtest import rtgtest
from .core import Value
from .circt import ir

from typing import Union


class CPUCore(Value):
  """
  Represents a CPU core in the test environment. A CPU core can be specified
  either by its hardware thread ID (hartid) or can represent all available
  cores. This class allows operations to target specific cores or all cores
  when generating randomized tests.
  """

  def __init__(self, hartid: Union[int, ir.Value]) -> CPUCore:
    """
    Creates a CPUCore instance for a specific hardware thread ID.
    """

    self._value = hartid

  def all() -> CPUCore:
    """
    Creates a CPUCore instance that represents all available CPU cores. This is
    useful when some instructions should be executed across all cores in the
    system.
    """

    return rtg.ConstantOp(rtg.AllContextsAttr.get(rtgtest.CPUType.get()))

  def _get_ssa_value(self) -> ir.Value:
    if isinstance(self._value, int):
      self = rtg.ConstantOp(rtgtest.CPUAttr.get(self._value))
    return self._value

  def get_type(self) -> ir.Type:
    return rtgtest.CPUType.get()

  def type(*args) -> ir.Type:
    return rtgtest.CPUType.get()
