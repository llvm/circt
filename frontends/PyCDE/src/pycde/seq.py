#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .circt.dialects import seq as raw_seq
from .constructs import Wire

from .types import Bits, Type
from .signals import _FromCirctValue, BitsSignal, ClockSignal, Signal


class FIFO:
  """Creates a FIFO operation with the specified type, depth, clock, and reset
  signal. Adds push and pop methods to wire up the FIFO."""

  def __init__(self,
               type: Type,
               depth: int,
               clk: ClockSignal,
               rst: BitsSignal,
               rd_latency: int = 0):
    self.type = type
    self.input = Wire(type)
    self.wr_en = Wire(Bits(1))
    self.rd_en = Wire(Bits(1))
    i1 = Bits(1)._type
    self.fifo = raw_seq.FIFOOp(self.input.type._type,
                               i1,
                               i1,
                               i1,
                               i1,
                               self.input.value,
                               self.rd_en.value,
                               self.wr_en.value,
                               clk.value,
                               rst.value,
                               depth,
                               rdLatency=rd_latency,
                               almostFullThreshold=depth,
                               almostEmptyThreshold=0)
    self._output = _FromCirctValue(self.fifo.output)

  def push(self, data: Signal, en: BitsSignal):
    """Connect 'data' to the FIFO input and 'en' to write enable."""
    self.input.assign(data)
    self.wr_en.assign(en)

  def pop(self, en: BitsSignal):
    """Wire up 'en' to read enable and returns the FIFO output."""
    self.rd_en.assign(en)
    return self._output

  @property
  def output(self):
    return self._output

  @property
  def full(self):
    return _FromCirctValue(self.fifo.full)

  @property
  def empty(self):
    return _FromCirctValue(self.fifo.empty)
