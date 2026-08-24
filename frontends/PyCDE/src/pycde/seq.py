#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .circt.dialects import seq as raw_seq

from .common import Clock, Input, Output, Reset
from .module import Module, generator, modparams
from .types import Bits, Type, UInt
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
    from .constructs import Wire

    if type.bitwidth is None:
      raise ValueError("FIFO type must have a defined bitwidth")
    if type.bitwidth == 0:
      raise ValueError("FIFO type must have a non-zero bitwidth")
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


@modparams
def Counter(width: int, increment_on_clear: bool = False):
  """Construct an up counter of the specified width. `increment` counts up by
  one, `clear` returns the count to zero. The count wraps around on overflow.

  `increment_on_clear` selects what happens when `clear` and `increment` are
  asserted in the same cycle:

  - `False` (default): `clear` takes priority and that cycle's `increment` is
    dropped -- the count becomes 0.
  - `True`: `clear` zeroes the *accumulated* count and that cycle's
    `increment` still counts -- the count becomes 1. Use this when `clear`
    means "I have consumed the count so far" and events must not be lost,
    e.g. a counter which is cleared when it is read out."""

  class Counter(Module):
    clk = Clock()
    rst = Reset()
    clear = Input(Bits(1))
    increment = Input(Bits(1))
    out = Output(UInt(width))

    @generator
    def construct(ports):
      from .constructs import Mux, Reg

      # Note: `Mux(sel, a, b)` selects `a` when `sel` is 0 and `b` when it
      # is 1 (it indexes its data inputs by `sel`).
      count = Reg(UInt(width),
                  clk=ports.clk,
                  rst=ports.rst,
                  rst_value=0,
                  ce=ports.increment | ports.clear,
                  name="count")
      zero = UInt(width)(0)
      if increment_on_clear:
        # The value this cycle's increment applies to.
        base = Mux(ports.clear, count, zero)
        base.name = "base"
        next = Mux(ports.increment, base, (base + 1).as_uint(width))
      else:
        next = Mux(ports.clear, (count + 1).as_uint(width), zero)
      count.assign(next)
      ports.out = count

  return Counter


@modparams
def DownCounter(width: int, decrement_on_load: bool = False):
  """Construct a saturating down counter of the specified width: a countdown
  timer / credit counter. `load` loads `load_value` into the counter and
  `decrement` counts down by one. The count saturates at zero -- it never
  wraps around -- and `is_zero` reports (combinationally, with no added
  latency) whether the current count is zero. The counter resets to zero, so
  `is_zero` is high until something is loaded.

  `decrement_on_load` selects what happens when `load` and `decrement` are
  asserted in the same cycle:

  - `False` (default): `load` takes priority and that cycle's `decrement` is
    dropped -- the count becomes `load_value`.
  - `True`: the decrement applies to the newly loaded value -- the count
    becomes `load_value - 1` (saturating at zero). Use this when `decrement`
    events must not be lost, e.g. a credit counter which is topped up while
    credits are being spent."""

  class DownCounter(Module):
    clk = Clock()
    rst = Reset()
    load = Input(Bits(1))
    load_value = Input(UInt(width))
    decrement = Input(Bits(1))
    out = Output(UInt(width))
    is_zero = Output(Bits(1))

    @generator
    def construct(ports):
      # Deferred: `constructs` imports from this module, so a top-level
      # import here would be circular.
      from .constructs import Mux, Reg

      # Note: `Mux(sel, a, b)` selects `a` when `sel` is 0 and `b` when it
      # is 1 (it indexes its data inputs by `sel`).
      count = Reg(UInt(width),
                  clk=ports.clk,
                  rst=ports.rst,
                  rst_value=0,
                  ce=ports.load | ports.decrement,
                  name="count")
      zero = UInt(width)(0)
      one = UInt(width)(1)
      count_is_zero = count == zero
      count_is_zero.name = "count_is_zero"

      # The value this cycle's decrement applies to, and whether decrementing
      # it would underflow.
      if decrement_on_load:
        base = Mux(ports.load, count, ports.load_value)
        base.name = "base"
        base_is_zero = Mux(ports.load, count_is_zero, ports.load_value == zero)
        base_is_zero.name = "base_is_zero"
      else:
        base = count
        base_is_zero = count_is_zero
      # Saturate rather than wrap.
      base_dec = Mux(base_is_zero, (base - one).as_uint(width), base)
      next = Mux(ports.decrement, base, base_dec)
      if not decrement_on_load:
        # `load` takes priority over a coincident `decrement`.
        next = Mux(ports.load, next, ports.load_value)
      count.assign(next)

      ports.out = count
      ports.is_zero = count_is_zero

  return DownCounter
