# ===- flow.py - channel flow-control primitives ---------------------------===//
#
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===//
#
#  Flow-control building blocks for ESI channels.
#
# ===-----------------------------------------------------------------------===//

from pycde.common import Clock, Input, InputChannel, Output, OutputChannel, Reset
from pycde.constructs import Mux, Reg, Wire
from pycde.module import Module, generator, modparams
from pycde.support import clog2
from pycde.types import Bits, Type, UInt


@modparams
def MaxOutstandingLimiter(
    data_type: Type, max_outstanding: int) -> type["MaxOutstandingLimiterImpl"]:
  """Rate-limit a channel to at most `max_outstanding` outstanding
  transactions. A transaction becomes outstanding when a message is accepted
  on `in_` and remains outstanding until `complete` is pulsed. `in_` is
  stalled whenever `max_outstanding` transactions are already outstanding.

  `complete` must pulse for exactly one cycle per completed transaction and
  must not pulse while zero transactions are outstanding; the module does not
  check this. When an input transaction and a `complete` pulse occur on the
  same cycle the outstanding count is left unchanged.
  """

  if max_outstanding < 1:
    raise ValueError("'max_outstanding' must be at least 1.")

  counter_width = clog2(max_outstanding + 1)

  class MaxOutstandingLimiterImpl(Module):
    clk = Clock()
    rst = Reset()
    in_ = InputChannel(data_type)
    # One-cycle pulse per completed transaction.
    complete = Input(Bits(1))
    out = OutputChannel(data_type)

    @generator
    def build(ports):
      clk = ports.clk
      rst = ports.rst
      complete = ports.complete

      # Registered count of outstanding transactions.
      count = Reg(UInt(counter_width), clk, rst, name="outstanding")

      # Only accept a new transaction while below the limit.
      can_issue = count < UInt(counter_width)(max_outstanding)

      # Forward the input channel to the output, gated by 'can_issue'.
      in_ready = Wire(Bits(1))
      in_data, in_valid = ports.in_.unwrap(in_ready)
      out_valid = (in_valid & can_issue).as_bits()
      out_chan, out_ready = MaxOutstandingLimiterImpl.out.type.wrap(
          in_data, out_valid)
      in_ready.assign(out_ready & can_issue)
      ports.out = out_chan

      # +1 on an accepted input, -1 on a 'complete' pulse. If both occur on
      # the same cycle the count is unchanged. 'can_issue' guarantees the
      # counter never overflows; the caller guarantees it never underflows.
      issue = out_valid & out_ready
      up = issue & ~complete
      down = complete & ~issue
      one = UInt(counter_width)(1)
      count_plus_1 = (count + one).as_uint(counter_width)
      count_minus_1 = (count - one).as_uint(counter_width)
      # up=1 -> count_plus_1; up=0, down=1 -> count_minus_1; else count.
      count.assign(Mux(up, Mux(down, count, count_minus_1), count_plus_1))

  return MaxOutstandingLimiterImpl
