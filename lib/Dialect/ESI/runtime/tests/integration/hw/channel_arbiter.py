#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Hardware for the ChannelArbiter cosim integration tests. Builds two
# independent DUTs under a single top:
#
#   * ChannelArbiterTest ("arbiter_test"): NUM_INPUTS host-driven `from_host`
#     UInt(32) channels multiplexed into a single `to_host` channel. The host
#     checks that every value is delivered exactly once (no loss / duplication).
#     Instantiated twice: a power-of-two ("balanced") count and a
#     non-power-of-two ("unbalanced") count.
#
#   * ChannelArbiterListTest ("list_test"): two in-hardware list-window
#     producers (distinct `src` ids and message lengths) contend for the
#     arbiter. A hardware checker verifies that, once a message is granted, its
#     flits arrive contiguously (no interleaving) up to the flit tagged `last`,
#     and reports (sticky-error, src, count) per completed message on a
#     `to_host` channel.
#
#   * ChannelArbiterTokenTest ("token_test"): TOKEN_NUM_INPUTS in-hardware
#     zero-width (`i0`) token producers, each emitting exactly TOKENS_PER_INPUT
#     tokens, contend for the arbiter. A hardware counter reports a running
#     1-based ordinal per delivered token, so the host can confirm every token
#     is delivered exactly once (conservation) and every producer is served
#     (no starvation) -- exercising the credit-counter-as-buffer i0 path.

import sys

from pycde import (AppID, Clock, Input, Module, Output, Reset, System,
                   generator)
from pycde.constructs import Counter, Mux, Wire
from pycde.types import Bits, Channel, List as ListType, StructType, UInt, Window
from pycde import esi

from esiaccel.bsp import get_bsp
from esiaccel.components import ChannelArbiter

NUM_INPUTS = 4  # power-of-two ("balanced") input count.
ODD_NUM_INPUTS = 3  # non-power-of-two ("unbalanced") input count.
PIPE_NUM_INPUTS = 6  # input count for the pipelined-mux-tree variant.
TOKEN_NUM_INPUTS = 5  # input count for the zero-width (i0) token variant.
TOKENS_PER_INPUT = 8  # tokens each producer emits in the token test.

# A list-window payload: a struct with a `src` tag and a variable-length list.
# `Window.default_of` adds a per-flit `last` field to the lowered frame struct,
# which is exactly what `ChannelArbiter` uses to keep messages contiguous.
ListInto = StructType({'src': UInt(8), 'items': ListType(UInt(16))})
Flit = Window.default_of(ListInto)
FlitLowered = Flit.lowered_type  # struct<src: ui8, items: ui16, last: i1>


def HostMux(num_inputs: int, mux_pipeline_levels=None):
  """A host-driven single-flit multiplexer: `num_inputs` `from_host` UInt(32)
  channels muxed into a single `to_host` channel. `mux_pipeline_levels` pipelines
  the selection mux tree."""

  class HostMux(Module):
    clk = Clock()
    rst = Reset()

    @generator
    def build(ports):
      ins = [
          esi.ChannelService.from_host(AppID(f"in_{i}"), UInt(32))
          for i in range(num_inputs)
      ]
      out = ChannelArbiter(ins,
                           ports.clk,
                           ports.rst,
                           mux_pipeline_levels=mux_pipeline_levels,
                           telemetry=False)
      esi.ChannelService.to_host(AppID("out"), out)

  HostMux.__name__ = f"HostMux_{num_inputs}_{mux_pipeline_levels}"
  return HostMux


def ListProducer(src_id: int, length: int):
  """A module which continuously emits back-to-back list messages of `length`
  flits tagged with `src_id`. `items` counts 0..length-1 and `last` is set on
  the final flit. Always valid, so two of these contend for the arbiter."""

  class ListProducer(Module):
    clk = Clock()
    rst = Reset()
    out = Output(Channel(Flit))

    @generator
    def build(ports):
      i = Wire(UInt(8))
      last = i == UInt(8)(length - 1)
      st = FlitLowered({
          'src': UInt(8)(src_id),
          'items': i.as_uint(16),
          'last': last,
      })
      chan, ready = Channel(Flit).wrap(Flit.wrap(st), Bits(1)(1))
      ports.out = chan
      # valid is constant 1, so a transaction happens whenever `ready`.
      nxt = Mux(last, (i + UInt(8)(1)).as_uint(8), UInt(8)(0))
      i.assign(nxt.reg(ports.clk, ports.rst, ce=ready, rst_value=0))

  ListProducer.__name__ = f"ListProducer_src{src_id}_len{length}"
  return ListProducer


class ListChecker(Module):
  """Consumes the muxed list-window stream and verifies message contiguity.

  Emits one UInt(32) report per completed message:
    bit  24    : sticky interleave-error flag (should stay 0)
    bits 23:16 : src of the completed message
    bits 15:0  : running completed-message count
  """

  clk = Clock()
  rst = Reset()
  in_ = Input(Channel(Flit))
  report = Output(Channel(UInt(32)))

  @generator
  def build(ports):
    active = Wire(Bits(1))
    err = Wire(Bits(1))

    in_ready = Wire(Bits(1))
    win, valid = ports.in_.unwrap(in_ready)
    st = win.unwrap()
    src = st['src']
    last = st['last']

    # A message boundary: the current flit completes a message.
    is_completing = valid & last
    # Emit a report exactly when a message completes; back-pressure the input
    # on that flit until the report is accepted.
    report_valid = is_completing

    # src of the in-progress message, latched at its first flit.
    start_any = valid & in_ready & ~active
    cur_src = src.reg(ports.clk, ports.rst, ce=start_any, rst_value=0)

    # Interleave error: a flit whose src differs from the owner mid-message.
    interleave_err = (valid & in_ready) & active & (src != cur_src)
    err.assign((err | interleave_err).reg(ports.clk, ports.rst, rst_value=0))

    # `active` tracks whether we are mid-message (past the first flit, before
    # `last`).
    xact = valid & in_ready
    begin_multi = xact & ~active & ~last
    end_msg = xact & last
    active_next = Mux(end_msg, Mux(begin_multi, active, Bits(1)(1)), Bits(1)(0))
    active.assign(active_next.reg(ports.clk, ports.rst, rst_value=0))

    # Completed-message counter.
    msg_count = Counter(16)(clk=ports.clk,
                            rst=ports.rst,
                            clear=Bits(1)(0),
                            increment=end_msg)

    report_data = ((err.as_uint(32) * UInt(32)(0x1000000)).as_uint(32) +
                   (cur_src.as_uint(32) * UInt(32)(0x10000)).as_uint(32) +
                   msg_count.out.as_uint(32)).as_uint(32)
    report_chan, report_ready = Channel(UInt(32)).wrap(report_data,
                                                       report_valid)
    ports.report = report_chan

    # Accept every non-completing flit; on a completing flit, only accept when
    # the report is accepted so no completion is dropped.
    in_ready.assign(Mux(is_completing, Bits(1)(1), report_ready))


class ChannelArbiterListTest(Module):
  """Two contending list producers -> arbiter -> contiguity checker."""

  clk = Clock()
  rst = Reset()

  @generator
  def build(ports):
    p0 = ListProducer(1, 3)(clk=ports.clk, rst=ports.rst)
    p1 = ListProducer(2, 4)(clk=ports.clk, rst=ports.rst)
    muxed = ChannelArbiter([p0.out, p1.out],
                           ports.clk,
                           ports.rst,
                           telemetry=False)
    chk = ListChecker(clk=ports.clk, rst=ports.rst, in_=muxed)
    esi.ChannelService.to_host(AppID("report"), chk.report)


def TokenProducer(count: int):
  """Emits exactly `count` zero-width (`i0`) tokens then idles: `valid` stays
  high until `count` tokens have been accepted. There is no payload -- only the
  valid/ready handshake carries information."""

  class TokenProducer(Module):
    clk = Clock()
    rst = Reset()
    out = Output(Channel(Bits(0)))

    @generator
    def build(ports):
      xact = Wire(Bits(1))
      sent = Counter(16)(clk=ports.clk,
                         rst=ports.rst,
                         clear=Bits(1)(0),
                         increment=xact)
      valid = sent.out < UInt(16)(count)
      chan, ready = Channel(Bits(0)).wrap(Bits(0)(0), valid)
      ports.out = chan
      xact.assign(valid & ready)

  TokenProducer.__name__ = f"TokenProducer_{count}"
  return TokenProducer


class TokenChecker(Module):
  """Consumes the muxed zero-width token stream and emits one UInt(32) report
  per delivered token carrying its 1-based ordinal (1, 2, 3, ...). Backpressure
  from the report channel is fed to the arbiter, exercising its credit buffer.
  """

  clk = Clock()
  rst = Reset()
  in_ = Input(Channel(Bits(0)))
  report = Output(Channel(UInt(32)))

  @generator
  def build(ports):
    in_ready = Wire(Bits(1))
    _tok, valid = ports.in_.unwrap(in_ready)  # zero-width payload: ignore data.

    # Running count of delivered tokens; this token's ordinal is count + 1.
    count = Counter(32)(clk=ports.clk,
                        rst=ports.rst,
                        clear=Bits(1)(0),
                        increment=valid & in_ready)
    report_data = (count.out + UInt(32)(1)).as_uint(32)
    report_chan, report_ready = Channel(UInt(32)).wrap(report_data, valid)
    ports.report = report_chan
    # Accept a token exactly when its report is consumed by the host.
    in_ready.assign(report_ready)


class ChannelArbiterTokenTest(Module):
  """`TOKEN_NUM_INPUTS` bounded zero-width token producers -> arbiter -> token
  counter. Each producer emits exactly `TOKENS_PER_INPUT` tokens, so the host
  must see exactly TOKEN_NUM_INPUTS * TOKENS_PER_INPUT tokens: no loss or
  duplication (conservation), and every producer served (no starvation, since
  the total can only be reached if each producer's tokens all get through)."""

  clk = Clock()
  rst = Reset()

  @generator
  def build(ports):
    prods = [
        TokenProducer(TOKENS_PER_INPUT)(clk=ports.clk, rst=ports.rst)
        for _ in range(TOKEN_NUM_INPUTS)
    ]
    muxed = ChannelArbiter([p.out for p in prods],
                           ports.clk,
                           ports.rst,
                           telemetry=False)
    chk = TokenChecker(clk=ports.clk, rst=ports.rst, in_=muxed)
    esi.ChannelService.to_host(AppID("token_report"), chk.report)


class Top(Module):
  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    HostMux(NUM_INPUTS)(clk=ports.clk,
                        rst=ports.rst,
                        appid=AppID("arbiter_test"))
    HostMux(ODD_NUM_INPUTS)(clk=ports.clk,
                            rst=ports.rst,
                            appid=AppID("arbiter_test_odd"))
    HostMux(PIPE_NUM_INPUTS,
            mux_pipeline_levels=1)(clk=ports.clk,
                                   rst=ports.rst,
                                   appid=AppID("arbiter_test_pipe"))
    ChannelArbiterListTest(clk=ports.clk,
                           rst=ports.rst,
                           appid=AppID("list_test"))
    ChannelArbiterTokenTest(clk=ports.clk,
                            rst=ports.rst,
                            appid=AppID("token_test"))


if __name__ == "__main__":
  bsp = get_bsp(sys.argv[2] if len(sys.argv) > 2 else None)
  s = System(bsp(Top), name="ChannelArbiterTest", output_directory=sys.argv[1])
  s.compile()
  s.package()
