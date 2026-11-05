#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Hardware for the cosim backpressure integration tests.
#
#   * GatedLoopback ("loopback"): loops `in` back to `out`, but only while the
#     last message on `gate` was non-zero. With the gate closed, `in` is never
#     consumed, so host writes back up (host -> simulation backpressure).
#
#   * Producer ("producer"): a `produce` command with N makes it emit 0..N-1 on
#     `stream` as fast as the host side accepts them. A message on `query`
#     returns, on `sent`, how many it has emitted so far. That lets the host
#     observe that the producer was stalled by an unread `stream` (simulation ->
#     host backpressure) rather than having buffered everything host-side.

import sys

from pycde import AppID, Clock, Module, Reset, System, esi, generator
from pycde.constructs import Counter, Wire
from pycde.types import Bits, Channel, UInt

from esiaccel.bsp import get_bsp


class GatedLoopback(Module):
  clk = Clock()
  rst = Reset()

  @generator
  def build(ports):
    gate = esi.ChannelService.from_host(AppID("gate"), UInt(8))
    gate_data, gate_valid = gate.unwrap(Bits(1)(1))
    enabled = gate_data.as_bits()[0].reg(ports.clk,
                                         ports.rst,
                                         ce=gate_valid,
                                         name="enabled")

    inp = esi.ChannelService.from_host(AppID("in"), UInt(32))
    in_ready = Wire(Bits(1))
    data, valid = inp.unwrap(in_ready)
    out, out_ready = Channel(UInt(32)).wrap(data, valid & enabled)
    in_ready.assign(out_ready & enabled)
    esi.ChannelService.to_host(AppID("out"), out)


class Producer(Module):
  clk = Clock()
  rst = Reset()

  @generator
  def build(ports):
    produce = esi.ChannelService.from_host(AppID("produce"), UInt(32))
    count, count_valid = produce.unwrap(Bits(1)(1))
    target = count.reg(ports.clk, ports.rst, ce=count_valid, name="target")

    xact = Wire(Bits(1))
    sent = Counter(32)(clk=ports.clk,
                       rst=ports.rst,
                       clear=Bits(1)(0),
                       increment=xact)
    valid = sent.out < target
    stream, stream_ready = Channel(UInt(32)).wrap(sent.out, valid)
    xact.assign(valid & stream_ready)
    esi.ChannelService.to_host(AppID("stream"), stream)

    query = esi.ChannelService.from_host(AppID("query"), UInt(8))
    query_ready = Wire(Bits(1))
    _, query_valid = query.unwrap(query_ready)
    resp, resp_ready = Channel(UInt(32)).wrap(sent.out, query_valid)
    query_ready.assign(resp_ready)
    esi.ChannelService.to_host(AppID("sent"), resp)


class Top(Module):
  clk = Clock()
  rst = Reset()

  @generator
  def build(ports):
    GatedLoopback(clk=ports.clk, rst=ports.rst, appid=AppID("loopback"))
    Producer(clk=ports.clk, rst=ports.rst, appid=AppID("producer"))


if __name__ == "__main__":
  bsp = get_bsp(sys.argv[2] if len(sys.argv) > 2 else None)
  s = System(bsp(Top), name="BackpressureTest", output_directory=sys.argv[1])
  s.compile()
  s.package()
