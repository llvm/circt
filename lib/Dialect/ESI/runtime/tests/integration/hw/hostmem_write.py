#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys

from pycde import AppID, Clock, Module, Reset, System, esi, generator
from pycde.constructs import Wire
from pycde.types import Bits, StructType, UInt

from esiaccel.bsp.common import TaggedWriteGearbox
from esiaccel.bsp.cosim import CosimBSP


def Writer(width):

  class Writer(Module):
    clk = Clock()
    rst = Reset()

    @generator
    def build(ports):
      req_type = esi.HostMem.write_req_channel_type(Bits(width))
      wire_type = esi.HostMem.write_req_channel_type(Bits((width + 7) // 8 * 8))
      req = esi.ChannelService.from_host(AppID("req"), wire_type)
      req = req.transform(lambda r: req_type({
          "address": r.address,
          "tag": r.tag,
          "data": r.data[:width],
      }))
      ack = esi.HostMem.write(AppID("host"), req)
      gate = esi.ChannelService.from_host(AppID("gate"), UInt(8))
      gate_data, gate_valid = gate.unwrap(Bits(1)(1))
      enabled = gate_data.as_bits()[0].reg(ports.clk,
                                           ports.rst,
                                           ce=gate_valid,
                                           name="enabled")
      ready = Wire(Bits(1))
      tag, valid = ack.unwrap(ready)
      out, out_ready = ack.type.wrap(tag, valid & enabled)
      ready.assign(out_ready & enabled)
      esi.ChannelService.to_host(AppID("ack"), out)

  return Writer


def Top(bus_width):

  class Top(Module):
    clk = Clock()
    rst = Reset()

    @generator
    def build(ports):
      for width in [13, bus_width, bus_width + 24, bus_width * 2 + 24, 16384]:
        Writer(width)(clk=ports.clk, rst=ports.rst, appid=AppID("wr", width))

      window = esi.HostMem.write_window(Bits(24), 1)
      req_type = StructType([("address", UInt(64)), ("data", Bits(24)),
                             ("last", UInt(8))])
      req = esi.ChannelService.from_host(AppID("list_req"), req_type)
      req = req.transform(lambda r: window.wrap(
          window.lowered_type({
              "address": r.address,
              "tag": 0,
              "data": [r.data],
              "data_size": 0,
              "last": r.last.as_bits()[0],
          })))
      ack = esi.HostMem.write(AppID("list_host"), req)
      esi.ChannelService.to_host(AppID("list_ack"), ack)

      # Observe the same gearbox independently to check exact beat counts,
      # addresses, byte enables and payload-cap framing without timing asserts.
      gearbox = TaggedWriteGearbox(16384, bus_width, max(256, bus_width // 8))
      req = esi.ChannelService.from_host(AppID("probe_req"),
                                         gearbox.in_.type.inner_type)
      probe = gearbox(clk=ports.clk, rst=ports.rst, in_=req)
      resp_type = StructType([("address", UInt(64)), ("tag", UInt(8)),
                              ("data", Bits(bus_width)),
                              ("valid_bytes", UInt(64)), ("last", UInt(8))])
      resp = probe.out.transform(lambda r: resp_type({
          "address": r.address,
          "tag": r.tag,
          "data": r.data,
          "valid_bytes": r.valid_bytes.as_uint(64),
          "last": r.last.as_uint(8),
      }))
      esi.ChannelService.to_host(AppID("probe_resp"), resp)

  return Top


if __name__ == "__main__":
  bus_width = int(sys.argv[2]) if len(sys.argv) > 2 else 64
  kwargs = {"hostmem_width": bus_width} if len(sys.argv) > 2 else {}
  system = System(CosimBSP(Top(bus_width), **kwargs),
                  name="HostMemWrite",
                  output_directory=sys.argv[1])
  system.compile()
  system.package()
