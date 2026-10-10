#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Hardware for the HostMem multiple-outstanding-read cosim test. One DUT per
# element width in ELEMENT_WIDTHS, each a host-driven `read_list` client:
#
#   * `req` (from_host): {address, length (elements)} read requests, forwarded
#     straight to the `read_list` request channel.
#   * `gate` (from_host): bit 0 opens (1) or closes (0) the response side. While
#     closed, no response element is consumed, so every request the BSP accepts
#     stays in flight inside the read processor.
#   * `resp` (to_host): every response element with its `last` flag, for the
#     host to check data, element counts and burst framing.
#   * `accepted` (telemetry): number of read requests accepted by the BSP.
#
# Usage: hostmem_read.py <output_dir> <max_outstanding_reads>

import sys

from pycde import AppID, Clock, Module, Reset, System, generator
from pycde import esi
from pycde.constructs import Counter, Mux, Reg, Wire
from pycde.types import Bits, Channel, StructType, UInt

from esiaccel.bsp.cosim import CosimBSP

# Narrower than the 64-bit cosim HostMem word (sub-word, non-dividing and
# dividing) and wider than it (non-multiple), so several read gearboxes are
# exercised.
ELEMENT_WIDTHS = [24, 32, 96]

ReqType = StructType([("address", UInt(64)), ("length", UInt(32))])


def HostMemReadDut(width: int):

  # Byte-aligned fields keep the host-side struct decoding simple.
  RespType = StructType([("data", UInt(width)), ("last", UInt(8))])

  class HostMemReadDut(Module):
    module_name = f"HostMemReadDut_{width}"
    clk = Clock()
    rst = Reset()

    @generator
    def build(ports):
      clk, rst = ports.clk, ports.rst

      host_req = esi.ChannelService.from_host(AppID("req"), ReqType)
      burst_req_type = esi.HostMem.read_req_burst_type(32)
      req = host_req.transform(lambda r: burst_req_type({
          "address": r.address,
          "tag": UInt(8)(0),
          "length": r.length,
      }))
      req_xact, _ = req.snoop_xact()
      accepted = Counter(32)(clk=clk,
                             rst=rst,
                             clear=Bits(1)(0),
                             increment=req_xact)
      esi.Telemetry.report_signal(clk, rst, AppID("accepted"), accepted.out)

      gate_in = esi.ChannelService.from_host(AppID("gate"), UInt(8))
      gate_data, gate_valid = gate_in.unwrap(Bits(1)(1))
      gate_open = Reg(Bits(1), clk=clk, rst=rst, rst_value=0, name="gate_open")
      gate_open.assign(Mux(gate_valid, gate_open, gate_data.as_bits()[0]))

      resp = esi.HostMem.read_list(appid=AppID("host"),
                                   req=req,
                                   element_type=Bits(width),
                                   num_items=1)
      resp_ready = Wire(Bits(1))
      frame, resp_valid = resp.unwrap(resp_ready)
      frame = frame.unwrap()
      out, out_ready = Channel(RespType).wrap(
          RespType({
              "data": frame["data"][0].as_uint(),
              "last": frame["last"].as_uint(8),
          }), resp_valid & gate_open)
      resp_ready.assign(out_ready & gate_open)
      esi.ChannelService.to_host(AppID("resp"), out)

  return HostMemReadDut


class Top(Module):
  clk = Clock()
  rst = Reset()

  @generator
  def build(ports):
    for w in ELEMENT_WIDTHS:
      HostMemReadDut(w)(clk=ports.clk,
                        rst=ports.rst,
                        appid=AppID("rd", w),
                        instance_name=f"rd_{w}")


if __name__ == "__main__":
  max_outstanding = int(sys.argv[2])
  s = System(CosimBSP(Top, max_outstanding_reads=max_outstanding),
             name="HostMemRead",
             output_directory=sys.argv[1])
  s.compile()
  s.package()
