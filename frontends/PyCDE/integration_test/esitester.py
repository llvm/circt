# ===- esitester.py - accelerator for testing ESI functionality -----------===//
#
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//
#
#  This accelerator is intended to eventually grow into a full ESI accelerator
#  test image. It will be used both to test system functionality and system
#  performance. The corresponding software appliciation in the ESI runtime and
#  the ESI cosim. Where this should live longer-term is a unclear.
#
# ===----------------------------------------------------------------------===//

# REQUIRES: esi-runtime, esi-cosim, rtl-sim, esitester
# RUN: rm -rf %t
# RUN: mkdir %t && cd %t
# RUN: %PYTHON% %s %t 2>&1
# RUN: esi-cosim.py --source %t -- esitester cosim env wait | FileCheck %s
# RUN: ESI_COSIM_MANIFEST_MMIO=1 esi-cosim.py --source %t -- esiquery cosim env info
# RUN: esi-cosim.py --source %t -- esitester cosim env dmatest

import pycde
from pycde import AppID, Clock, Module, Reset, generator, modparams
from pycde.bsp import cosim
from pycde.constructs import Counter, Reg, Wire
from pycde.esi import CallService
import pycde.esi as esi
from pycde.types import Bits, Channel, UInt

import sys

# CHECK: [INFO] [CONNECT] connecting to backend


class PrintfExample(Module):
  """Call a printf function on the host once at startup."""

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    # CHECK: [DEBUG] [ESITESTER] Received PrintfExample message
    # CHECK:                     data: 7000
    # CHECK: PrintfExample: 7
    arg_data = UInt(32)(7)

    sent_signal = Wire(Bits(1), "sent_signal")
    sent = Bits(1)(1).reg(ports.clk,
                          ports.rst,
                          ce=sent_signal,
                          rst_value=Bits(1)(0))
    arg_valid = ~sent & ~ports.rst
    arg_chan, arg_ready = Channel(UInt(32)).wrap(arg_data, arg_valid)
    sent_signal.assign(arg_ready & arg_valid)
    CallService.call(AppID("PrintfExample"), arg_chan, Bits(0))


@modparams
def ReadMem(width: int):

  class ReadMem(Module):
    """Module which reads host memory at a certain address as given by writes to
    MMIO register 0x8. Stores the read value and responds to all MMIO reads with
    the stored value."""

    clk = Clock()
    rst = Reset()

    @generator
    def construct(ports):
      cmd_chan_wire = Wire(Channel(esi.MMIOReadWriteCmdType))
      resp_ready_wire = Wire(Bits(1))
      cmd, cmd_valid = cmd_chan_wire.unwrap(resp_ready_wire)
      mmio_xact = cmd_valid & resp_ready_wire

      read_loc_ce = mmio_xact & cmd.write & (cmd.offset == 0x8)
      read_loc = Reg(UInt(64),
                     clk=ports.clk,
                     rst=ports.rst,
                     rst_value=0,
                     ce=read_loc_ce,
                     name="read_loc")
      read_loc.assign(cmd.data.as_uint())

      mem_data_ce = Wire(Bits(1))
      mem_data = Reg(Bits(width),
                     clk=ports.clk,
                     rst=ports.rst,
                     rst_value=0,
                     ce=mem_data_ce,
                     name="mem_data")

      response_data = mem_data.as_bits(64)
      response_chan, response_ready = Channel(Bits(64)).wrap(
          response_data, cmd_valid)
      resp_ready_wire.assign(response_ready)

      mmio_rw = esi.MMIO.read_write(appid=AppID("ReadMem"))
      mmio_rw_cmd_chan = mmio_rw.unpack(data=response_chan)['cmd']
      cmd_chan_wire.assign(mmio_rw_cmd_chan)

      tag = Counter(8)(clk=ports.clk,
                       rst=ports.rst,
                       clear=Bits(1)(0),
                       increment=mmio_xact)

      # Ignoring the ready signal isn't safe, but for cosim it's probably fine.
      hostmem_read_req, hostmem_read_req_ready = Channel(
          esi.HostMem.ReadReqType).wrap({
              "tag": tag.out,
              "address": read_loc
          }, read_loc_ce.reg(ports.clk, ports.rst))

      hostmem_read_resp = esi.HostMem.read(appid=AppID("ReadMem_hostread"),
                                           req=hostmem_read_req,
                                           data_type=mem_data.type)
      hostmem_read_resp_data, hostmem_read_resp_valid = hostmem_read_resp.unwrap(
          1)
      mem_data.assign(hostmem_read_resp_data.data)
      mem_data_ce.assign(hostmem_read_resp_valid)

  return ReadMem


class WriteMem(Module):
  """Writes a cycle count to host memory at address 0 in MMIO upon each MMIO
  transaction."""
  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    cmd_chan_wire = Wire(Channel(esi.MMIOReadWriteCmdType))
    resp_ready_wire = Wire(Bits(1))
    cmd, cmd_valid = cmd_chan_wire.unwrap(resp_ready_wire)
    mmio_xact = cmd_valid & resp_ready_wire

    write_loc_ce = mmio_xact & cmd.write & (cmd.offset == UInt(32)(0))
    write_loc = Reg(UInt(64),
                    clk=ports.clk,
                    rst=ports.rst,
                    rst_value=0,
                    ce=write_loc_ce)
    write_loc.assign(cmd.data.as_uint())

    response_data = write_loc.as_bits()
    response_chan, response_ready = Channel(Bits(64)).wrap(
        response_data, cmd_valid)
    resp_ready_wire.assign(response_ready)

    mmio_rw = esi.MMIO.read_write(appid=AppID("WriteMem"))
    mmio_rw_cmd_chan = mmio_rw.unpack(data=response_chan)['cmd']
    cmd_chan_wire.assign(mmio_rw_cmd_chan)

    tag = Counter(8)(clk=ports.clk,
                     rst=ports.rst,
                     clear=Bits(1)(0),
                     increment=mmio_xact)

    cycle_counter = Counter(64)(clk=ports.clk,
                                rst=ports.rst,
                                clear=Bits(1)(0),
                                increment=Bits(1)(1))

    hostmem_write_req, _ = esi.HostMem.wrap_write_req(
        write_loc,
        cycle_counter.out.as_bits(),
        tag.out,
        valid=mmio_xact.reg(ports.clk, ports.rst))

    hostmem_write_resp = esi.HostMem.write(appid=AppID("WriteMem_hostwrite"),
                                           req=hostmem_write_req)


class EsiTesterTop(Module):
  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    PrintfExample(clk=ports.clk, rst=ports.rst)
    # Once I get read muxing working, enable all three.
    # ReadMem(32)(clk=ports.clk, rst=ports.rst)
    # ReadMem(64)(clk=ports.clk, rst=ports.rst)
    ReadMem(96)(clk=ports.clk, rst=ports.rst)
    WriteMem(clk=ports.clk, rst=ports.rst)


if __name__ == "__main__":
  s = pycde.System(cosim.CosimBSP(EsiTesterTop),
                   name="EsiTester",
                   output_directory=sys.argv[1])
  s.compile()
  s.package()
