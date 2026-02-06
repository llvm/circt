# REQUIRES: esi-cosim, esi-runtime, rtl-sim
# RUN: rm -rf %t
# RUN: mkdir %t && cd %t

# Build the system.
# RUN: %PYTHON% %s %t 2>&1

# Query against the manifest.
# RUN: esiquery trace w:%t/esi_system_manifest.json info | FileCheck %s --check-prefix=QUERY-INFO
# RUN: esiquery trace w:%t/esi_system_manifest.json hier | FileCheck %s --check-prefix=QUERY-HIER

# Run a cosimulation with a python test driver.
# RUN: esi-cosim.py --source %t -- %PYTHON% %S/test_software/loopback.py cosim env

# Test C++ header generation against the manifest file
# RUN: %PYTHON% -m esiaccel.codegen --file %t/esi_system_manifest.json --output-dir %t/include/loopback/
# RUN: %cxx -I %t/include %S/test_software/loopback.cpp -o %t/test
# RUN: %t/test | FileCheck %s --check-prefix=CPP-TEST
# RUN: FileCheck %s --check-prefix=LOOPBACK-H --input-file %t/include/loopback/LoopbackIP.h

# Test C++ header generation from the live accelerator.
# RUN: esi-cosim.py --source %t -- %PYTHON% -m esiaccel.codegen --platform cosim --connection env --output-dir %t/include/loopback/
# RUN: %cxx -I %t/include %S/test_software/loopback.cpp -o %t/test
# RUN: %t/test | FileCheck %s --check-prefix=CPP-TEST

import sys

import pycde
from pycde import (AppID, Clock, Module, Reset, generator)
from pycde.bsp import get_bsp
from pycde.common import Constant
from pycde.constructs import Wire
from pycde.module import Metadata
from pycde.types import (Bits, Bundle, BundledChannel, Channel,
                         ChannelDirection, SInt, StructType, UInt)
from pycde import esi

SendI8 = Bundle([BundledChannel("send", ChannelDirection.FROM, Bits(8))])
RecvI8 = Bundle([BundledChannel("recv", ChannelDirection.TO, Bits(8))])
SendI0 = Bundle([BundledChannel("send", ChannelDirection.FROM, Bits(0))])
RecvI0 = Bundle([BundledChannel("recv", ChannelDirection.TO, Bits(0))])


@esi.ServiceDecl
class HostComms:
  Send = SendI8
  Recv = RecvI8


@esi.ServiceDecl
class MyService:
  Send = SendI0
  Recv = RecvI0


class Loopback(Module):
  clk = Clock()

  metadata = Metadata(
      name="LoopbackIP",
      version="v0.0",
      summary="IP which simply echos bytes",
      misc={"foo": 1},
  )

  depth = Constant(UInt(32), 5)

  @generator
  def construct(ports):
    data_in_bundle = HostComms.Recv(AppID("loopback_tohw"))
    data_in = data_in_bundle.unpack()["recv"]

    data_out_bundle = HostComms.Send(AppID("loopback_fromhw"))
    data_out_bundle.unpack(send=data_in)

    send_bundle = MyService.Recv(AppID("mysvc_recv"))
    send_chan = send_bundle.unpack()["recv"]
    sendi0_bundle = MyService.Send(AppID("mysvc_send"))
    sendi0_bundle.unpack(send=send_chan)


ArgStruct = StructType({"a": UInt(16), "b": SInt(8)})
ResultStruct = StructType({"x": SInt(8), "y": SInt(8)})


class LoopbackStruct(Module):

  @generator
  def construct(ports):
    result_wire = Wire(Channel(ResultStruct))
    args = esi.FuncService.get_call_chans(AppID("structFunc"),
                                          arg_type=ArgStruct,
                                          result=result_wire)

    ready = Wire(Bits(1))
    arg_data, valid = args.unwrap(ready)
    b_val = arg_data["b"]
    b_plus_one = (b_val + SInt(8)(1)).as_sint(8)
    result = ResultStruct({"x": b_plus_one, "y": b_val})
    result_chan, result_ready = Channel(ResultStruct).wrap(result, valid)
    ready.assign(result_ready)
    result_wire.assign(result_chan)


ArgArray = SInt(8) * 1
ResultArray = SInt(8) * 2


class LoopbackArray(Module):

  @generator
  def construct(ports):
    result_wire = Wire(Channel(ResultArray))
    args = esi.FuncService.get_call_chans(AppID("arrayFunc"),
                                          arg_type=ArgArray,
                                          result=result_wire)

    ready = Wire(Bits(1))
    arg_data, valid = args.unwrap(ready)
    elem = arg_data[0]
    elem_plus_one = (elem + SInt(8)(1)).as_sint(8)
    # ArrayCreate reverses element order; provide reversed list to match MLIR.
    result_array = pycde.Signal.create([elem, elem_plus_one])
    result_chan, result_ready = Channel(ResultArray).wrap(result_array, valid)
    ready.assign(result_ready)
    result_wire.assign(result_chan)


MemA = esi.DeclareRandomAccessMemory(Bits(64), 20, name="MemA")


class MemoryAccess1(Module):
  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    MemA.instantiate_builtin(appid=AppID("mem"),
                             builtin="sv_mem",
                             result_types=[],
                             inputs=[ports.clk, ports.rst])

    write_bundle = MemA.write(AppID("internal_write"))
    write_req_type = MemA.write.type.req
    write_req, _ = write_req_type.wrap(
        {
            "address": UInt(MemA.address_width)(0),
            "data": Bits(64)(0)
        },
        Bits(1)(0))
    write_bundle.unpack(req=write_req)


class CallableFunc1(Module):

  @generator
  def construct(ports):
    result_wire = Wire(Channel(UInt(16)))
    args = esi.FuncService.get_call_chans(AppID("func1"),
                                          arg_type=UInt(16),
                                          result=result_wire)

    ready = Wire(Bits(1))
    _, valid = args.unwrap(ready)
    result_chan, result_ready = Channel(UInt(16)).wrap(UInt(16)(0), valid)
    ready.assign(result_ready)
    result_wire.assign(result_chan)


class Top(Module):
  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    Loopback(clk=ports.clk, appid=AppID("loopback_inst", 0))
    Loopback(clk=ports.clk, appid=AppID("loopback_inst", 1))
    MemoryAccess1(clk=ports.clk, rst=ports.rst)
    CallableFunc1()
    LoopbackStruct()
    LoopbackArray()


if __name__ == "__main__":
  bsp = get_bsp(sys.argv[2] if len(sys.argv) > 2 else None)
  s = pycde.System(bsp(Top), name="Loopback", output_directory=sys.argv[1])
  s.compile()
  s.package()

# CPP-TEST: depth: 0x5

# QUERY-INFO: API version: 0
# QUERY-INFO: ********************************
# QUERY-INFO: * Module information
# QUERY-INFO: ********************************
# QUERY-INFO: - LoopbackIP v0.0
# QUERY-INFO:   IP which simply echos bytes
# QUERY-INFO:   Constants:
# QUERY-INFO:     depth: 5
# QUERY-INFO:   Extra metadata:
# QUERY-INFO:     foo: 1

# QUERY-HIER: ********************************
# QUERY-HIER: * Design hierarchy
# QUERY-HIER: ********************************
# QUERY-HIER: * Instance: {{.*}}
# QUERY-HIER: * Ports:
# QUERY-HIER:     func1: function uint16(uint16)
# QUERY-HIER:     structFunc: function struct {x: sint8, y: sint8}(struct {a: uint16, b: sint8})
# QUERY-HIER:     arrayFunc: function sint8[2](sint8[1])
# QUERY-HIER: * Children:
# QUERY-HIER:   * Instance: loopback_inst[0]
# QUERY-HIER:   * Ports:
# QUERY-HIER:       loopback_tohw:
# QUERY-HIER:         recv: bits8
# QUERY-HIER:       loopback_fromhw:
# QUERY-HIER:         send: bits8
# QUERY-HIER:       mysvc_recv:
# QUERY-HIER:         recv: void
# QUERY-HIER:       mysvc_send:
# QUERY-HIER:         send: void
# QUERY-HIER:   * Instance: loopback_inst[1]
# QUERY-HIER:   * Ports:
# QUERY-HIER:       loopback_tohw:
# QUERY-HIER:         recv: bits8
# QUERY-HIER:       loopback_fromhw:
# QUERY-HIER:         send: bits8
# QUERY-HIER:       mysvc_recv:
# QUERY-HIER:         recv: void
# QUERY-HIER:       mysvc_send:
# QUERY-HIER:         send: void

# LOOPBACK-H:       /// Generated header for esi_system module LoopbackIP.
# LOOPBACK-H-NEXT:  #pragma once
# LOOPBACK-H-NEXT:  #include "types.h"
# LOOPBACK-H-LABEL: namespace esi_system {
# LOOPBACK-H-LABEL: class LoopbackIP {
# LOOPBACK-H-NEXT:  public:
# LOOPBACK-H-NEXT:    static constexpr uint32_t depth = 0x5;
# LOOPBACK-H-NEXT:  };
# LOOPBACK-H-NEXT:  } // namespace esi_system
