#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Hardware for the DRAM service (`esi.Dram`) cosim integration test. For each
# DRAM channel, a DramChannelTest ("dram<N>") exposes host-callable functions
# which issue DRAM operations on that channel:
#
#   write64({address, data: i64, byteenable: i8}) -> ackTag
#       Single-message, byte-enabled write of 8 bytes at a byte address.
#   read64(address) -> i64
#       Single-message read of 8 bytes at a byte address.
#   write128({address, data: i128, byteenable: i16}) -> ackTag
#   read128(address) -> i128
#       Wider-than-DRAM-word variants (exercise the write gearbox chunking).
#   write24({address, data: i24}) -> ackTag
#   read24(address) -> i24
#       Odd-sized (3 byte) elements without a byteenable mask.
#   read_list24({address, length}) -> {count, xor, sum}
#       Burst (list) read of 'length' 3-byte elements; returns a checksum.
#   write_list24({address, count, seed}) -> count
#       Burst (list) write of 'count' 3-byte elements with values seed + i;
#       returns once every element has been acked.

import sys

from pycde import (AppID, Clock, Module, Reset, System, generator, modparams)
from pycde.constructs import ControlReg, Counter, Mux, Reg, Wire
from pycde.types import Bits, Channel, StructType, UInt
from pycde import esi

from esiaccel.bsp import get_bsp

NUM_CHANNELS = 2

ChecksumType = StructType([
    ("count", UInt(16)),
    ("xor", Bits(24)),
    ("sum", UInt(32)),
])


def single_rw(ch: int, width: int, byteenable: bool):
  """Expose a single-message write/read function pair for 'width'-bit data."""
  data_type = Bits(width)
  arg_fields = [("address", UInt(64)), ("data", data_type)]
  if byteenable:
    arg_fields.append(("byteenable", esi.Dram.byteenable_type(data_type)))
  arg_type = StructType(arg_fields)
  req_type = esi.Dram.write_req_channel_type(data_type, byteenable)

  def to_req(a):
    fields = {"address": a.address, "tag": UInt(8)(0), "data": a.data}
    if byteenable:
      fields["byteenable"] = a.byteenable
    return req_type(fields)

  ack = Wire(Channel(UInt(8)))
  wr_args = esi.FuncService.get_call_chans(AppID(f"write{width}"), arg_type,
                                           ack)
  ack.assign(
      esi.Dram.write(AppID(f"dram_write{width}"),
                     wr_args.transform(to_req),
                     channel=ch))

  rd_result = Wire(Channel(data_type))
  rd_args = esi.FuncService.get_call_chans(AppID(f"read{width}"), UInt(64),
                                           rd_result)
  rd_req = rd_args.transform(lambda a: esi.HostMem.ReadReqType({
      "address": a,
      "tag": UInt(8)(0)
  }))
  resp = esi.Dram.read(AppID(f"dram_read{width}"),
                       rd_req,
                       data_type,
                       channel=ch)
  rd_result.assign(resp.transform(lambda r: r.data))


@modparams
def DramChannelTest(ch: int):

  class DramChannelTest(Module):
    clk = Clock()
    rst = Reset()

    @generator
    def construct(ports):
      clk = ports.clk
      rst = ports.rst

      single_rw(ch, 64, byteenable=True)
      single_rw(ch, 128, byteenable=True)
      single_rw(ch, 24, byteenable=False)

      # --- read_list24: burst read with an in-hardware checksum ---
      elem_type = Bits(24)
      list_arg_type = StructType([("address", UInt(64)), ("length", UInt(16))])
      checksum = Wire(Channel(ChecksumType))
      list_args = esi.FuncService.get_call_chans(AppID("read_list24"),
                                                 list_arg_type, checksum)
      burst_req_type = esi.HostMem.read_req_burst_type(16)
      frames = esi.Dram.read_list(AppID("dram_read_list24"),
                                  list_args.transform(lambda a: burst_req_type({
                                      "address": a.address,
                                      "tag": UInt(8)(0),
                                      "length": a.length
                                  })),
                                  elem_type,
                                  num_items=1,
                                  channel=ch)
      result_valid = Wire(Bits(1))
      frames_ready = (~result_valid).as_bits()
      frame_win, frame_valid = frames.unwrap(frames_ready)
      frame = frame_win.unwrap()
      elem = frame["data"][0]
      frame_xact = (frame_valid & frames_ready).as_bits()
      last_xact = (frame_xact & frame["last"]).as_bits()

      result_xact = Wire(Bits(1))
      count = Reg(UInt(16), clk=clk, rst=rst, rst_value=0, name="count")
      xor = Reg(Bits(24), clk=clk, rst=rst, rst_value=0, name="xor")
      total = Reg(UInt(32), clk=clk, rst=rst, rst_value=0, name="sum")
      # Accumulate on each frame; clear once the result has been sent.
      count.assign(
          Mux(result_xact, Mux(frame_xact, count, (count + 1).as_uint(16)),
              UInt(16)(0)))
      xor.assign(Mux(result_xact, Mux(frame_xact, xor, xor ^ elem),
                     Bits(24)(0)))
      total.assign(
          Mux(result_xact,
              Mux(frame_xact, total, (total + elem.as_uint()).as_uint(32)),
              UInt(32)(0)))
      result_valid.assign(
          ControlReg(clk, rst, [last_xact], [result_xact], name="result_valid"))
      result_chan, result_ready = Channel(ChecksumType).wrap(
          ChecksumType({
              "count": count,
              "xor": xor,
              "sum": total
          }), result_valid)
      result_xact.assign((result_valid & result_ready).as_bits())
      checksum.assign(result_chan)

      # --- write_list24: burst write of generated elements ---
      wl_arg_type = StructType([("address", UInt(64)), ("count", UInt(16)),
                                ("seed", UInt(24))])
      wl_result = Wire(Channel(UInt(16)))
      wl_args = esi.FuncService.get_call_chans(AppID("write_list24"),
                                               wl_arg_type, wl_result)
      write_win = esi.Dram.write_window(elem_type, 1)
      lowered = write_win.lowered_type

      busy = Wire(Bits(1))
      streaming = Wire(Bits(1))
      wl_ready = (~busy).as_bits()
      wl, wl_valid = wl_args.unwrap(wl_ready)
      start = (wl_valid & wl_ready).as_bits()
      base = wl.address.reg(clk, rst, ce=start, rst_value=0, name="wl_base")
      wl_count = wl.count.reg(clk, rst, ce=start, rst_value=0, name="wl_count")
      seed = wl.seed.reg(clk, rst, ce=start, rst_value=0, name="wl_seed")

      frame_sent = Wire(Bits(1))
      idx = Counter(16)(clk=clk, rst=rst, clear=start, increment=frame_sent)
      is_last = (idx.out == (wl_count - 1).as_uint(16)).as_bits()
      streaming.assign(
          ControlReg(clk,
                     rst, [start], [(frame_sent & is_last).as_bits()],
                     name="streaming"))
      value = (seed + idx.out.as_uint(24)).as_uint(24).as_bits()
      frame_val = lowered({
          "address": base,
          "tag": UInt(8)(0),
          "data": [value],
          "data_size": Bits(0)(0),
          "last": is_last,
      })
      wframe, wframe_ready = Channel(write_win).wrap(write_win.wrap(frame_val),
                                                     streaming)
      frame_sent.assign((streaming & wframe_ready).as_bits())
      acks = esi.Dram.write(AppID("dram_write_list24"), wframe, channel=ch)
      _, ack_valid = acks.unwrap(Bits(1)(1))
      ack_count = Counter(16)(clk=clk,
                              rst=rst,
                              clear=start,
                              increment=ack_valid)
      done_valid = (busy & ~streaming & (ack_count.out == wl_count)).as_bits()
      done_chan, done_ready = Channel(UInt(16)).wrap(wl_count, done_valid)
      busy.assign(
          ControlReg(clk,
                     rst, [start], [(done_valid & done_ready).as_bits()],
                     name="busy"))
      wl_result.assign(done_chan)

  return DramChannelTest


class Top(Module):
  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    for ch in range(NUM_CHANNELS):
      DramChannelTest(ch)(clk=ports.clk,
                          rst=ports.rst,
                          appid=AppID(f"dram{ch}"))


if __name__ == "__main__":
  bsp = get_bsp(sys.argv[2] if len(sys.argv) > 2 else None)
  s = System(bsp(Top), name="DramTest", output_directory=sys.argv[1])
  s.compile()
  s.package()
