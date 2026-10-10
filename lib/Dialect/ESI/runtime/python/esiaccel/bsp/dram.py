#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Service implementation for the ESI standard DRAM service (`esi.Dram`).

`ChannelDram` multiplexes any number of byte-addressed `esi.Dram` clients onto
one or more independent DRAM channels. Each channel is exposed as a pair of
generic, word-addressed ESI channel bundles which map almost 1:1 onto an
Avalon-MM memory interface:

  read{N}:  req  (to memory)   {address: UInt(addr_width),   // word address
                                 burstcount: UInt(burst_width),
                                 tag: UInt(8)}
            resp (from memory) {tag: UInt(8), data: Bits(data_width),
                                 last: Bits(1)}
  write{N}: req  (to memory)   {address: UInt(addr_width),   // word address
                                 burstcount: UInt(burst_width),
                                 tag: UInt(8),
                                 data: Bits(data_width),
                                 byteenable: Bits(data_width / 8),
                                 last: Bits(1)}
            ackTag (from memory) UInt(8)   // one per write burst

Suggested Avalon-MM mapping (the conversion itself is left to the user):
  - `read` / `write` = request valid; request ready = !`waitrequest`.
  - `address`, `burstcount`, `writedata` and `byteenable` come straight from
    the request fields.
  - Read response valid = `readdatavalid`, data = `readdata`. Avalon returns
    read data in order without a tag, so keep a FIFO of {tag, burstcount} per
    accepted read to regenerate 'tag' and 'last'.
  - Avalon-MM has no write response by default, so an ack can be generated
    when the final beat of a write burst is accepted.

Requirements on the memory side: read responses and write acks must be
returned in request order (per channel), which Avalon-MM guarantees. The
byte-to-word adapters in this file rely on that ordering.

This implementation currently issues single-beat write bursts (burstcount=1,
last=1) and read bursts of up to `2**(burst_width-1)` words (the largest
Avalon-MM burst representable in `burst_width` bits).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional

from pycde.common import Clock, Input, InputChannel, Output, OutputChannel, Reset
from pycde.constructs import ControlReg, Mux, Wire
from pycde.dialects import comb
from pycde import esi
from pycde.module import Module, generator, modparams
from pycde.seq import FIFO
from pycde.signals import BitsSignal, BundleSignal, ClockSignal
from pycde.support import clog2
from pycde.types import (Bits, Bundle, BundledChannel, Channel,
                         ChannelDirection, StructType, UInt)

from .common import (DEFAULT_MAX_WRITE_PAYLOAD_BYTES, HostmemReadProcessor,
                     HostMemWriteProcessor)

TagType = UInt(8)


def dram_read_req_type(addr_width: int, burst_width: int) -> StructType:
  return StructType([
      ("address", UInt(addr_width)),
      ("burstcount", UInt(burst_width)),
      ("tag", TagType),
  ])


def dram_read_resp_type(data_width: int) -> StructType:
  return StructType([
      ("tag", TagType),
      ("data", Bits(data_width)),
      ("last", Bits(1)),
  ])


def dram_write_req_type(data_width: int, addr_width: int,
                        burst_width: int) -> StructType:
  return StructType([
      ("address", UInt(addr_width)),
      ("burstcount", UInt(burst_width)),
      ("tag", TagType),
      ("data", Bits(data_width)),
      ("byteenable", Bits(data_width // 8)),
      ("last", Bits(1)),
  ])


def dram_read_bundle_type(data_width: int, addr_width: int,
                          burst_width: int) -> Bundle:
  return Bundle([
      BundledChannel("req", ChannelDirection.TO,
                     dram_read_req_type(addr_width, burst_width)),
      BundledChannel("resp", ChannelDirection.FROM,
                     dram_read_resp_type(data_width)),
  ])


def dram_write_bundle_type(data_width: int, addr_width: int,
                           burst_width: int) -> Bundle:
  return Bundle([
      BundledChannel("req", ChannelDirection.TO,
                     dram_write_req_type(data_width, addr_width, burst_width)),
      BundledChannel("ackTag", ChannelDirection.FROM, TagType),
  ])


def _byte_addr_ports(data_width: int) -> SimpleNamespace:
  """The byte-addressed upstream interface produced by the shared HostMem
  read/write processors, in the shape those processors expect of their
  'hostmem_module' argument."""
  read_req = StructType([
      ("address", UInt(64)),
      ("length", UInt(32)),  # In bytes.
      ("tag", TagType),
  ])
  write_req = StructType([
      ("address", UInt(64)),
      ("tag", TagType),
      ("data", Bits(data_width)),
      ("byteenable", Bits(data_width // 8)),
      ("last", Bits(1)),
  ])
  return SimpleNamespace(
      UpstreamReadReq=read_req,
      read=SimpleNamespace(type=Bundle([
          BundledChannel("req", ChannelDirection.TO, read_req),
          BundledChannel("resp", ChannelDirection.FROM,
                         dram_read_resp_type(data_width)),
      ])),
      UpstreamWriteReq=write_req,
      write=SimpleNamespace(type=Bundle([
          BundledChannel("req", ChannelDirection.TO, write_req),
          BundledChannel("ackTag", ChannelDirection.FROM, TagType),
      ])),
  )


def _check_data_width(data_width: int):
  word_bytes = data_width // 8
  if data_width % 8 != 0 or word_bytes < 2 or \
      (word_bytes & (word_bytes - 1)) != 0:
    raise ValueError("DRAM data width must be a power-of-two number of bytes "
                     f">= 2, got {data_width} bits.")


def _split_byte_address(address: BitsSignal, word_bytes: int, addr_width: int):
  """Split a 64-bit byte address into (byte offset within word, word address)."""
  off_bits = clog2(word_bytes)
  addr_bits = address.as_bits()
  offset = addr_bits[:off_bits]
  word_addr = addr_bits[off_bits:].as_uint(addr_width)
  return offset, word_addr


def _shl(value: BitsSignal, amount: BitsSignal) -> BitsSignal:
  """Logical left shift of 'value' by the (unsigned) signal 'amount'."""
  width = value.type.width
  return comb.ShlOp(value, amount.pad_or_truncate(width)).as_bits()


@modparams
def DramReadAligner(data_width: int,
                    addr_width: int,
                    burst_width: int,
                    max_outstanding: int = 32):
  """Convert byte-addressed read requests {address, length (bytes), tag} into
  word-addressed DRAM bursts {address, burstcount, tag} and realign the
  word-granular DRAM response stream so that the first response word starts at
  the requested byte address (exactly what a host memory read returns).

  An unaligned request fetches one additional word; the response stream is
  realigned with a funnel shift over consecutive DRAM words and the extra word
  is dropped. Up to 'max_outstanding' requests may be in flight. Requires DRAM
  responses in request order."""

  _check_data_width(data_width)
  word_bytes = data_width // 8
  off_bits = clog2(word_bytes)
  ports_ns = _byte_addr_ports(data_width)
  byte_req_type = ports_ns.UpstreamReadReq
  resp_type = dram_read_resp_type(data_width)
  dram_req_type = dram_read_req_type(addr_width, burst_width)

  class DramReadAlignerImpl(Module):
    clk = Clock()
    rst = Reset()
    byte_req = InputChannel(byte_req_type)
    byte_resp = OutputChannel(resp_type)
    dram_req = OutputChannel(dram_req_type)
    dram_resp = InputChannel(resp_type)

    @generator
    def build(ports):
      clk = ports.clk
      rst = ports.rst

      # --- Request path ---
      offsets = FIFO(Bits(off_bits), max_outstanding, clk, rst)
      req_ready = Wire(Bits(1))
      req, req_valid = ports.byte_req.unwrap(req_ready)
      offset, word_addr = _split_byte_address(req.address, word_bytes,
                                              addr_width)
      # Number of DRAM words = ceil((offset + length) / word_bytes).
      span = (req.length.as_uint(33) + offset.as_uint(33) +
              UInt(33)(word_bytes - 1))
      burstcount = span.as_bits()[off_bits:].as_uint(burst_width)
      dram_valid = (req_valid & ~offsets.full).as_bits()
      dram_req, dram_req_ready = Channel(dram_req_type).wrap(
          dram_req_type({
              "address": word_addr,
              "burstcount": burstcount,
              "tag": req.tag,
          }), dram_valid)
      ports.dram_req = dram_req
      req_ready.assign((dram_req_ready & ~offsets.full).as_bits())
      offsets.push(offset, (dram_valid & dram_req_ready).as_bits())

      # --- Response path ---
      resp_ready = Wire(Bits(1))
      resp, resp_valid = ports.dram_resp.unwrap(resp_ready)
      resp_xact = (resp_valid & resp_ready).as_bits()
      cur_offset = offsets.pop((resp_xact & resp.last).as_bits())
      aligned = cur_offset == Bits(off_bits)(0)

      # For an unaligned burst, the first DRAM word is absorbed into
      # 'prev_word' and each subsequent word produces one output word made of
      # the top bytes of the previous word and the bottom bytes of this one.
      have_prev = Wire(Bits(1))
      absorb = (~aligned & ~have_prev).as_bits()
      prev_word = resp.data.reg(clk, ce=resp_xact, name="prev_word")
      joined = BitsSignal.concat([resp.data, prev_word])
      shift_bits = BitsSignal.concat([cur_offset, Bits(3)(0)])
      shifted = joined.slice(shift_bits.pad_or_truncate(clog2(2 * data_width)),
                             data_width)
      out_data = Mux(aligned, shifted, resp.data)
      out_valid = (resp_valid & ~absorb).as_bits()
      out, out_ready = Channel(resp_type).wrap(
          resp_type({
              "tag": resp.tag,
              "data": out_data,
              "last": resp.last,
          }), out_valid)
      ports.byte_resp = out
      # While absorbing we are not forwarding a beat, so do not wait on the
      # downstream ready.
      resp_ready.assign((absorb | out_ready).as_bits())
      have_prev.assign(
          ControlReg(clk,
                     rst, [(resp_xact & absorb & ~resp.last).as_bits()],
                     [(resp_xact & resp.last).as_bits()],
                     name="have_prev"))

  return DramReadAlignerImpl


@modparams
def DramWriteAligner(data_width: int,
                     addr_width: int,
                     burst_width: int,
                     max_outstanding: int = 32):
  """Convert byte-addressed, byte-enabled write words {address, tag, data,
  byteenable, last} into word-addressed DRAM write beats. The data and mask are
  rotated to the byte offset of the address; if the enabled bytes straddle a
  word boundary the write is split into two single-beat bursts. The resulting
  DRAM acks are coalesced so that exactly one ack is returned per input word.
  Up to 'max_outstanding' DRAM writes may be un-acked. Requires DRAM acks in
  request order."""

  _check_data_width(data_width)
  word_bytes = data_width // 8
  ports_ns = _byte_addr_ports(data_width)
  byte_req_type = ports_ns.UpstreamWriteReq
  dram_req_type = dram_write_req_type(data_width, addr_width, burst_width)

  class DramWriteAlignerImpl(Module):
    clk = Clock()
    rst = Reset()
    byte_req = InputChannel(byte_req_type)
    byte_ack = OutputChannel(TagType)
    dram_req = OutputChannel(dram_req_type)
    dram_ack = InputChannel(TagType)

    @generator
    def build(ports):
      clk = ports.clk
      rst = ports.rst

      # One entry per issued DRAM write: set if its ack should be swallowed
      # (the first half of a split write).
      swallow_fifo = FIFO(Bits(1), max_outstanding, clk, rst)

      in_ready = Wire(Bits(1))
      req, in_valid = ports.byte_req.unwrap(in_ready)
      offset, word_addr = _split_byte_address(req.address, word_bytes,
                                              addr_width)
      shift_bits = BitsSignal.concat([offset, Bits(3)(0)])
      data2 = _shl(req.data.pad_or_truncate(2 * data_width), shift_bits)
      be2 = _shl(req.byteenable.pad_or_truncate(2 * word_bytes), offset)
      lo_data = data2[:data_width]
      hi_data = data2[data_width:]
      lo_be = be2[:word_bytes]
      hi_be = be2[word_bytes:]
      need_hi = hi_be.or_reduce()
      # Always issue at least one beat (even for an all-zero mask) so that the
      # client gets its ack.
      need_lo = (lo_be.or_reduce() | ~need_hi).as_bits()
      two_beats = (need_lo & need_hi).as_bits()

      second = Wire(Bits(1))
      use_hi = (second | ~need_lo).as_bits()
      final_beat = (second | ~two_beats).as_bits()
      next_word_addr = (word_addr + UInt(1)(1)).as_uint(addr_width)

      out_valid = (in_valid & ~swallow_fifo.full).as_bits()
      out, out_ready = Channel(dram_req_type).wrap(
          dram_req_type({
              "address": Mux(use_hi, word_addr, next_word_addr),
              "burstcount": UInt(burst_width)(1),
              "tag": req.tag,
              "data": Mux(use_hi, lo_data, hi_data),
              "byteenable": Mux(use_hi, lo_be, hi_be),
              "last": Bits(1)(1),
          }), out_valid)
      ports.dram_req = out
      out_xact = (out_valid & out_ready).as_bits()
      in_ready.assign((out_ready & ~swallow_fifo.full & final_beat).as_bits())
      second.assign(
          ControlReg(clk,
                     rst, [(out_xact & ~final_beat).as_bits()],
                     [(out_xact & final_beat).as_bits()],
                     name="second_beat"))
      swallow_fifo.push((~final_beat).as_bits(), out_xact)

      # --- Ack coalescing ---
      ack_ready = Wire(Bits(1))
      ack, ack_valid = ports.dram_ack.unwrap(ack_ready)
      swallow = swallow_fifo.pop((ack_valid & ack_ready).as_bits())
      ack_out, ack_out_ready = Channel(TagType).wrap(ack, (ack_valid &
                                                           ~swallow).as_bits())
      ports.byte_ack = ack_out
      ack_ready.assign((swallow | ack_out_ready).as_bits())

  return DramWriteAlignerImpl


@modparams
def ChannelDram(data_width: int = 512,
                addr_width: int = 32,
                burst_width: int = 7,
                num_channels: int = 1,
                max_outstanding: int = 32,
                max_read_request_bytes: Optional[int] = None,
                max_write_payload_bytes: int = DEFAULT_MAX_WRITE_PAYLOAD_BYTES):
  """Build a DRAM service implementation (for `esi.Dram`) with 'num_channels'
  independent channels, each exposed as a `read{N}` / `write{N}` bundle pair
  (see the module docstring for the interface and its Avalon-MM mapping).

  Clients select a channel with the `channel` request option (default 0). Each
  channel reuses the HostMem read/write processors (client multiplexing,
  gearboxing, list windows, request splitting) and then converts their
  byte-addressed stream to the word-addressed DRAM interface.

  data_width:   DRAM word width in bits (AVL_DATA_WIDTH). Must be a
                power-of-two number of bytes.
  addr_width:   Word address width (AVL_ADDR_WIDTH).
  burst_width:  Burst count width (AVL_SIZE). Bursts are limited to
                2**(burst_width-1) words.
  max_outstanding: Maximum in-flight DRAM reads / un-acked writes per channel.
  max_read_request_bytes: Largest byte-addressed read chunk. Defaults to the
                largest chunk which (with realignment) fits in one burst.
  """

  _check_data_width(data_width)
  word_bytes = data_width // 8
  if burst_width < 2:
    raise ValueError("burst_width must be >= 2")
  if num_channels < 1:
    raise ValueError("num_channels must be >= 1")
  max_burst_words = 2**(burst_width - 1)
  if max_read_request_bytes is None:
    max_read_request_bytes = (max_burst_words - 1) * word_bytes
  if max_read_request_bytes < word_bytes or \
      max_read_request_bytes // word_bytes + 1 > max_burst_words:
    raise ValueError(
        f"max_read_request_bytes ({max_read_request_bytes}) must be at least "
        f"one word and (in words, plus one for realignment) fit in a "
        f"{max_burst_words}-word burst.")

  byte_ports = _byte_addr_ports(data_width)
  read_bundle = dram_read_bundle_type(data_width, addr_width, burst_width)
  write_bundle = dram_write_bundle_type(data_width, addr_width, burst_width)
  read_resp_type = dram_read_resp_type(data_width)

  class ChannelDramImpl(esi.ServiceImplementation):
    clk = Clock()
    rst = Reset()

    for _ch in range(num_channels):
      locals()[f"read{_ch}"] = Output(read_bundle)
      locals()[f"write{_ch}"] = Output(write_bundle)
    del _ch

    @generator
    def generate(ports, bundles: esi._ServiceGeneratorBundles):
      clk = ports.clk
      rst = ports.rst

      reqs_by_channel: List[List] = [[] for _ in range(num_channels)]
      for req in bundles.to_client_reqs:
        channel = req.options.get("channel", 0)
        if not isinstance(channel, int) or not (0 <= channel < num_channels):
          raise ValueError(
              f"DRAM client '{req.client_name_str}' requested channel "
              f"{channel}, but only {num_channels} channel(s) are available.")
        reqs_by_channel[channel].append(req)

      for ch, reqs in enumerate(reqs_by_channel):
        # Read side: HostMem read processor -> read aligner -> DRAM.
        read_reqs = [r for r in reqs if r.port in ("read", "read_list")]
        read_proc_mod = HostmemReadProcessor(data_width, byte_ports, read_reqs,
                                             max_read_request_bytes)
        read_proc = read_proc_mod(clk=clk,
                                  rst=rst,
                                  instance_name=f"read_proc{ch}")
        for r in read_reqs:
          r.assign(getattr(read_proc, read_proc_mod.reqPortMap[r]))

        byte_resp = Wire(Channel(read_resp_type))
        byte_read_req = read_proc.upstream.unpack(resp=byte_resp)["req"]
        dram_resp = Wire(Channel(read_resp_type))
        read_aligner = DramReadAligner(data_width, addr_width, burst_width,
                                       max_outstanding)(
                                           clk=clk,
                                           rst=rst,
                                           byte_req=byte_read_req,
                                           dram_resp=dram_resp,
                                           instance_name=f"read_aligner{ch}")
        byte_resp.assign(read_aligner.byte_resp)
        dram_read, read_froms = read_bundle.pack(req=read_aligner.dram_req)
        dram_resp.assign(read_froms["resp"])
        setattr(ports, f"read{ch}", dram_read)

        # Write side: HostMem write processor -> write aligner -> DRAM.
        write_reqs = [r for r in reqs if r.port == "write"]
        write_proc_mod = HostMemWriteProcessor(data_width, byte_ports,
                                               write_reqs,
                                               max_write_payload_bytes)
        write_proc = write_proc_mod(clk=clk,
                                    rst=rst,
                                    instance_name=f"write_proc{ch}")
        for r in write_reqs:
          r.assign(getattr(write_proc, write_proc_mod.reqPortMap[r]))

        byte_ack = Wire(Channel(TagType))
        byte_write_req = write_proc.upstream.unpack(ackTag=byte_ack)["req"]
        dram_ack = Wire(Channel(TagType))
        write_aligner = DramWriteAligner(data_width, addr_width, burst_width,
                                         max_outstanding)(
                                             clk=clk,
                                             rst=rst,
                                             byte_req=byte_write_req,
                                             dram_ack=dram_ack,
                                             instance_name=f"write_aligner{ch}")
        byte_ack.assign(write_aligner.byte_ack)
        dram_write, write_froms = write_bundle.pack(req=write_aligner.dram_req)
        dram_ack.assign(write_froms["ackTag"])
        setattr(ports, f"write{ch}", dram_write)

  return ChannelDramImpl


@modparams
def EsiDramModel(DATA_WIDTH: int,
                 ADDR_WIDTH: int,
                 BURST_WIDTH: int,
                 READ_LATENCY: int = 20,
                 WRITE_LATENCY: int = 10,
                 JITTER: int = 8,
                 QUEUE_DEPTH: int = 16,
                 STALL_PERCENT: int = 0,
                 SEED: int = 1):
  """External module declaration for the behavioral cosim DRAM model
  (`EsiDramModel.sv`, shipped with the cosim collateral)."""

  class EsiDramModelImpl(Module):
    module_name = "EsiDramModel"

    clk = Clock()
    rst = Input(Bits(1))

    rd_req_valid = Input(Bits(1))
    rd_req_ready = Output(Bits(1))
    rd_req_address = Input(Bits(ADDR_WIDTH))
    rd_req_burstcount = Input(Bits(BURST_WIDTH))
    rd_req_tag = Input(Bits(8))

    rd_resp_valid = Output(Bits(1))
    rd_resp_ready = Input(Bits(1))
    rd_resp_tag = Output(Bits(8))
    rd_resp_data = Output(Bits(DATA_WIDTH))
    rd_resp_last = Output(Bits(1))

    wr_req_valid = Input(Bits(1))
    wr_req_ready = Output(Bits(1))
    wr_req_address = Input(Bits(ADDR_WIDTH))
    wr_req_burstcount = Input(Bits(BURST_WIDTH))
    wr_req_tag = Input(Bits(8))
    wr_req_data = Input(Bits(DATA_WIDTH))
    wr_req_byteenable = Input(Bits(DATA_WIDTH // 8))
    wr_req_last = Input(Bits(1))

    wr_ack_valid = Output(Bits(1))
    wr_ack_ready = Input(Bits(1))
    wr_ack_tag = Output(Bits(8))

  return EsiDramModelImpl


def connect_dram_model(clk: ClockSignal,
                       rst: BitsSignal,
                       read: BundleSignal,
                       write: BundleSignal,
                       data_width: int,
                       addr_width: int,
                       burst_width: int,
                       instance_name: str = "dram_model",
                       **model_params) -> None:
  """Instantiate the behavioral DRAM model and connect it to one channel's
  `read{N}` / `write{N}` bundles of a `ChannelDram` instance. Extra keyword
  arguments are passed through as `EsiDramModel` parameters."""

  read_resp_type = dram_read_resp_type(data_width)

  rd_resp_ready = Wire(Bits(1))
  wr_ack_ready = Wire(Bits(1))
  rd_resp_wire = Wire(Channel(read_resp_type))
  wr_ack_wire = Wire(Channel(TagType))

  rd_req = read.unpack(resp=rd_resp_wire)["req"]
  wr_req = write.unpack(ackTag=wr_ack_wire)["req"]
  rd_req_ready = Wire(Bits(1))
  wr_req_ready = Wire(Bits(1))
  rd, rd_valid = rd_req.unwrap(rd_req_ready)
  wr, wr_valid = wr_req.unwrap(wr_req_ready)

  model = EsiDramModel(data_width, addr_width, burst_width, **model_params)(
      clk=clk,
      rst=rst,
      rd_req_valid=rd_valid,
      rd_req_address=rd.address.as_bits(),
      rd_req_burstcount=rd.burstcount.as_bits(),
      rd_req_tag=rd.tag.as_bits(),
      rd_resp_ready=rd_resp_ready,
      wr_req_valid=wr_valid,
      wr_req_address=wr.address.as_bits(),
      wr_req_burstcount=wr.burstcount.as_bits(),
      wr_req_tag=wr.tag.as_bits(),
      wr_req_data=wr.data,
      wr_req_byteenable=wr.byteenable,
      wr_req_last=wr.last,
      wr_ack_ready=wr_ack_ready,
      instance_name=instance_name)
  rd_req_ready.assign(model.rd_req_ready)
  wr_req_ready.assign(model.wr_req_ready)

  rd_resp, rd_resp_ready_sig = Channel(read_resp_type).wrap(
      read_resp_type({
          "tag": model.rd_resp_tag.as_uint(),
          "data": model.rd_resp_data,
          "last": model.rd_resp_last,
      }), model.rd_resp_valid)
  rd_resp_ready.assign(rd_resp_ready_sig)
  rd_resp_wire.assign(rd_resp)

  wr_ack, wr_ack_ready_sig = Channel(TagType).wrap(model.wr_ack_tag.as_uint(),
                                                   model.wr_ack_valid)
  wr_ack_ready.assign(wr_ack_ready_sig)
  wr_ack_wire.assign(wr_ack)
