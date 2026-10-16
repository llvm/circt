#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Service implementation for the ESI standard DRAM service (`esi.Dram`).

Each DRAM channel is its own `esi.Dram[i]` service declaration, implemented by
its own `ChannelDram` instance. A `ChannelDram` multiplexes any number of
byte-addressed clients of its declaration onto one DRAM channel, exposed as a
pair of generic, word-addressed ESI channel bundles which map almost 1:1 onto
an Avalon-MM memory interface:

  read:     req  (to memory)   {address: UInt(addr_width),   // word address
                                 burstcount: UInt(burst_width),
                                 tag: UInt(8)}
            resp (from memory) {tag: UInt(8), data: Bits(data_width),
                                 last: Bits(1)}
  write:    req  (to memory)   {address: UInt(addr_width),   // word address
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
returned in request order, which Avalon-MM guarantees.

Client addresses must be word-aligned; the low-order bits are ignored and data
is never shifted. List elements are laid out at a word-multiple stride. This
implementation issues single-beat write bursts (burstcount=1, last=1) and read
bursts of up to `2**(burst_width-1)` words (the largest Avalon-MM burst
representable in `burst_width` bits).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

from pycde.common import Clock, Input, Output, Reset
from pycde.constructs import Wire
from pycde import esi
from pycde.module import Module, generator, modparams
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


def _word_address(address: BitsSignal, word_bytes: int,
                  addr_width: int) -> BitsSignal:
  """Convert a 64-bit byte address into a word address. Addresses must be
  word-aligned; the low-order (within-word) bits are ignored."""
  return address.as_bits()[clog2(word_bytes):].as_uint(addr_width)


@modparams
def ChannelDram(data_width: int = 512,
                addr_width: int = 32,
                burst_width: int = 7,
                max_read_request_bytes: Optional[int] = None,
                max_write_payload_bytes: int = DEFAULT_MAX_WRITE_PAYLOAD_BYTES,
                max_outstanding_reads: int = 1):
  """Build a service implementation for one DRAM channel (one `esi.Dram[i]`
  declaration), exposed as a `read` / `write` bundle pair (see the module
  docstring for the interface and its Avalon-MM mapping). Instantiate it once
  per channel, passing that channel's declaration as `decl`.

  The HostMem read/write processors (client multiplexing, gearboxing, list
  windows, request splitting) handle the clients, with list elements laid out
  at a word-multiple stride. Their word-aligned byte addresses are then
  converted to word addresses.

  data_width:   DRAM word width in bits (AVL_DATA_WIDTH). Must be a
                power-of-two number of bytes.
  addr_width:   Word address width (AVL_ADDR_WIDTH).
  burst_width:  Burst count width (AVL_SIZE). Bursts are limited to
                2**(burst_width-1) words.
  max_read_request_bytes: Largest read burst, in bytes. Defaults to the
                largest burst representable in 'burst_width' bits.
  max_outstanding_reads: Per-client limit on logical read requests in flight
                (see `HostmemReadProcessor`). DRAM returns read responses in
                request order, so values above 1 are safe.
  """

  _check_data_width(data_width)
  word_bytes = data_width // 8
  word_shift = clog2(word_bytes)
  if burst_width < 2:
    raise ValueError("burst_width must be >= 2")
  max_burst_words = 2**(burst_width - 1)
  if max_read_request_bytes is None:
    max_read_request_bytes = max_burst_words * word_bytes
  if not (word_bytes <= max_read_request_bytes <= max_burst_words * word_bytes):
    raise ValueError(
        f"max_read_request_bytes ({max_read_request_bytes}) must be at least "
        f"one word and at most a {max_burst_words}-word burst.")

  byte_ports = _byte_addr_ports(data_width)
  read_bundle = dram_read_bundle_type(data_width, addr_width, burst_width)
  write_bundle = dram_write_bundle_type(data_width, addr_width, burst_width)
  read_req_type = dram_read_req_type(addr_width, burst_width)
  read_resp_type = dram_read_resp_type(data_width)
  write_req_type = dram_write_req_type(data_width, addr_width, burst_width)

  class ChannelDramImpl(esi.ServiceImplementation):
    clk = Clock()
    rst = Reset()

    read = Output(read_bundle)
    write = Output(write_bundle)

    @generator
    def generate(ports, bundles: esi._ServiceGeneratorBundles):
      clk = ports.clk
      rst = ports.rst
      reqs = bundles.to_client_reqs

      # Read side: the HostMem read processor issues word-aligned byte-addressed
      # requests whose lengths are whole words. Convert them to word bursts;
      # the in-order responses pass straight through.
      read_reqs = [r for r in reqs if r.port in ("read", "read_list")]
      read_proc_mod = HostmemReadProcessor(
          data_width,
          byte_ports,
          read_reqs,
          max_read_request_bytes,
          max_outstanding_reads=max_outstanding_reads,
          element_alignment=word_bytes)
      read_proc = read_proc_mod(clk=clk, rst=rst, instance_name="read_proc")
      for r in read_reqs:
        r.assign(getattr(read_proc, read_proc_mod.reqPortMap[r]))

      dram_resp = Wire(Channel(read_resp_type))
      byte_read_req = read_proc.upstream.unpack(resp=dram_resp)["req"]
      dram_read_req = byte_read_req.transform(lambda r: read_req_type({
          "address": _word_address(r.address, word_bytes, addr_width),
          "burstcount": ((r.length + UInt(32)(word_bytes - 1)).as_bits()[
              word_shift:].as_uint(burst_width)),
          "tag": r.tag,
      }))
      dram_read, read_froms = read_bundle.pack(req=dram_read_req)
      dram_resp.assign(read_froms["resp"])
      ports.read = dram_read

      # Write side: the HostMem write processor issues one word-aligned,
      # byte-enabled word per message. Each becomes a single-beat burst, and
      # the acks (one per burst) pass straight through.
      write_reqs = [r for r in reqs if r.port == "write"]
      write_proc_mod = HostMemWriteProcessor(data_width,
                                             byte_ports,
                                             write_reqs,
                                             max_write_payload_bytes,
                                             element_alignment=word_bytes)
      write_proc = write_proc_mod(clk=clk, rst=rst, instance_name="write_proc")
      for r in write_reqs:
        r.assign(getattr(write_proc, write_proc_mod.reqPortMap[r]))

      dram_ack = Wire(Channel(TagType))
      byte_write_req = write_proc.upstream.unpack(ackTag=dram_ack)["req"]
      dram_write_req = byte_write_req.transform(lambda w: write_req_type({
          "address": _word_address(w.address, word_bytes, addr_width),
          "burstcount": UInt(burst_width)(1),
          "tag": w.tag,
          "data": w.data,
          "byteenable": w.byteenable,
          "last": Bits(1)(1),
      }))
      dram_write, write_froms = write_bundle.pack(req=dram_write_req)
      dram_ack.assign(write_froms["ackTag"])
      ports.write = dram_write

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
  """Instantiate the behavioral DRAM model and connect it to the `read` /
  `write` bundles of a `ChannelDram` instance. Extra keyword arguments are
  passed through as `EsiDramModel` parameters."""

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
