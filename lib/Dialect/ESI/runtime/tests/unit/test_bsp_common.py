#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from pycde import Module

from esiaccel.bsp.common import (ChannelEngineService, ChannelMMIO,
                                 HostMemReadReqSplitter, TaggedWriteGearbox)
from esiaccel.bsp.cosim import CosimBSP, CosimBSP_DMA
from esiaccel.bsp.dma import OneItemBuffersFromHost, OneItemBuffersToHost


class _Bundle:

  def __init__(self, port, options=None):
    self.port = port
    self.options = options or {}
    self.records = []

  def add_record(self, details):
    self.records.append(details)


class _Bundles:

  def __init__(self, *bundles):
    self.to_client_reqs = bundles


def test_channel_mmio_allocations():
  default = _Bundle("read")
  custom = _Bundle("read_write", {"size": 0x1203})

  table, manifest_loc = ChannelMMIO.build_table(_Bundles(default, custom))

  assert list(table) == [0x100, 0x2000]
  assert default.records == [{"offset": 0x100, "size": 0x100, "type": "ro"}]
  assert custom.records == [{"offset": 0x2000, "size": 0x2000, "type": "rw"}]
  # The manifest lands directly above the last client, rounded to a power of
  # two so a single prefix test separates it from the client space.
  assert manifest_loc == 0x4000


def test_channel_mmio_manifest_tracks_client_space():
  """The manifest window must not be a fixed reservation: it follows whatever
  the clients actually use so the BAR stays as small as possible."""

  small, = _Bundles(_Bundle("read", {"size": 8})).to_client_reqs
  _, small_loc = ChannelMMIO.build_table(_Bundles(small))
  assert small_loc == 0x200

  big = _Bundle("read", {"size": 1 << 20})
  _, big_loc = ChannelMMIO.build_table(_Bundles(big))
  assert big_loc == 1 << 21


def test_channel_mmio_places_manifest_after_clients():
  table, manifest_loc = ChannelMMIO.build_table(
      _Bundles(_Bundle("read"), _Bundle("read", {"size": 0x1203})))
  last_base = list(table)[-1]
  last_size = table[last_base][0]

  assert manifest_loc >= last_base + last_size
  # A power-of-two base means 'address has a bit set above the clients' is the
  # whole decode: no comparator, and the region can't overflow.
  assert manifest_loc & (manifest_loc - 1) == 0


@pytest.mark.parametrize("size", [True, 1.5, "256", 0, -8])
def test_channel_mmio_rejects_invalid_size(size):
  with pytest.raises(ValueError, match="option 'size'"):
    ChannelMMIO.build_table(_Bundles(_Bundle("read", {"size": size})))


def test_channel_mmio_rejects_address_overflow(monkeypatch):
  monkeypatch.setattr(ChannelMMIO, "initial_offset", 0xFFFF_FFF8)
  with pytest.raises(ValueError, match="exceeds the 32-bit space"):
    ChannelMMIO.build_table(_Bundles(_Bundle("read", {"size": 8})))


def test_cosim_bsp_rejects_conflicting_channel_services():

  class Top(Module):
    pass

  engines = (OneItemBuffersToHost, OneItemBuffersFromHost)
  channel_service = ChannelEngineService(*engines)
  with pytest.raises(ValueError, match="dma_engine_pair and channel_service"):
    CosimBSP(Top, dma_engine_pair=engines, channel_service=channel_service)


def test_channel_engine_service_connects_read_list(tmp_path):
  """An engine with a `hostmem_read_list` port gets a burst-read connection
  to HostMem: the design compiles only if the service wires that port."""
  from pycde import (Clock, Input, InputChannel, OutputChannel, Reset, System,
                     esi, generator)
  from pycde.types import Bits, Bundle, BundledChannel, Channel, ChannelDirection

  word = Bits(64)
  read_list_type = Bundle([
      BundledChannel("req", ChannelDirection.FROM,
                     esi.HostMem.read_req_burst_type(16)),
      BundledChannel("resp", ChannelDirection.TO,
                     esi.HostMem.read_window(word, 1)),
  ])

  def to_host(client_type):

    class ToHost(esi.EngineModule):

      @property
      def TypeName(self):
        return "ReadListTestToHost"

      clk = Clock()
      rst = Reset()
      input_channel = InputChannel(client_type)

      @generator
      def build(ports):
        pass

    return ToHost

  def from_host(client_type):

    class FromHost(esi.EngineModule):

      @property
      def TypeName(self):
        return "ReadListTestFromHost"

      clk = Clock()
      rst = Reset()
      hostmem_read_list = Input(read_list_type)
      output_channel = OutputChannel(client_type)

      @generator
      def build(ports):
        # Read one word at address 0 forever and forward each word.
        req_type = esi.HostMem.read_req_burst_type(16)
        req, _ = Channel(req_type).wrap(
            req_type({
                "address": 0,
                "tag": 0,
                "length": 1
            }),
            Bits(1)(1))
        resp = ports.hostmem_read_list.unpack(req=req)["resp"]
        ports.output_channel = resp.transform(
            lambda w: w.unwrap()["data"][0].bitcast(client_type))

    return FromHost

  class Top(Module):
    clk = Clock()
    rst = Reset()

    @generator
    def build(ports):
      words = esi.ChannelService.from_host(esi.AppID("words"), word)
      esi.ChannelService.to_host(esi.AppID("echo"), words)

  bsp = CosimBSP(Top, channel_service=ChannelEngineService(to_host, from_host))
  System(bsp, output_directory=str(tmp_path)).compile()


@pytest.mark.parametrize("max_outstanding", [1, 2, 64])
@pytest.mark.parametrize("word_bits", [8, 64, 256, 512, 2048, 4096])
@pytest.mark.parametrize("round_read_lengths", [False, True])
def test_read_req_splitter_generates(tmp_path, word_bits, max_outstanding,
                                     round_read_lengths):
  """Build and compile `HostMemReadReqSplitter` for a range of word widths
  and outstanding-request limits (no simulation)."""
  from pycde import Clock, Input, Module, Output, Reset, System, generator
  from pycde.types import Bits, Channel, StructType, UInt

  req_t = Channel(
      StructType([("address", UInt(64)), ("length", UInt(32)),
                  ("tag", UInt(8))]))
  resp_t = Channel(
      StructType([("tag", UInt(8)), ("data", Bits(word_bits)),
                  ("last", Bits(1))]))
  splitter = HostMemReadReqSplitter(req_t, resp_t, max(256, word_bits // 8),
                                    max_outstanding, round_read_lengths)

  class Top(Module):
    clk = Clock()
    rst = Reset()
    req_in = Input(req_t)
    resp_in = Input(resp_t)
    req_out = Output(req_t)
    resp_out = Output(splitter.resp_out.type)

    @generator
    def build(ports):
      s = splitter(clk=ports.clk,
                   rst=ports.rst,
                   req_in=ports.req_in,
                   resp_in=ports.resp_in)
      ports.req_out = s.req_out
      ports.resp_out = s.resp_out

  System(Top, output_directory=str(tmp_path)).compile()


def test_read_req_splitter_rejects_zero_outstanding():
  from pycde.types import Bits, Channel, StructType, UInt
  req_t = Channel(
      StructType([("address", UInt(64)), ("length", UInt(32)),
                  ("tag", UInt(8))]))
  resp_t = Channel(
      StructType([("tag", UInt(8)), ("data", Bits(64)), ("last", Bits(1))]))
  with pytest.raises(AssertionError):
    HostMemReadReqSplitter(req_t, resp_t, 256, 0)


@pytest.mark.parametrize("builder", [CosimBSP, CosimBSP_DMA])
@pytest.mark.parametrize("width", [True, 64.0, "64", 0, -8, 1, 4, 7, 24, 72])
def test_cosim_rejects_invalid_hostmem_width(builder, width):

  class Top(Module):
    pass

  with pytest.raises(ValueError, match="power of two and at least 8"):
    builder(Top, hostmem_width=width)


@pytest.mark.parametrize("builder", [CosimBSP, CosimBSP_DMA])
@pytest.mark.parametrize("width", [None, 8, 64, 256, 512, 2048, 4096])
def test_cosim_forwards_hostmem_width(monkeypatch, builder, width):
  import esiaccel.bsp.cosim as cosim

  class Top(Module):
    pass

  channel_hostmem = cosim.ChannelHostMem
  configurations = []

  def capture(**kwargs):
    configurations.append(kwargs)
    return channel_hostmem(**kwargs)

  monkeypatch.setattr(cosim, "ChannelHostMem", capture)
  builder(Top, **({"hostmem_width": width} if width is not None else {}))
  expected = width if width is not None else 64
  assert configurations == [{
      "read_width": expected,
      "write_width": expected,
      "max_read_request_bytes": max(256, expected // 8),
      "max_write_payload_bytes": max(256, expected // 8),
      "max_outstanding_reads": 4,
      "round_read_lengths": False,
  }]


@pytest.mark.parametrize("width", [8, 64, 256, 512, 2048, 4096])
def test_write_gearbox_generates(tmp_path, width):
  from pycde import Clock, Input, Output, Reset, System, generator

  gearbox = TaggedWriteGearbox(width * 3 + 13, width, max(256, width // 8))

  class Top(Module):
    clk = Clock()
    rst = Reset()
    req = Input(gearbox.in_.type)
    resp = Output(gearbox.out.type)

    @generator
    def build(ports):
      ports.resp = gearbox(clk=ports.clk, rst=ports.rst, in_=ports.req).out

  System(Top, output_directory=str(tmp_path)).compile()
