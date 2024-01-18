#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from os import name
from requests import head
from ..common import Clock, Input, Output
from ..constructs import ControlReg, Mux, NamedWire, Reg, Wire
from .. import esi
from ..module import Module, generator
from ..signals import ArraySignal, BundleSignal
from ..types import Array, Bits, ChannelDirection, UInt

from typing import Dict, Tuple

MagicNumberLo = 0xE5100E51  # ESI__ESI
MagicNumberHi = 0x207D98E5  # Random
VersionNumber = 0  # Version 0: format subject to change


class ESI_Manifest_ROM(Module):
  module_name = "__ESI_Manifest_ROM"

  clk = Clock()
  address = Input(Bits(30))
  # Data is two cycles delayed after address changes.
  data = Output(Bits(32))


class AxiMMIO(esi.ServiceImplementation):
  clk = Clock()
  rst = Input(Bits(1))

  # MMIO read: address channel.
  arvalid = Input(Bits(1))
  arready = Output(Bits(1))
  araddr = Input(Bits(20))

  # MMIO read: data response channel.
  rvalid = Output(Bits(1))
  rready = Input(Bits(1))
  rdata = Output(Bits(32))
  rresp = Output(Bits(2))

  # MMIO write: address channel.
  awvalid = Input(Bits(1))
  awready = Output(Bits(1))
  awaddr = Input(Bits(20))

  # MMIO write: data channel.
  wvalid = Input(Bits(1))
  wready = Output(Bits(1))
  wdata = Input(Bits(32))

  # MMIO write: write response channel.
  bvalid = Output(Bits(1))
  bready = Input(Bits(1))
  bresp = Output(Bits(2))

  # Start at this address
  initial_offset: int = 0x100

  @generator
  def generate(self, bundles: esi._ServiceGeneratorBundles):
    read_table, write_table, manifest_loc = AxiMMIO.build_table(self, bundles)
    AxiMMIO.build_read(self, manifest_loc, read_table)
    AxiMMIO.build_write(self, write_table)
    return True

  def build_table(
      self,
      bundles) -> Tuple[Dict[int, BundleSignal], Dict[int, BundleSignal], int]:
    """Build a table of read and write addresses to BundleSignals."""
    offset = AxiMMIO.initial_offset
    read_table = {}
    write_table = {}
    for bundle in bundles.to_client_reqs:
      if bundle.direction == ChannelDirection.Input:
        read_table[offset] = bundle
        offset += 4
      elif bundle.direction == ChannelDirection.Output:
        write_table[offset] = bundle
        offset += 4

    manifest_loc = 1 << offset.bit_length()
    return read_table, write_table, manifest_loc

  def build_read(self, manifest_loc: int, bundles):
    i32 = Bits(32)
    i2 = Bits(2)
    i1 = Bits(1)
    addr_width = self.araddr.type.width

    address_written = NamedWire(i1, "address_written")
    response_written = NamedWire(i1, "response_written")

    req_outstanding = ControlReg(self.clk,
                                 self.rst, [address_written],
                                 [response_written],
                                 name="req_outstanding")
    self.arready = ~req_outstanding
    address_written.assign(self.arvalid & ~req_outstanding)
    address = self.araddr.reg(self.clk, ce=address_written, name="address")
    address_valid = address_written.reg(name="address_valid")

    data_pipeline_valid = NamedWire(i1, "data_pipeline_valid")
    data_pipeline = NamedWire(i32, "data_pipeline")
    data_pipeline_rresp = NamedWire(i2, "data_pipeline_rresp")
    data_out_valid = ControlReg(self.clk,
                                self.rst, [data_pipeline_valid],
                                [response_written],
                                name="data_out_valid")
    self.rvalid = data_out_valid
    response_written.assign(data_out_valid & self.rready)
    self.rdata = data_pipeline.reg(self.clk,
                                   self.rst,
                                   ce=data_pipeline_valid,
                                   name="data_pipeline_reg")
    self.rresp = data_pipeline_rresp.reg(self.clk,
                                         self.rst,
                                         ce=data_pipeline_valid,
                                         name="data_pipeline_rresp_reg")

    address_offset_words = address[2:]

    header_upper = NamedWire(
        address_offset_words[AxiMMIO.initial_offset.bit_length() - 2:],
        "header_upper")
    header_sel = (header_upper == header_upper.type(0))
    header_sel.name = "header_sel"
    header = Array(Bits(32), 6)(
        [0, 0, MagicNumberLo, MagicNumberHi, VersionNumber, manifest_loc])
    header.name = "header"
    header_valid = address_valid
    header_line_sel = NamedWire(address[2:5], "header_line_sel")
    header_out = header[header_line_sel]
    header_out.name = "header_out"
    header_rresp = i2(0)

    rom_address = NamedWire(
        (address_offset_words.as_uint() - (manifest_loc >> 2)).as_bits(30),
        "rom_address")
    mani_rom = ESI_Manifest_ROM(clk=self.clk, address=rom_address)
    mani_valid = address_valid.reg(self.clk,
                                   self.rst,
                                   rst_value=i1(0),
                                   cycles=2,
                                   name="mani_valid_reg")
    mani_rresp = i2(0)
    # mani_sel = (address.as_uint() >= manifest_loc)

    data_mux_inputs = [header_out, mani_rom.data]
    data_valid_mux_inputs = [header_valid, mani_valid]
    rresp_mux_inputs = [header_rresp, mani_rresp]

    sel = NamedWire(~header_sel, "sel")
    data_pipeline.assign(Mux(sel, *data_mux_inputs))
    data_pipeline_rresp.assign(Mux(sel, *rresp_mux_inputs))
    data_pipeline_valid.assign(Mux(sel, *data_valid_mux_inputs))

  def build_write(self, bundles):
    # So that we don't wedge the AXI-lite for writes, just ack all of them.
    write_happened = Wire(Bits(1))
    latched_aw = ControlReg(self.clk, self.rst, [self.awvalid],
                            [write_happened])
    latched_w = ControlReg(self.clk, self.rst, [self.wvalid], [write_happened])
    write_happened.assign(latched_aw & latched_w)

    self.awready = 1
    self.wready = 1
    self.bvalid = write_happened
    self.bresp = 0
