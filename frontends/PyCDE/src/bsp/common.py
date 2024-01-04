#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..common import Clock, Input, Output
from ..constructs import ControlReg, NamedWire, Reg, Wire
from .. import esi
from ..module import Module, generator
from ..signals import BundleSignal
from ..types import Bits, ChannelDirection, UInt

from typing import Dict, Tuple


class ESI_Metadata_ROM(Module):
  module_name = "__ESI_Metadata_ROM"

  address = Input(Bits(30))
  data = Output(Bits(32))


class AxiMMIO(esi.ServiceImplementation):
  clk = Clock()
  rst = Input(Bits(1))

  # MMIO read: address channel.
  arvalid = Input(Bits(1))
  arready = Output(Bits(1))
  araddr = Input(UInt(32))

  # MMIO read: data response channel.
  rvalid = Output(Bits(1))
  rready = Input(Bits(1))
  rdata = Output(Bits(32))
  rresp = Output(Bits(2))

  # MMIO write: address channel.
  awvalid = Input(Bits(1))
  awready = Output(Bits(1))
  awaddr = Input(UInt(32))

  # MMIO write: data channel.
  wvalid = Input(Bits(1))
  wready = Output(Bits(1))
  wdata = Input(Bits(32))

  # MMIO write: write response channel.
  bvalid = Output(Bits(1))
  bready = Input(Bits(1))
  bresp = Output(Bits(2))

  # 4 MB of address space for the ESI magic num, ESI version number, and manifest. Should be plenty.
  MetadataSize = 4 * 1024 * 1024
  MetadataMaxAddress = MetadataSize.bit_length()
  ManifestMaxSize = MetadataSize - 256

  @generator
  def generate(self, bundles: esi._ServiceGeneratorBundles):
    read_table, write_table = AxiMMIO.build_table(self, bundles)
    AxiMMIO.build_read(self, read_table)
    AxiMMIO.build_write(self, write_table)
    return True

  def build_table(
      self, bundles) -> Tuple[Dict[int, BundleSignal], Dict[int, BundleSignal]]:
    """Build a table of read and write addresses to BundleSignals."""
    offset: int = AxiMMIO.MetadataMaxAddress
    read_table = {}
    write_table = {}
    for bundle in bundles.to_client_reqs:
      if bundle.direction == ChannelDirection.Input:
        read_table[offset] = bundle
        offset += 4
      elif bundle.direction == ChannelDirection.Output:
        write_table[offset] = bundle
        offset += 4

    return read_table, write_table

  def build_read(self, bundles):
    i32 = Bits(32)
    i1 = Bits(1)

    address_written = Wire(i1)
    response_written = NamedWire(i1, "response_written")

    address_reg_valid = ControlReg(self.clk, self.rst, [address_written],
                                   [response_written])
    address_reg_empty = ~address_reg_valid
    address_reg = self.araddr.reg(self.clk, self.rst, ce=address_reg_empty)
    self.arready = address_reg_empty
    address_written.assign(address_reg_empty & self.arvalid)

    rvalid = Wire(i1)
    response_written.assign(rvalid & self.rready)
    rvalid.assign(address_reg_valid)

    meta_rom = ESI_Metadata_ROM(address=address_reg.as_bits()[2:])

    self.rvalid = response_written
    self.rdata = meta_rom.data
    self.rresp = 0

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
