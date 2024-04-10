#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..common import Clock, Input, Output
from ..constructs import ControlReg, Mux, NamedWire, Wire
from .. import esi
from ..module import Module, generator
from ..signals import BitsSignal, BundleSignal, ChannelSignal
from ..types import Array, Bits, Bundle, Channel, UInt

from typing import Dict, Tuple

MagicNumberLo = 0xE5100E51  # ESI__ESI
MagicNumberHi = 0x207D98E5  # Random
VersionNumber = 0  # Version 0: format subject to change


class ESI_Manifest_ROM(Module):
  """Module which will be created later by CIRCT which will contain the
  compressed manifest."""

  module_name = "__ESI_Manifest_ROM"

  clk = Clock()
  address = Input(Bits(30))
  # Data is two cycles delayed after address changes.
  data = Output(Bits(32))


class ESI_Manifest_ROM_Wrapper(Module):
  """Wrap the manifest ROM with ESI bundle."""

  clk = Clock()
  read = Input(esi.MMIO.read.type)

  @generator
  def build(self):
    data, data_valid = Wire(Bits(32)), Wire(Bits(1))
    data_chan, data_ready = Channel(Bits(32)).wrap(data, data_valid)
    address_chan = self.read.unpack(data=data_chan)['offset']
    address, address_valid = address_chan.unwrap(data_ready)
    address_words = address.as_bits(32)[2:]  # Lop off the lower two bits.

    rom = ESI_Manifest_ROM(clk=self.clk, address=address_words)
    data.assign(rom.data)
    data_valid.assign(address_valid.reg(self.clk, name="data_valid", cycles=2))


class AxiMMIO(esi.ServiceImplementation):
  """MMIO service implementation with an AXI-lite protocol. This assumes a 20
  bit address bus for 1MB of addressable MMIO space. Which should be fine for
  now, though nothing should assume this limit. It also only supports 32-bit
  aligned accesses and just throws awary the lower two bits of address.

  Only allows for one outstanding request at a time. If a client doesn't return
  a response, the MMIO service will hang. TODO: add some kind of timeout.

  Implementation-defined MMIO layout:
    - 0x0: 0 constanst
    - 0x4: 0 constanst
    - 0x8: Magic number low (0xE5100E51)
    - 0xC: Magic number high (random constant: 0x207D98E5)
    - 0x10: ESI version number (0)
    - 0x14: Location of the manifest ROM (absolute address)

    - 0x100: Start of MMIO space for requests. Mapping is contained in the
             manifest so can be dynamically queried.

    - addr(Manifest ROM) + 0: Size of compressed manifest
    - addr(Manifest ROM) + 4: Start of compressed manifest

  This layout _should_ be pretty standard, but different BSPs may have various
  different restrictions.
  """

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

  # Amount of register space each client gets. This is a GIANT HACK and needs to
  # be replaced by parameterizable services.
  RegisterSpace = 1024
  RegisterSpaceBits = RegisterSpace.bit_length()

  # Start at this address for assigning MMIO addresses to service requests.
  initial_offset: int = RegisterSpace

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
      if bundle.port == 'read':
        read_table[offset] = bundle
        offset += AxiMMIO.RegisterSpace
      else:
        assert False, "Unrecognized port name."
      # elif bundle.direction == ChannelDirection.Output:
      #   write_table[offset] = bundle
      #   offset += AxiMMIO.RegisterSpace

    manifest_loc = 1 << offset.bit_length()
    return read_table, write_table, manifest_loc

  def build_araddr_chan(self,
                        table_entries: int) -> Tuple[BitsSignal, ChannelSignal]:
    """Build a channel for the address read request. Returns the registered raw
    address and a channel for the masked address to be passed to the clients."""
    i1 = Bits(1)

    # Capture the address if a the bus transaction occured.
    address_written = NamedWire(i1, "address_written")
    address = self.araddr.reg(self.clk, ce=address_written, name="address")
    address_masked = address.as_bits() & Bits(20)(
        1 << table_entries.bit_length())
    address_valid = address_written.reg(name="address_valid")
    addr_ui32 = BitsSignal.concat([Bits(12)(0), address_masked]).as_uint()
    address_chan, address_chan_ready = Channel(UInt(32)).wrap(
        addr_ui32, address_valid)
    self.arready = address_chan_ready
    address_written.assign(self.arvalid & ~address_chan_ready)

    return address, address_chan

  def connect_rdata(self, data_chan: ChannelSignal):
    i32 = Bits(32)
    i2 = Bits(2)
    i1 = Bits(1)

    # Set up the output of the data response pipeline. `data_pipeline*` are to
    # be connected below.
    response_written = NamedWire(i1, "response_written")
    data_pipeline, data_pipeline_valid = data_chan.unwrap(self.arready)
    data_pipeline = NamedWire(i32, "data_pipeline")
    data_out_valid = ControlReg(self.clk,
                                self.rst, [data_pipeline_valid],
                                [response_written],
                                name="data_out_valid")
    self.rvalid = data_out_valid
    self.rdata = data_pipeline.reg(self.clk,
                                   self.rst,
                                   ce=data_pipeline_valid,
                                   name="data_pipeline_reg")
    self.rresp = i2(0)

    # Clear the `req_outstanding` flag when the response has been transmitted.
    response_written.assign(data_out_valid & self.rready)

  def build_header_bundle(self, mmio_req: BundleSignal, manifest_loc: int):
    data_chan_wire = Wire(Channel(Bits(32)))
    input_bundles = mmio_req.unpack(data=data_chan_wire)
    address_chan = input_bundles['offset']

    address_ready = Wire(Bits(1))
    address, address_valid = address_chan.unwrap(address_ready)
    address_words = address.as_bits()[2:]  # Lop off the lower two bits.

    # Handle reads from the header (< 0x100).
    header_upper = address_words[AxiMMIO.initial_offset.bit_length() - 2:]
    # Is the address in the header?
    header_sel = (header_upper == header_upper.type(0))
    header_sel.name = "header_sel"
    # Layout the header as an array.
    header = Array(Bits(32), 6)(
        [0, 0, MagicNumberLo, MagicNumberHi, VersionNumber, manifest_loc])
    header.name = "header"
    header_response_valid = address_valid  # Zero latency read.
    header_out = header[address_words[:3]]
    header_out.name = "header_out"
    data_chan, data_chan_ready = Channel(Bits(32)).wrap(header_out,
                                                        header_response_valid)
    data_chan_wire.assign(data_chan)
    address_ready.assign(data_chan_ready)

  def build_read(self, manifest_loc: int, read_table: Dict[int, Wire]):
    """Builds the read side of the MMIO service."""

    header_bundle_wire = Wire(esi.MMIO.read.type)
    read_table[0] = header_bundle_wire
    AxiMMIO.build_header_bundle(self, header_bundle_wire, manifest_loc)

    mani_bundle_wire = Wire(esi.MMIO.read.type)
    read_table[manifest_loc] = mani_bundle_wire
    ESI_Manifest_ROM_Wrapper(clk=self.clk, read=mani_bundle_wire)

    address, address_chan = AxiMMIO.build_araddr_chan(self, len(read_table))

    demux_mod = esi.PipelinedDemux(Channel(UInt(32)), len(read_table))
    sel_bits = demux_mod.sel.type.width
    # TODO: addresses with non zero upper bits should go to the manifest ROM.
    client_sel = address[AxiMMIO.RegisterSpaceBits:AxiMMIO.RegisterSpaceBits +
                         sel_bits]

    client_address_channels = []
    client_data_channels = []
    for offset in sorted(read_table.keys()):
      bundle_wire = read_table[offset]
      address_chan_wire = Wire(Channel(UInt(32)))
      client_address_channels.append(address_chan_wire)
      bundle, bundle_froms = bundle_wire.type.pack(offset=address_chan_wire)
      client_data_channels.append(bundle_froms['data'])
      bundle_wire.assign(bundle)

    demux = demux_mod(clk=self.clk,
                      rst=self.rst,
                      sel=client_sel,
                      input=address_chan)
    for i in range(len(read_table)):
      client_address_channels[i].assign(demux.output_channels[i])

    data_channel_type = client_data_channels[0].type
    data_mux = esi.ChannelMux(data_channel_type, len(client_data_channels))(
        clk=self.clk, rst=self.rst, input_channels=client_data_channels)
    self.rdata, self.rvalid = data_mux.output_channel.unwrap(self.rready)
    self.rresp = 0

  def build_write(self, bundles):
    # TODO: this.

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
