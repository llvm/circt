#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..common import Clock, Input, Output, Reset
from ..constructs import AssignableSignal, NamedWire, Wire
from .. import esi
from ..module import Module, generator, modparams
from ..signals import BitsSignal, ChannelSignal
from ..support import clog2
from ..types import Array, Bits, Channel, UInt

from typing import Dict, Tuple

MagicNumber = 0x207D98E5_E5100E51  # random + ESI__ESI
VersionNumber = 0  # Version 0: format subject to change


class ESI_Manifest_ROM(Module):
  """Module which will be created later by CIRCT which will contain the
  compressed manifest."""

  module_name = "__ESI_Manifest_ROM"

  clk = Clock()
  address = Input(Bits(29))
  # Data is two cycles delayed after address changes.
  data = Output(Bits(64))


class ESI_Manifest_ROM_Wrapper(Module):
  """Wrap the manifest ROM with ESI bundle."""

  clk = Clock()
  read = Input(esi.MMIO.read.type)

  @generator
  def build(self):
    data, data_valid = Wire(Bits(64)), Wire(Bits(1))
    data_chan, data_ready = Channel(Bits(64)).wrap(data, data_valid)
    address_chan = self.read.unpack(data=data_chan)['offset']
    address, address_valid = address_chan.unwrap(data_ready)
    address_words = address.as_bits(32)[3:]  # Lop off the lower three bits.

    rom = ESI_Manifest_ROM(clk=self.clk, address=address_words)
    data.assign(rom.data)
    data_valid.assign(address_valid.reg(self.clk, name="data_valid", cycles=2))


@modparams
def HeaderMMIO(manifest_loc: int) -> Module:

  class HeaderMMIO(Module):
    """Construct the ESI header MMIO adhering to the MMIO layout specified in
    the ChannelMMIO service implementation."""

    clk = Clock()
    rst = Reset()
    read = Input(esi.MMIO.read.type)

    @generator
    def build(ports):
      data_chan_wire = Wire(Channel(esi.MMIODataType))
      input_bundles = ports.read.unpack(data=data_chan_wire)
      address_chan = input_bundles['offset']

      address_ready = Wire(Bits(1))
      address, address_valid = address_chan.unwrap(address_ready)
      address_words = address.as_bits()[3:]  # Lop off the lower three bits.

      # Layout the header as an array.
      header = Array(Bits(64), 4)([0, MagicNumber, VersionNumber, manifest_loc])
      header.name = "header"
      header_response_valid = address_valid  # Zero latency read.
      # Select the approptiate header index.
      header_out = header[address_words[:2]]
      header_out.name = "header_out"
      # Wrap the response.
      data_chan, data_chan_ready = Channel(esi.MMIODataType).wrap(
          header_out, header_response_valid)
      data_chan_wire.assign(data_chan)
      address_ready.assign(data_chan_ready)

  return HeaderMMIO


class ChannelMMIO(esi.ServiceImplementation):
  """MMIO service implementation with MMIO bundle interfaces. Should be
  relatively easy to adapt to physical interfaces by wrapping the wires to
  channels then bundles. Allows the implementation to be shared and (hopefully)
  platform independent.
  
  Whether or not to support unaligned accesses is up to the clients. The header
  and manifest do not support unaligned accesses and throw away the lower three
  bits.

  Only allows for one outstanding request at a time. If a client doesn't return
  a response, the MMIO service will hang. TODO: add some kind of timeout.

  Implementation-defined MMIO layout:
    - 0x0: 0 constant
    - 0x8: Magic number (0x207D98E5_E5100E51)
    - 0x12: ESI version number (0)
    - 0x18: Location of the manifest ROM (absolute address)

    - 0x10000: Start of MMIO space for requests. Mapping is contained in the
               manifest so can be dynamically queried.

    - addr(Manifest ROM) + 0: Size of compressed manifest
    - addr(Manifest ROM) + 8: Start of compressed manifest

  This layout _should_ be pretty standard, but different BSPs may have various
  different restrictions. Any BSP which uses this service implementation will
  have this layout, possibly with an offset or address window.
  """

  clk = Clock()
  rst = Input(Bits(1))

  read = Input(esi.MMIO.read.type)

  # Amount of register space each client gets. This is a GIANT HACK and needs to
  # be replaced by parameterizable services.
  # TODO: make the amount of register space each client gets a parameter.
  # Supporting this will require more address decode logic.
  #
  # TODO: if the compressed manifest is larger than 'RegisterSpace', we won't be
  # allocating enough address space. This should be fixed with the more complex
  # address decode logic mentioned above.
  #
  # TODO: only supports one outstanding transaction at a time. This is NOT
  # enforced or checked! Enforce this.

  RegisterSpace = 0x10000
  RegisterSpaceBits = RegisterSpace.bit_length() - 1
  AddressMask = 0xFFFF

  # Start at this address for assigning MMIO addresses to service requests.
  initial_offset: int = RegisterSpace

  @generator
  def generate(ports, bundles: esi._ServiceGeneratorBundles):
    read_table, write_table, manifest_loc = ChannelMMIO.build_table(
        ports, bundles)
    ChannelMMIO.build_read(ports, manifest_loc, read_table)
    ChannelMMIO.build_write(ports, write_table)
    return True

  @staticmethod
  def build_table(
      ports, bundles
  ) -> Tuple[Dict[int, AssignableSignal], Dict[int, AssignableSignal], int]:
    """Build a table of read and write addresses to BundleSignals."""
    offset = ChannelMMIO.initial_offset
    read_table: Dict[int, AssignableSignal] = {}
    write_table: Dict[int, AssignableSignal] = {}
    for bundle in bundles.to_client_reqs:
      if bundle.port == 'read':
        read_table[offset] = bundle
        bundle.add_record({"offset": offset})
        offset += ChannelMMIO.RegisterSpace
      else:
        assert False, "Unrecognized port name."

    manifest_loc = offset
    return read_table, write_table, manifest_loc

  @staticmethod
  def build_read(ports, manifest_loc: int, read_table: Dict[int,
                                                            AssignableSignal]):
    """Builds the read side of the MMIO service."""

    # Instantiate the header and manifest ROM. Fill in the read_table with
    # bundle wires to be assigned identically to the other MMIO clients.
    header_bundle_wire = Wire(esi.MMIO.read.type)
    read_table[0] = header_bundle_wire
    HeaderMMIO(manifest_loc)(clk=ports.clk,
                             rst=ports.rst,
                             read=header_bundle_wire)

    mani_bundle_wire = Wire(esi.MMIO.read.type)
    read_table[manifest_loc] = mani_bundle_wire
    ESI_Manifest_ROM_Wrapper(clk=ports.clk, read=mani_bundle_wire)

    # Unpack the read bundle.
    data_resp_channel = Wire(Channel(esi.MMIODataType))
    counted_output = Wire(Channel(esi.MMIODataType))
    read_addr_channel = ports.read.unpack(data=counted_output)["offset"]
    counted_output.assign(data_resp_channel)

    # Get the selection index and the address to hand off to the clients.
    sel_bits, client_address_chan = ChannelMMIO.build_addr_read(
        read_addr_channel)

    # Build the demux/mux and assign the results of each appropriately.
    read_clients_clog2 = clog2(len(read_table))
    client_addr_channels = esi.ChannelDemux(
        sel=sel_bits.pad_or_truncate(read_clients_clog2),
        input=client_address_chan,
        num_outs=len(read_table))
    client_data_channels = []
    for (idx, offset) in enumerate(sorted(read_table.keys())):
      bundle, bundle_froms = esi.MMIO.read.type.pack(
          offset=client_addr_channels[idx])
      client_data_channels.append(bundle_froms["data"])
      read_table[offset].assign(bundle)
    resp_channel = esi.ChannelMux(client_data_channels)
    data_resp_channel.assign(resp_channel)

  @staticmethod
  def build_addr_read(
      read_addr_chan: ChannelSignal) -> Tuple[BitsSignal, ChannelSignal]:
    """Build a channel for the address read request. Returns the index to select
    the client and a channel for the masked address to be passed to the
    clients."""

    # Decoding the selection bits is very simple as of now. This might need to
    # change to support more flexibility in addressing. Not clear if what we're
    # doing now it sufficient or not.

    addr_ready_wire = Wire(Bits(1))
    addr, addr_valid = read_addr_chan.unwrap(addr_ready_wire)
    addr = addr.as_bits()
    sel_bits = NamedWire(Bits(32 - ChannelMMIO.RegisterSpaceBits), "sel_bits")
    sel_bits.assign(addr[ChannelMMIO.RegisterSpaceBits:])
    client_addr = NamedWire(Bits(32), "client_addr")
    client_addr.assign(addr & Bits(32)(ChannelMMIO.AddressMask))
    client_addr_chan, client_addr_ready = Channel(UInt(32)).wrap(
        client_addr.as_uint(), addr_valid)
    addr_ready_wire.assign(client_addr_ready)
    return sel_bits, client_addr_chan

  def build_write(self, bundles):
    # TODO: this.
    pass
