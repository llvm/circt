#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
from math import ceil

from ..common import Clock, Input, InputChannel, Output, OutputChannel, Reset
from ..constructs import (AssignableSignal, ControlReg, Counter, Mux, NamedWire,
                          Reg, Wire)
from .. import esi
from ..module import Module, generator, modparams
from ..signals import BitsSignal, BundleSignal, ChannelSignal
from ..support import clog2
from ..types import (Array, Bits, Bundle, BundledChannel, Channel,
                     ChannelDirection, StructType, Type, UInt)

from typing import Callable, Dict, List, Tuple
import typing

MagicNumber = 0x207D98E5_E5100E51  # random + ESI__ESI
VersionNumber = 0  # Version 0: format subject to change

IndirectionMagicNumber = 0x312bf0cc_E5100E51  # random + ESI__ESI
IndirectionVersionNumber = 0  # Version 0: format subject to change


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

  cmd = Input(esi.MMIO.read_write.type)

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

  RegisterSpace = 0x100000
  RegisterSpaceBits = RegisterSpace.bit_length() - 1
  AddressMask = 0xFFFFF

  # Start at this address for assigning MMIO addresses to service requests.
  initial_offset: int = RegisterSpace

  @generator
  def generate(ports, bundles: esi._ServiceGeneratorBundles):
    table, manifest_loc = ChannelMMIO.build_table(bundles)
    ChannelMMIO.build_read(ports, manifest_loc, table)
    return True

  @staticmethod
  def build_table(bundles) -> Tuple[Dict[int, AssignableSignal], int]:
    """Build a table of read and write addresses to BundleSignals."""
    offset = ChannelMMIO.initial_offset
    table: Dict[int, AssignableSignal] = {}
    for bundle in bundles.to_client_reqs:
      if bundle.port == 'read':
        table[offset] = bundle
        bundle.add_record(details={
            "offset": offset,
            "size": ChannelMMIO.RegisterSpace,
            "type": "ro"
        })
        offset += ChannelMMIO.RegisterSpace
      elif bundle.port == 'read_write':
        table[offset] = bundle
        bundle.add_record(details={
            "offset": offset,
            "size": ChannelMMIO.RegisterSpace,
            "type": "rw"
        })
        offset += ChannelMMIO.RegisterSpace
      else:
        assert False, "Unrecognized port name."

    manifest_loc = offset
    return table, manifest_loc

  @staticmethod
  def build_read(ports, manifest_loc: int, table: Dict[int, AssignableSignal]):
    """Builds the read side of the MMIO service."""

    # Instantiate the header and manifest ROM. Fill in the read_table with
    # bundle wires to be assigned identically to the other MMIO clients.
    header_bundle_wire = Wire(esi.MMIO.read.type)
    table[0] = header_bundle_wire
    HeaderMMIO(manifest_loc)(clk=ports.clk,
                             rst=ports.rst,
                             read=header_bundle_wire)

    mani_bundle_wire = Wire(esi.MMIO.read.type)
    table[manifest_loc] = mani_bundle_wire
    ESI_Manifest_ROM_Wrapper(clk=ports.clk, read=mani_bundle_wire)

    # Unpack the cmd bundle.
    data_resp_channel = Wire(Channel(esi.MMIODataType))
    counted_output = Wire(Channel(esi.MMIODataType))
    cmd_channel = ports.cmd.unpack(data=counted_output)["cmd"]
    counted_output.assign(data_resp_channel)

    # Get the selection index and the address to hand off to the clients.
    sel_bits, client_cmd_chan = ChannelMMIO.build_addr_read(cmd_channel)

    # Build the demux/mux and assign the results of each appropriately.
    read_clients_clog2 = clog2(len(table))
    client_cmd_channels = esi.ChannelDemux(
        sel=sel_bits.pad_or_truncate(read_clients_clog2),
        input=client_cmd_chan,
        num_outs=len(table),
        instance_name="client_cmd_demux")
    client_data_channels = []
    for (idx, offset) in enumerate(sorted(table.keys())):
      bundle_wire = table[offset]
      bundle_type = bundle_wire.type
      if bundle_type == esi.MMIO.read.type:
        offset = client_cmd_channels[idx].transform(lambda cmd: cmd.offset)
        bundle, bundle_froms = esi.MMIO.read.type.pack(offset=offset)
      elif bundle_type == esi.MMIO.read_write.type:
        bundle, bundle_froms = esi.MMIO.read_write.type.pack(
            cmd=client_cmd_channels[idx])
      else:
        assert False, "Unrecognized bundle type."
      bundle_wire.assign(bundle)
      client_data_channels.append(bundle_froms["data"])
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

    cmd_ready_wire = Wire(Bits(1))
    cmd, cmd_valid = read_addr_chan.unwrap(cmd_ready_wire)
    sel_bits = NamedWire(Bits(32 - ChannelMMIO.RegisterSpaceBits), "sel_bits")
    sel_bits.assign(cmd.offset.as_bits()[ChannelMMIO.RegisterSpaceBits:])
    client_cmd = NamedWire(esi.MMIOReadWriteCmdType, "client_cmd")
    client_cmd.assign(
        esi.MMIOReadWriteCmdType({
            "write":
                cmd.write,
            "offset": (cmd.offset.as_bits() &
                       Bits(32)(ChannelMMIO.AddressMask)).as_uint(),
            "data":
                cmd.data
        }))
    client_addr_chan, client_addr_ready = Channel(
        esi.MMIOReadWriteCmdType).wrap(client_cmd, cmd_valid)
    cmd_ready_wire.assign(client_addr_ready)
    return sel_bits, client_addr_chan


class MMIOIndirection(Module):
  """Some platforms do not support MMIO space greater than a certain size (e.g.
  Vitis 2022's limit is 4k). This module implements a level of indirection to
  provide access to a full 32-bit address space.

  MMIO addresses:
    - 0x0:  0 constant
    - 0x8:  64 bit ESI magic number for Indirect MMIO (0x312bf0cc_E5100E51)
    - 0x10: Version number for Indirect MMIO (0)
    - 0x18: Location of read/write in the virtual MMIO space.
    - 0x20: A read from this location will initiate a read in the virtual MMIO
            space specified by the address stored in 0x18 and return the result.
            A write to this location will initiate a write into the virtual MMIO
            space to the virtual address specified in 0x18.
  """
  clk = Clock()
  rst = Reset()

  upstream = Input(esi.MMIO.read_write.type)
  downstream = Output(esi.MMIO.read_write.type)

  @generator
  def build(ports):
    # This implementation assumes there is only one outstanding upstream MMIO
    # transaction in flight at once. TODO: enforce this or make it more robust.

    reg_bits = 8
    location_reg = UInt(reg_bits)(0x18)
    indirect_mmio_reg = UInt(reg_bits)(0x20)
    virt_address = Wire(UInt(32))

    # Set up the upstream MMIO interface. Capture last upstream command in a
    # mailbox which never empties to give access to the last command for all
    # time.
    upstream_resp_chan_wire = Wire(Channel(esi.MMIODataType))
    upstream_cmd_chan = ports.upstream.unpack(
        data=upstream_resp_chan_wire)["cmd"]
    _, _, upstream_cmd_data = upstream_cmd_chan.snoop()

    # Set up a channel demux to separate the MMIO commands which get processed
    # locally with ones which should be transformed and fowarded downstream.
    phys_loc = upstream_cmd_data.offset.as_uint(reg_bits)
    fwd_upstream = NamedWire(phys_loc == indirect_mmio_reg, "fwd_upstream")
    local_reg_cmd_chan, downstream_cmd_channel = esi.ChannelDemux(
        upstream_cmd_chan, fwd_upstream, 2, "upstream_demux")

    # Set up the downstream MMIO interface.
    downstream_cmd_channel = downstream_cmd_channel.transform(
        lambda cmd: esi.MMIOReadWriteCmdType({
            "write": cmd.write,
            "offset": virt_address,
            "data": cmd.data
        }))
    ports.downstream, froms = esi.MMIO.read_write.type.pack(
        cmd=downstream_cmd_channel)
    downstream_data_chan = froms["data"]

    # Process local regs.
    (local_reg_cmd_valid, local_reg_cmd_ready,
     local_reg_cmd) = local_reg_cmd_chan.snoop()
    write_virt_address = (local_reg_cmd_valid & local_reg_cmd_ready &
                          local_reg_cmd.write & (phys_loc == location_reg))
    virt_address.assign(
        local_reg_cmd.data.as_uint(32).reg(
            name="virt_address",
            clk=ports.clk,
            ce=write_virt_address,
        ))

    # Build the pysical MMIO register space.
    local_reg_resp_array = Array(Bits(64), 4)([
        0x0,  # 0x0
        IndirectionMagicNumber,  # 0x8
        IndirectionVersionNumber,  # 0x10
        virt_address.as_bits(64),  # 0x18
    ])
    local_reg_resp_chan = local_reg_cmd_chan.transform(
        lambda cmd: local_reg_resp_array[cmd.offset.as_uint(2)])

    upstream_resp = esi.ChannelMux([local_reg_resp_chan, downstream_data_chan])
    upstream_resp_chan_wire.assign(upstream_resp)

    # # Build the command into the virtual space.
    # virt_cmd = esi.MMIOReadWriteCmdType({
    #     "write": upstream_mailbox.data.write,
    #     "offset": virt_address,
    #     "data": upstream_mailbox.data.data
    # })
    # downstream_cmd_chan, downstream_data_chan_ready = Channel(
    #     esi.MMIOReadWriteCmdType).wrap(
    #         virt_cmd, upstream_mailbox.valid & (phys_loc == indirect_mmio_reg))
    # downstream_cmd_chan_wire.assign(downstream_cmd_chan)

    # upstream_addr_words = NamedWire(upstream_mailbox.data.offset.as_bits()[3:6],
    #                                 "upstream_addr_words")
    # upstream_resp_data = upstream_data_resp_array[upstream_addr_words]
    # upstream_resp_valid = Mux(
    #     phys_loc == indirect_mmio_reg,
    #     downstream_data_valid,
    #     upstream_mailbox.valid,
    # )
    # upstream_cmd_ready.assign(
    #     Mux(
    #         phys_loc == indirect_mmio_reg,
    #         downstream_data_chan_ready,
    #         Bits(1)(1),
    #     ))

    # # Wrap the response.
    # upstream_resp_chan, upstream_resp_ready = Channel(esi.MMIODataType).wrap(
    #     upstream_resp_data, upstream_resp_valid)
    # upstream_resp_ready_wire.assign(upstream_resp_ready)
    # upstream_resp_chan_wire.assign(upstream_resp_chan)


@modparams
def TaggedReadGearbox(input_bitwidth: int,
                      output_bitwidth: int) -> type["TaggedReadGearboxImpl"]:
  """Build a gearbox to convert the upstream data to the client data
  type. Assumes a struct {tag, data} and only gearboxes the data. Tag is stored
  separately and the struct is re-assembled later on."""

  class TaggedReadGearboxImpl(Module):
    clk = Clock()
    rst = Reset()
    in_ = InputChannel(
        StructType([
            ("tag", esi.HostMem.TagType),
            ("data", Bits(input_bitwidth)),
        ]))
    out = OutputChannel(
        StructType([
            ("tag", esi.HostMem.TagType),
            ("data", Bits(output_bitwidth)),
        ]))

    @generator
    def build(ports):
      ready_for_upstream = Wire(Bits(1), name="ready_for_upstream")
      upstream_tag_and_data, upstream_valid = ports.in_.unwrap(
          ready_for_upstream)
      upstream_data = upstream_tag_and_data.data
      upstream_xact = ready_for_upstream & upstream_valid

      # Determine if gearboxing is necessary and whether it needs to be
      # gearboxed up or just sliced down.
      if output_bitwidth == input_bitwidth:
        client_data_bits = upstream_data
        client_valid = upstream_valid
      elif output_bitwidth < input_bitwidth:
        client_data_bits = upstream_data[:output_bitwidth]
        client_valid = upstream_valid
      else:
        # Create registers equal to the number of upstream transactions needed
        # to fill the client data. Set the output to the concatenation of said
        # registers.
        chunks = ceil(output_bitwidth / input_bitwidth)
        reg_ces = [Wire(Bits(1)) for _ in range(chunks)]
        regs = [
            upstream_data.reg(ports.clk,
                              ports.rst,
                              ce=reg_ces[idx],
                              name=f"chunk_reg_{idx}") for idx in range(chunks)
        ]
        client_data_bits = BitsSignal.concat(reversed(regs))[:output_bitwidth]

        # Use counter to determine to which register to write and determine if
        # the registers are all full.
        clear_counter = Wire(Bits(1))
        counter_width = clog2(chunks)
        counter = Counter(counter_width)(clk=ports.clk,
                                         rst=ports.rst,
                                         clear=clear_counter,
                                         increment=upstream_xact)
        set_client_valid = counter.out == chunks - 1
        client_xact = Wire(Bits(1))
        client_valid = ControlReg(ports.clk, ports.rst,
                                  [set_client_valid & upstream_xact],
                                  [client_xact])
        client_xact.assign(client_valid & ready_for_upstream)
        clear_counter.assign(client_xact)
        for idx, reg_ce in enumerate(reg_ces):
          reg_ce.assign(upstream_xact &
                        (counter.out == UInt(counter_width)(idx)))

      # Construct the output channel. Shared logic across all three cases.
      tag_reg = upstream_tag_and_data.tag.reg(ports.clk,
                                              ports.rst,
                                              ce=upstream_xact,
                                              name="tag_reg")

      client_channel, client_ready = TaggedReadGearboxImpl.out.type.wrap(
          {
              "tag": tag_reg,
              "data": client_data_bits,
          }, client_valid)
      ready_for_upstream.assign(client_ready)
      ports.out = client_channel

  return TaggedReadGearboxImpl


def HostmemReadProcessor(read_width: int, hostmem_module,
                         reqs: List[esi._OutputBundleSetter]):
  """Construct a host memory read request module to orchestrate the the read
  connections. Responsible for both gearboxing the data, multiplexing the
  requests, reassembling out-of-order responses and routing the responses to the
  correct clients.

  Generate this module dynamically to allow for multiple read clients of
  multiple types to be directly accomodated."""

  class HostmemReadProcessorImpl(Module):
    clk = Clock()
    rst = Reset()

    # Add an output port for each read client.
    reqPortMap: Dict[esi._OutputBundleSetter, str] = {}
    for req in reqs:
      name = "client_" + req.client_name_str
      locals()[name] = Output(req.type)
      reqPortMap[req] = name

    # And then the port which goes to the host.
    upstream = Output(hostmem_module.read.type)

    @generator
    def build(ports):
      """Build the read side of the HostMem service."""

      # If there's no read clients, just return a no-op read bundle.
      if len(reqs) == 0:
        upstream_req_channel, _ = Channel(hostmem_module.UpstreamReadReq).wrap(
            {
                "tag": 0,
                "length": 0,
                "address": 0
            }, 0)
        upstream_read_bundle, _ = hostmem_module.read.type.pack(
            req=upstream_req_channel)
        ports.upstream = upstream_read_bundle
        return

      # Since we use the tag to identify the client, we can't have more than 256
      # read clients. Supporting more than 256 clients would require
      # tag-rewriting, which we'll probably have to implement at some point.
      # TODO: Implement tag-rewriting.
      assert len(reqs) <= 256, "More than 256 read clients not supported."

      # Pack the upstream bundle and leave the request as a wire.
      upstream_req_channel = Wire(Channel(hostmem_module.UpstreamReadReq))
      upstream_read_bundle, froms = hostmem_module.read.type.pack(
          req=upstream_req_channel)
      ports.upstream = upstream_read_bundle
      upstream_resp_channel = froms["resp"]

      demux = esi.TaggedDemux(len(reqs), upstream_resp_channel.type)(
          clk=ports.clk, rst=ports.rst, in_=upstream_resp_channel)

      tagged_client_reqs = []
      for idx, client in enumerate(reqs):
        # Find the response channel in the request bundle.
        resp_type = [
            c.channel for c in client.type.channels if c.name == 'resp'
        ][0]
        demuxed_upstream_channel = demux.get_out(idx)

        # TODO: Should responses come back out-of-order (interleaved tags),
        # re-order them here so the gearbox doesn't get confused. (Longer term.)
        # For now, only support one outstanding transaction at a time.  This has
        # the additional benefit of letting the upstream tag be the client
        # identifier. TODO: Implement the gating logic here.

        # Gearbox the data to the client's data type.
        client_type = resp_type.inner_type
        gearbox = TaggedReadGearbox(read_width, client_type.data.bitwidth)(
            clk=ports.clk, rst=ports.rst, in_=demuxed_upstream_channel)
        client_resp_channel = gearbox.out.transform(lambda m: client_type({
            "tag": m.tag,
            "data": m.data.bitcast(client_type.data)
        }))

        # Assign the client response to the correct port.
        client_bundle, froms = client.type.pack(resp=client_resp_channel)
        client_req = froms["req"]
        tagged_client_req = client_req.transform(
            lambda r: hostmem_module.UpstreamReadReq({
                "address": r.address,
                "length": (client_type.data.bitwidth + 7) // 8,
                # TODO: Change this once we support tag-rewriting.
                "tag": idx
            }))
        tagged_client_reqs.append(tagged_client_req)

        # Set the port for the client request.
        setattr(ports, HostmemReadProcessorImpl.reqPortMap[client],
                client_bundle)

      # Assign the multiplexed read request to the upstream request.
      # TODO: Don't release a request until the client is ready to accept
      # the response otherwise the system could deadlock.
      muxed_client_reqs = esi.ChannelMux(tagged_client_reqs)
      upstream_req_channel.assign(muxed_client_reqs)
      HostmemReadProcessorImpl.reqPortMap.clear()

  return HostmemReadProcessorImpl


@modparams
def TaggedWriteGearbox(input_bitwidth: int,
                       output_bitwidth: int) -> type["TaggedWriteGearboxImpl"]:
  """Build a gearbox to convert the client data to upstream write chunks.
  Assumes a struct {address, tag, data} and only gearboxes the data. Tag is
  stored separately and the struct is re-assembled later on."""

  if output_bitwidth % 8 != 0:
    raise ValueError("Output bitwidth must be a multiple of 8.")
  if input_bitwidth % 8 != 0:
    raise ValueError("Input bitwidth must be a multiple of 8.")

  class TaggedWriteGearboxImpl(Module):
    clk = Clock()
    rst = Reset()
    in_ = InputChannel(
        StructType([
            ("address", UInt(64)),
            ("tag", esi.HostMem.TagType),
            ("data", Bits(input_bitwidth)),
        ]))
    out = OutputChannel(
        StructType([
            ("address", UInt(64)),
            ("tag", esi.HostMem.TagType),
            ("data", Bits(output_bitwidth)),
            ("valid_bytes", Bits(8)),
        ]))

    num_chunks = ceil(input_bitwidth / output_bitwidth)

    @generator
    def build(ports):
      upstream_ready = Wire(Bits(1))
      ready_for_client = Wire(Bits(1))
      client_tag_and_data, client_valid = ports.in_.unwrap(ready_for_client)
      client_data = client_tag_and_data.data
      client_xact = ready_for_client & client_valid
      input_bitwidth_bytes = input_bitwidth // 8
      output_bitwidth_bytes = output_bitwidth // 8

      # Determine if gearboxing is necessary and whether it needs to be
      # gearboxed up or just sliced down.
      if output_bitwidth == input_bitwidth:
        upstream_data_bits = client_data
        upstream_valid = client_valid
        ready_for_client.assign(upstream_ready)
        tag = client_tag_and_data.tag
        address = client_tag_and_data.address
        valid_bytes = Bits(8)(input_bitwidth_bytes)
      elif output_bitwidth > input_bitwidth:
        upstream_data_bits = client_data.as_bits(output_bitwidth)
        upstream_valid = client_valid
        ready_for_client.assign(upstream_ready)
        tag = client_tag_and_data.tag
        address = client_tag_and_data.address
        valid_bytes = Bits(8)(input_bitwidth_bytes)
      else:
        # Create registers equal to the number of upstream transactions needed
        # to complete the transmission.
        num_chunks = TaggedWriteGearboxImpl.num_chunks
        num_chunks_idx_bitwidth = clog2(num_chunks)
        if input_bitwidth % output_bitwidth == 0:
          padding_numbits = 0
        else:
          padding_numbits = output_bitwidth - (input_bitwidth % output_bitwidth)
        assert padding_numbits % 8 == 0, "Padding must be a multiple of 8."
        client_data_padded = BitsSignal.concat(
            [Bits(padding_numbits)(0), client_data])
        chunks = [
            client_data_padded[i * output_bitwidth:(i + 1) * output_bitwidth]
            for i in range(num_chunks)
        ]
        chunk_regs = Array(Bits(output_bitwidth), num_chunks)([
            c.reg(ports.clk, ce=client_xact, name=f"chunk_{idx}")
            for idx, c in enumerate(chunks)
        ])
        increment = Wire(Bits(1))
        clear = Wire(Bits(1))
        counter = Counter(num_chunks_idx_bitwidth)(clk=ports.clk,
                                                   rst=ports.rst,
                                                   increment=increment,
                                                   clear=clear)
        upstream_data_bits = chunk_regs[counter.out]
        upstream_valid = ControlReg(ports.clk, ports.rst, [client_xact],
                                    [clear])
        upstream_xact = upstream_valid & upstream_ready
        clear.assign(upstream_xact & (counter.out == (num_chunks - 1)))
        increment.assign(upstream_xact)
        ready_for_client.assign(~upstream_valid)
        address_padding_bits = clog2(output_bitwidth_bytes)
        counter_bytes = BitsSignal.concat(
            [counter.out.as_bits(),
             Bits(address_padding_bits)(0)]).as_uint()

        # Construct the output channel. Shared logic across all three cases.
        tag_reg = client_tag_and_data.tag.reg(ports.clk,
                                              ce=client_xact,
                                              name="tag_reg")
        addr_reg = client_tag_and_data.address.reg(ports.clk,
                                                   ce=client_xact,
                                                   name="address_reg")
        address = (addr_reg + counter_bytes).as_uint(64)
        tag = tag_reg
        valid_bytes = Mux(counter.out == (num_chunks - 1),
                          Bits(8)(output_bitwidth_bytes),
                          Bits(8)((output_bitwidth - padding_numbits) // 8))

      upstream_channel, upstrm_ready_sig = TaggedWriteGearboxImpl.out.type.wrap(
          {
              "address": address,
              "tag": tag,
              "data": upstream_data_bits,
              "valid_bytes": valid_bytes
          }, upstream_valid)
      upstream_ready.assign(upstrm_ready_sig)
      ports.out = upstream_channel

  return TaggedWriteGearboxImpl


def HostMemWriteProcessor(
    write_width: int, hostmem_module,
    reqs: List[esi._OutputBundleSetter]) -> type["HostMemWriteProcessorImpl"]:
  """Construct a host memory write request module to orchestrate the the write
  connections. Responsible for both gearboxing the data, multiplexing the
  requests, reassembling out-of-order responses and routing the responses to the
  correct clients.

  Generate this module dynamically to allow for multiple write clients of
  multiple types to be directly accomodated."""

  class HostMemWriteProcessorImpl(Module):

    clk = Clock()
    rst = Reset()

    # Add an output port for each read client.
    reqPortMap: Dict[esi._OutputBundleSetter, str] = {}
    for req in reqs:
      name = "client_" + req.client_name_str
      locals()[name] = Output(req.type)
      reqPortMap[req] = name

    # And then the port which goes to the host.
    upstream = Output(hostmem_module.write.type)

    @generator
    def build(ports):
      # If there's no write clients, just create a no-op write bundle
      if len(reqs) == 0:
        req, _ = Channel(hostmem_module.UpstreamWriteReq).wrap(
            {
                "address": 0,
                "tag": 0,
                "data": 0,
                "valid_bytes": 0,
            }, 0)
        write_bundle, _ = hostmem_module.write.type.pack(req=req)
        ports.upstream = write_bundle
        return

      assert len(reqs) <= 256, "More than 256 write clients not supported."

      upstream_req_channel = Wire(Channel(hostmem_module.UpstreamWriteReq))
      upstream_write_bundle, froms = hostmem_module.write.type.pack(
          req=upstream_req_channel)
      ports.upstream = upstream_write_bundle
      upstream_ack_tag = froms["ackTag"]

      demuxed_acks = esi.TaggedDemux(len(reqs), upstream_ack_tag.type)(
          clk=ports.clk, rst=ports.rst, in_=upstream_ack_tag)

      # TODO: re-write the tags and store the client and client tag.

      # Build the write request channels and ack wires.
      write_channels: List[ChannelSignal] = []
      for idx, req in enumerate(reqs):
        # Get the request channel and its data type.
        reqch = [c.channel for c in req.type.channels if c.name == 'req'][0]
        client_type = reqch.inner_type

        # Pack up the bundle and assign the request channel.
        write_req_bundle_type = esi.HostMem.write_req_bundle_type(
            client_type.data)
        bundle_sig, froms = write_req_bundle_type.pack(
            ackTag=demuxed_acks.get_out(idx))

        gearbox_mod = TaggedWriteGearbox(client_type.data.bitwidth, write_width)
        gearbox_in_type = gearbox_mod.in_.type.inner_type
        tagged_client_req = froms["req"]
        bitcast_client_req = tagged_client_req.transform(
            lambda m: gearbox_in_type({
                "tag": m.tag,
                "address": m.address,
                "data": m.data.bitcast(gearbox_in_type.data)
            }))

        # Gearbox the data to the client's data type.
        gearbox = gearbox_mod(clk=ports.clk,
                              rst=ports.rst,
                              in_=bitcast_client_req)
        write_channels.append(
            gearbox.out.transform(lambda m: m.type({
                "address": m.address,
                "tag": idx,
                "data": m.data,
                "valid_bytes": m.valid_bytes
            })))
        # Set the port for the client request.
        setattr(ports, HostMemWriteProcessorImpl.reqPortMap[req], bundle_sig)

      # Build a channel mux for the write requests.
      muxed_write_channel = esi.ChannelMux(write_channels)
      upstream_req_channel.assign(muxed_write_channel)

  return HostMemWriteProcessorImpl


@modparams
def ChannelHostMem(read_width: int,
                   write_width: int) -> typing.Type['ChannelHostMemImpl']:

  class ChannelHostMemImpl(esi.ServiceImplementation):
    """Builds a HostMem service which multiplexes multiple HostMem clients into
    two (read and write) bundles of the given data width."""

    clk = Clock()
    rst = Reset()

    UpstreamReadReq = StructType([
        ("address", UInt(64)),
        ("length", UInt(32)),  # In bytes.
        ("tag", UInt(8)),
    ])
    read = Output(
        Bundle([
            BundledChannel("req", ChannelDirection.TO, UpstreamReadReq),
            BundledChannel(
                "resp", ChannelDirection.FROM,
                StructType([
                    ("tag", esi.HostMem.TagType),
                    ("data", Bits(read_width)),
                ])),
        ]))

    if write_width % 8 != 0:
      raise ValueError("Write width must be a multiple of 8.")
    UpstreamWriteReq = StructType([
        ("address", UInt(64)),
        ("tag", UInt(8)),
        ("data", Bits(write_width)),
        ("valid_bytes", Bits(8)),
    ])
    write = Output(
        Bundle([
            BundledChannel("req", ChannelDirection.TO, UpstreamWriteReq),
            BundledChannel("ackTag", ChannelDirection.FROM, UInt(8)),
        ]))

    @generator
    def generate(ports, bundles: esi._ServiceGeneratorBundles):
      # Split the read side out into a separate module. Must assign the output
      # ports to the clients since we can't service a request in a different
      # module.
      read_reqs = [req for req in bundles.to_client_reqs if req.port == 'read']
      read_proc_module = HostmemReadProcessor(read_width, ChannelHostMemImpl,
                                              read_reqs)
      read_proc = read_proc_module(clk=ports.clk, rst=ports.rst)
      ports.read = read_proc.upstream
      for req in read_reqs:
        req.assign(getattr(read_proc, read_proc_module.reqPortMap[req]))

      # The write side.
      write_reqs = [
          req for req in bundles.to_client_reqs if req.port == 'write'
      ]
      write_proc_module = HostMemWriteProcessor(write_width, ChannelHostMemImpl,
                                                write_reqs)
      write_proc = write_proc_module(clk=ports.clk, rst=ports.rst)
      ports.write = write_proc.upstream
      for req in write_reqs:
        req.assign(getattr(write_proc, write_proc_module.reqPortMap[req]))

  return ChannelHostMemImpl


@modparams
def DummyToHostEngine(client_type: Type) -> type['DummyToHostEngineImpl']:
  """Create a fake DMA engine which just throws everything away."""

  class DummyToHostEngineImpl(esi.EngineModule):

    @property
    def TypeName(self):
      return "DummyToHostEngine"

    clk = Clock()
    rst = Reset()
    input_channel = InputChannel(client_type)

    @generator
    def build(ports):
      pass

  return DummyToHostEngineImpl


@modparams
def DummyFromHostEngine(client_type: Type) -> type['DummyFromHostEngineImpl']:
  """Create a fake DMA engine which just never produces messages."""

  class DummyFromHostEngineImpl(esi.EngineModule):

    @property
    def TypeName(self):
      return "DummyFromHostEngine"

    clk = Clock()
    rst = Reset()
    output_channel = OutputChannel(client_type)

    @generator
    def build(ports):
      valid = Bits(1)(0)
      data = Bits(client_type.bitwidth)(0).bitcast(client_type)
      channel, ready = Channel(client_type).wrap(data, valid)
      ports.output_channel = channel

  return DummyFromHostEngineImpl


def ChannelEngineService(
    to_host_engine_gen: Callable,
    from_host_engine_gen: Callable) -> type['ChannelEngineService']:
  """Returns a channel service implementation which calls
  to_host_engine_gen(<client_type>) or from_host_engine_gen(<client_type>) to
  generate the to_host and from_host engines for each channel. Does not support
  engines which can service multiple clients at once."""

  class ChannelEngineService(esi.ServiceImplementation):
    """Service implementation which services the clients via a per-channel DMA
    engine."""

    clk = Clock()
    rst = Reset()

    @generator
    def build(ports, bundles: esi._ServiceGeneratorBundles):

      def build_engine_appid(client_appid: List[esi.AppID],
                             channel_name: str) -> str:
        appid_strings = [str(appid) for appid in client_appid]
        return f"{'_'.join(appid_strings)}.{channel_name}"

      def build_engine(bc: BundledChannel, input_channel=None) -> Type:
        idbase = build_engine_appid(bundle.client_name, bc.name)
        eng_appid = esi.AppID(idbase)
        if bc.direction == ChannelDirection.FROM:
          engine_mod = to_host_engine_gen(bc.channel.inner_type)
        else:
          engine_mod = from_host_engine_gen(bc.channel.inner_type)
        eng_inputs = {
            "clk": ports.clk,
            "rst": ports.rst,
        }
        eng_details: Dict[str, object] = {"engine_inst": eng_appid}
        if input_channel is not None:
          eng_inputs["input_channel"] = input_channel
        if hasattr(engine_mod, "mmio"):
          mmio_appid = esi.AppID(idbase + ".mmio")
          eng_inputs["mmio"] = esi.MMIO.read_write(mmio_appid)
          eng_details["mmio"] = mmio_appid
        if hasattr(engine_mod, "hostmem"):
          eng_inputs["hostmem"] = esi.HostMem.write_from_bundle(
              esi.AppID(idbase + ".hostmem"), engine_mod.hostmem.type)
        engine = engine_mod(appid=eng_appid, **eng_inputs)
        engine_rec = bundles.emit_engine(engine, details=eng_details)
        engine_rec.add_record(bundle, {bc.name: {}})
        return engine

      for bundle in bundles.to_client_reqs:
        bundle_type = bundle.type
        to_channels = {}
        # Create a DMA engine for each channel headed TO the client (from the host).
        for bc in bundle_type.channels:
          if bc.direction == ChannelDirection.TO:
            engine = build_engine(bc)
            to_channels[bc.name] = engine.output_channel

        client_bundle_sig, froms = bundle_type.pack(**to_channels)
        bundle.assign(client_bundle_sig)

        # Create a DMA engine for each channel headed FROM the client (to the host).
        for bc in bundle_type.channels:
          if bc.direction == ChannelDirection.FROM:
            build_engine(bc, froms[bc.name])

  return ChannelEngineService
