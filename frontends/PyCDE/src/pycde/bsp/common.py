#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
from math import ceil

from ..common import Clock, Input, InputChannel, Output, OutputChannel, Reset
from ..constructs import (AssignableSignal, ControlReg, Counter, NamedWire, Reg,
                          Wire)
from .. import esi
from ..module import Module, generator, modparams
from ..signals import BitsSignal, BundleSignal, ChannelSignal
from ..support import clog2
from ..types import (Array, Bits, Bundle, BundledChannel, Channel,
                     ChannelDirection, StructType, Type, UInt)

from typing import Dict, List, Tuple
import typing

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

  RegisterSpace = 0x10000
  RegisterSpaceBits = RegisterSpace.bit_length() - 1
  AddressMask = 0xFFFF

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
        num_outs=len(table))
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


@modparams
def TaggedGearbox(input_bitwidth: int,
                  output_bitwidth: int) -> type["TaggedGearboxImpl"]:
  """Build a gearbox to convert the upstream data to the client data
  type. Assumes a struct {tag, data} and only gearboxes the data. Tag is stored
  separately and the struct is re-assembled later on."""

  class TaggedGearboxImpl(Module):
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
        client_valid = ControlReg(ports.clk, ports.rst, [set_client_valid],
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
      client_channel, client_ready = TaggedGearboxImpl.out.type.wrap(
          {
              "tag": tag_reg,
              "data": client_data_bits,
          }, client_valid)
      ready_for_upstream.assign(client_ready)
      ports.out = client_channel

  return TaggedGearboxImpl


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

      # TODO: mux together multiple read clients.
      assert len(reqs) == 1, "Only one read client supported for now."

      # Pack the upstream bundle and leave the request as a wire.
      upstream_req_channel = Wire(Channel(hostmem_module.UpstreamReadReq))
      upstream_read_bundle, froms = hostmem_module.read.type.pack(
          req=upstream_req_channel)
      ports.upstream = upstream_read_bundle
      upstream_resp_channel = froms["resp"]

      for client in reqs:
        # Find the response channel in the request bundle.
        resp_type = [
            c.channel for c in client.type.channels if c.name == 'resp'
        ][0]
        # TODO: route the response to the correct client.
        # TODO: tag re-writing to deal with tag aliasing.
        # Pretend to demux the upstream response channel.
        demuxed_upstream_channel = upstream_resp_channel

        # TODO: Should responses come back out-of-order (interleaved tags),
        # re-order them here so the gearbox doesn't get confused. (Longer term.)
        # For now, only support one outstanding transaction at a time.  This has
        # the additional benefit of letting the upstream tag be the client
        # identifier. TODO: Implement the gating logic here.

        # Gearbox the data to the client's data type.
        client_type = resp_type.inner_type
        gearbox = TaggedGearbox(read_width, client_type.data.bitwidth)(
            clk=ports.clk, rst=ports.rst, in_=demuxed_upstream_channel)
        client_resp_channel = gearbox.out.transform(lambda m: client_type({
            "tag": m.tag,
            "data": m.data.bitcast(client_type.data)
        }))

        # Assign the client response to the correct port.
        client_bundle, froms = client.type.pack(resp=client_resp_channel)
        client_req = froms["req"]
        # Set the port for the client request.
        setattr(ports, HostmemReadProcessorImpl.reqPortMap[client],
                client_bundle)

        # Assign the multiplexed read request to the upstream request.
        # TODO: mux together multiple read clients.
        upstream_req_channel.assign(
            client_req.transform(lambda r: hostmem_module.UpstreamReadReq({
                "address": r.address,
                "length": (client_type.data.bitwidth + 7) // 8,
                "tag": r.tag
            })))
      HostmemReadProcessorImpl.reqPortMap.clear()

  return HostmemReadProcessorImpl


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
    UpstreamWriteReq = StructType([
        ("address", UInt(64)),
        ("tag", UInt(8)),
        ("data", Bits(write_width)),
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
      ports.write = ChannelHostMemImpl.build_tagged_write_mux(ports, write_reqs)

    @staticmethod
    def build_tagged_write_mux(
        ports, reqs: List[esi._OutputBundleSetter]) -> BundleSignal:
      """Build the write side of the HostMem service."""

      # If there's no write clients, just return a no-op write bundle
      if len(reqs) == 0:
        req, _ = Channel(ChannelHostMemImpl.UpstreamWriteReq).wrap(
            {
                "address": 0,
                "tag": 0,
                "data": 0
            }, 0)
        write_bundle, _ = ChannelHostMemImpl.write.type.pack(req=req)
        return write_bundle

      # TODO: mux together multiple write clients.
      assert len(reqs) == 1, "Only one write client supported for now."

      # Build the write request channels and ack wires.
      write_channels: List[ChannelSignal] = []
      write_acks = []
      for req in reqs:
        # Get the request channel and its data type.
        reqch = [c.channel for c in req.type.channels if c.name == 'req'][0]
        data_type = reqch.inner_type.data
        assert data_type == Bits(
            write_width
        ), f"Gearboxing not yet supported. Client {req.client_name}"

        # Write acks to be filled in later.
        write_ack = Wire(Channel(UInt(8)))
        write_acks.append(write_ack)

        # Pack up the bundle and assign the request channel.
        write_req_bundle_type = esi.HostMem.write_req_bundle_type(data_type)
        bundle_sig, froms = write_req_bundle_type.pack(ackTag=write_ack)
        tagged_client_req = froms["req"]
        req.assign(bundle_sig)
        write_channels.append(tagged_client_req)

      # TODO: re-write the tags and store the client and client tag.

      # Build a channel mux for the write requests.
      tagged_write_channel = esi.ChannelMux(write_channels)
      upstream_write_bundle, froms = ChannelHostMemImpl.write.type.pack(
          req=tagged_write_channel)
      ack_tag = froms["ackTag"]
      # TODO: decode the ack tag and assign it to the correct client.
      write_acks[0].assign(ack_tag)
      return upstream_write_bundle

  return ChannelHostMemImpl
