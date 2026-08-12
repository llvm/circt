#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
from math import ceil

from pycde.common import Clock, Input, InputChannel, Output, OutputChannel, Reset
from pycde.constructs import (AssignableSignal, ControlReg, Counter, Mux,
                              NamedWire, Reg, Wire)
from pycde import esi
from pycde.module import Module, generator, modparams
from pycde.signals import BitsSignal, ChannelSignal, StructSignal
from pycde.support import clog2
from pycde.system import System
from pycde.types import (Array, Bits, Bundle, BundledChannel, Channel,
                         ChannelDirection, StructType, Type, UInt, Window)

from ..components import ChannelArbiter, MaxOutstandingLimiter

from typing import Callable, Dict, List, Tuple
import typing

MagicNumber = 0x207D98E5_E5100E51  # random + ESI__ESI
VersionNumber = 0  # Version 0: format subject to change

IndirectionMagicNumber = 0x312bf0cc_E5100E51  # random + ESI__ESI
IndirectionVersionNumber = 0  # Version 0: format subject to change

# Magic value which, when written by the host to header slot 7, requests a
# design reset. Keep in sync with 'ResetMagicNumber' in the runtime
# (cpp/include/esi/Accelerator.h). This magic number guards against "write
# spraying" which other devices have been know to do on boot.
ResetMagicNumber = 0x00000E510000B007
# Number of cycles to wait after a reset is requested before asserting it. This
# gives in-flight transactions time to drain.
ResetCycles = 8192


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
    read = Input(esi.MMIO.read_write.type)
    # Asserted for one cycle when the host writes the reset magic number to
    # header slot 7. Propagates up to the BSP which performs the actual reset.
    reset_request = Output(Bits(1))

    @generator
    def build(ports):
      clk = ports.clk
      rst = ports.rst
      data_chan_wire = Wire(Channel(esi.MMIODataType))
      input_bundles = ports.read.unpack(data=data_chan_wire)
      cmd_chan = input_bundles['cmd']

      # Two-stage half-throughput pipeline: stage 1 captures the incoming
      # command, stage 2 holds the looked-up response. Each stage carries its
      # own occupancy bit.
      cmd_ready = Wire(Bits(1))
      s1_to_s2_xact = Wire(Bits(1))
      cmd_raw, cmd_valid = cmd_chan.unwrap(cmd_ready)

      # Stage 1: command capture register and occupancy bit.
      s1_load = cmd_valid & cmd_ready
      cmd = cmd_raw.reg(clk, rst, ce=s1_load, name="cmd")
      s1_valid = ControlReg(clk,
                            rst,
                            asserts=[s1_load],
                            resets=[s1_to_s2_xact],
                            name="s1_valid")
      # Accept a new command when stage 1 is empty.
      cmd_ready.assign(~s1_valid)

      address_words = cmd.offset.as_bits()[3:]  # Lop off the lower three bits.
      slot = address_words[:3]

      cycles = Counter(64)(clk=ports.clk,
                           rst=ports.rst,
                           clear=Bits(1)(0),
                           increment=Bits(1)(1),
                           instance_name="cycle_counter")

      # Layout the header as an array.
      core_freq = System.current().core_freq
      if core_freq is None:
        core_freq = 0
      header = Array(Bits(64), 8)([
          0,  # Generally a good idea to not use address 0.
          MagicNumber,  # ESI magic number.
          VersionNumber,  # ESI version number.
          manifest_loc,  # Absolute address of the manifest ROM.
          0,  # Reserved for future use.
          cycles.out.as_bits(),  # Cycle counter.
          core_freq,  # Core frequency, if known.
          0,  # Slot 7: write the reset magic number here to request a reset.
      ])
      header.name = "header"

      # Stage 2: registered response value and its occupancy bit.
      s2_valid = Wire(Bits(1))
      data_chan_ready = Wire(Bits(1))
      s2_xact = s2_valid & data_chan_ready
      # Stage 1 advances into stage 2 only when stage 2 is empty.
      s1_to_s2_xact.assign(s1_valid & ~s2_valid)

      header_out = header[slot].reg(clk=clk,
                                    rst=rst,
                                    ce=s1_to_s2_xact,
                                    name="header_out")
      s2_valid.assign(
          ControlReg(clk,
                     rst,
                     asserts=[s1_to_s2_xact],
                     resets=[s2_xact],
                     name="header_out_valid"))
      # Wrap the response.
      data_chan, data_chan_ready_sig = Channel(esi.MMIODataType).wrap(
          header_out, s2_valid)
      data_chan_wire.assign(data_chan)
      data_chan_ready.assign(data_chan_ready_sig)

      # Detect a write of the reset magic number to slot 7. Register the request
      # so it is a clean one-cycle pulse, asserted as the command advances into
      # the response stage. 'DesignResetController' latches it, so a single-cycle
      # pulse is sufficient to trigger the reset.
      reset_detect = (cmd.write & (slot == Bits(3)(7)) &
                      (cmd.data == Bits(64)(ResetMagicNumber)))
      ports.reset_request = reset_detect & s1_to_s2_xact

  return HeaderMMIO


@modparams
def ChannelDemuxN_HalfStage_ReadyBlocking(
    data_type: Type, num_outs: int,
    next_sel_width: int) -> type["ChannelDemuxNImpl"]:
  """N-way channel demultiplexer for valid/ready signaling. Contains
    valid/ready registers on the output channels. The selection signal is now
    embedded in the input channel payload as a struct {sel, data}. Input
    signals ready when the selected output register is empty."""

  assert num_outs >= 1, "num_outs must be at least 1."

  class ChannelDemuxNImpl(Module):
    clk = Clock()
    rst = Reset()

    # Input channel now carries selection along with data.
    InPayloadType = StructType([
        ("sel", Bits(clog2(num_outs))),
        ("next_sel", Bits(next_sel_width)),
        ("data", data_type),
    ])
    inp = Input(Channel(InPayloadType))
    OutPayloadType = StructType([
        ("next_sel", Bits(next_sel_width)),
        ("data", data_type),
    ])
    # Outputs are channels of OutPayloadType, which includes both 'next_sel' and 'data' fields.
    for i in range(num_outs):
      locals()[f"output_{i}"] = Output(Channel(OutPayloadType))

    @generator
    def generate(ports) -> None:
      # Half-stage demux: one register per output channel. Input is ready
      # when the currently selected output register is empty (not valid).
      clk = ports.clk
      rst = ports.rst
      sel_width = clog2(num_outs)

      # Unwrap input with backpressure from selected output register.
      input_ready = Wire(Bits(1), name="input_ready")
      in_payload, in_valid = ports.inp.unwrap(input_ready)
      in_sel = in_payload.sel
      in_next_sel = in_payload.next_sel
      in_data = in_payload.data

      # Track per-output valid regs and build a purely combinational
      # expression 'selected_valid_expr' = OR_i((sel==i)&valid_i). Avoid
      # assigning to a Wire multiple times.
      valid_regs: List[BitsSignal] = []
      selected_valid_expr = Bits(1)(0)

      for i in range(num_outs):
        # Write when input transaction targets this output and output not holding data yet.
        will_write = Wire(Bits(1), name=f"will_write_{i}")
        write_cond = (in_valid & input_ready & (in_sel == Bits(sel_width)(i)))
        will_write.assign(write_cond)

        # Data and next_sel registers.
        out_msg_reg = ChannelDemuxNImpl.OutPayloadType({
            "next_sel": in_next_sel,
            "data": in_data
        }).reg(clk=clk, rst=rst, ce=will_write, name=f"out{i}_msg_reg")

        # Valid register cleared on successful downstream consume.
        consume = Wire(Bits(1), name=f"consume_{i}")
        valid_reg = ControlReg(
            clk=clk,
            rst=rst,
            asserts=[will_write],
            resets=[consume],
            name=f"out{i}_valid_reg",
        )
        valid_regs.append(valid_reg)

        # Channel wrapper.
        ch_sig, ch_ready = Channel(ChannelDemuxNImpl.OutPayloadType).wrap(
            out_msg_reg, valid_reg)
        setattr(ports, f"output_{i}", ch_sig)
        consume.assign(valid_reg & ch_ready)

        # Accumulate selected_valid expression.
        selected_valid_expr = selected_valid_expr | (
            (in_sel == Bits(sel_width)(i)) & valid_reg)

      # Input ready only when selected output has no valid data latched.
      input_ready.assign(selected_valid_expr ^ Bits(1)(1))

    def get_out(self, index: int) -> ChannelSignal:
      return getattr(self, f"output_{index}")

  return ChannelDemuxNImpl


@modparams
def ChannelDemuxTree_HalfStage_ReadyBlocking(
    data_type: Type, num_outs: int,
    branching_factor_log2: int) -> type["ChannelDemuxTree"]:
  """Pipelined N-way channel demultiplexer for valid/ready signaling. This
    implementation uses a tree structure of
    ChannelDemuxN_HalfStage_ReadyBlocking modules to reduce fanout pressure.
    Supports maximum half-throughput to save complexity and area.
    """

  root_sel_width = clog2(num_outs)
  # Simplify algorithm by making sure num_outs is a power of two.
  num_outs = 2**root_sel_width
  sel_width = branching_factor_log2
  fanout = 2**sel_width

  class ChannelDemuxTree(Module):
    clk = Clock()
    rst = Reset()
    # Input now embeds selection bits alongside data.
    InPayloadType = StructType([
        ("sel", Bits(clog2(num_outs))),
        ("data", data_type),
    ])
    inp = Input(Channel(InPayloadType))

    # Outputs (data only).
    for i in range(num_outs):
      locals()[f"output_{i}"] = Output(Channel(data_type))

    @generator
    def build(ports) -> None:
      assert branching_factor_log2 > 0
      if num_outs == 1:
        # Strip selection bits and return single channel.
        setattr(ports, "output_0", ports.inp.transform(lambda p: p.data))
        return

      def payload_type(sel_width: int, next_sel_width: int) -> Type:
        return StructType([
            ("sel", Bits(sel_width)),
            ("next_sel", Bits(next_sel_width)),
            ("data", data_type),
        ])

      def next_sel_width_calc(curr_sel_width) -> int:
        return max(curr_sel_width - sel_width, 0)

      def payload_next(curr_msg: StructSignal) -> StructSignal:
        """Given current level payload, produce next level payload by
        stripping off the top selection bits."""

        next_sel_width = next_sel_width_calc(curr_msg.next_sel.type.width)
        curr_sel_width = curr_msg.next_sel.type.width
        new_sel_width = min(curr_sel_width, sel_width)
        return payload_type(
            new_sel_width,
            next_sel_width,
        )({
            # Use the MSB bits of next_sel as the next level selection.
            "sel": (curr_msg.next_sel[next_sel_width:]
                    if curr_sel_width > 0 else Bits(0)(0)),
            "next_sel": (curr_msg.next_sel[:next_sel_width]
                         if next_sel_width > 0 else Bits(0)(0)),
            "data": curr_msg.data,
        })

      current_channels: List[ChannelSignal] = [
          ports.inp.transform(lambda m: payload_type(0, root_sel_width)({
              "sel": Bits(0)(0),
              "next_sel": m.sel,
              "data": m.data,
          }))
      ]

      curr_sel_width = root_sel_width
      level = 0
      while len(current_channels) < num_outs:
        next_level: List[ChannelSignal] = []
        level_num_outs = min(2**curr_sel_width, fanout)
        for i, c in enumerate(current_channels):
          dmux = ChannelDemuxN_HalfStage_ReadyBlocking(
              data_type,
              num_outs=level_num_outs,
              next_sel_width=next_sel_width_calc(curr_sel_width),
          )(
              clk=ports.clk,
              rst=ports.rst,
              inp=c.transform(payload_next),
              instance_name=f"demux_l{level}_i{i}",
          )
          for j in range(level_num_outs):
            next_level.append(dmux.get_out(j))
        current_channels = next_level
        curr_sel_width -= sel_width
        level += 1

      for i in range(num_outs):
        # Strip off next_sel bits for final output.
        setattr(
            ports,
            f"output_{i}",
            current_channels[i].transform(lambda p: p.data),
        )

    def get_out(self, index: int) -> ChannelSignal:
      return getattr(self, f"output_{index}")

  return ChannelDemuxTree


@modparams
def DesignResetController(
    delay_cycles: int) -> type["DesignResetControllerImpl"]:
  """Counts `delay_cycles` clock cycles after a reset request is observed, then
  asserts `design_reset` for one cycle. This module must be driven by the
  *external* reset only (not the reset it generates) so that the countdown is
  not disturbed by the reset it produces.

  `reset_pending` is asserted from the moment a reset is requested until it
  fires. It is intended to be used to quiesce the design (e.g. stop accepting
  new transactions) so that nothing is in flight when the reset is asserted."""

  if delay_cycles < 1:
    raise ValueError("'delay_cycles' must be at least 1.")

  counter_width = max(clog2(delay_cycles), 1)

  class DesignResetControllerImpl(Module):
    clk = Clock()
    rst = Reset()
    reset_request = Input(Bits(1))
    design_reset = Output(Bits(1))
    # High from the cycle a reset is requested until it fires. Use this to stop
    # accepting new work so in-flight transactions can drain before the reset.
    reset_pending = Output(Bits(1))

    @generator
    def build(ports):
      fire = Wire(Bits(1))
      # Latch that a reset has been requested until we fire the reset.
      pending = ControlReg(clk=ports.clk,
                           rst=ports.rst,
                           asserts=[ports.reset_request],
                           resets=[fire],
                           name="reset_pending")
      # Count cycles while a reset is pending.
      count = Counter(counter_width)(clk=ports.clk,
                                     rst=ports.rst,
                                     clear=fire | ~pending,
                                     increment=pending,
                                     instance_name="reset_delay_counter")
      fire.assign(pending &
                  (count.out == UInt(counter_width)(delay_cycles - 1)))
      ports.design_reset = fire
      ports.reset_pending = pending

  return DesignResetControllerImpl


class ChannelMMIO(esi.ServiceImplementation):
  """MMIO service implementation with MMIO bundle interfaces. Should be
  relatively easy to adapt to physical interfaces by wrapping the wires to
  channels then bundles. Allows the implementation to be shared and (hopefully)
  platform independent.

  Whether or not to support unaligned accesses is up to the clients. The header
  and manifest do not support unaligned accesses and throw away the lower three
  bits.

  Only allows one outstanding request at a time. This is enforced in hardware
  by a `MaxOutstandingLimiter` on the command channel, which stalls incoming
  commands until the previous response has been consumed. If a client fails to
  return a response, the MMIO service will hang. TODO: add some kind of
  timeout.

  Implementation-defined MMIO layout:
    - 0x0: 0 constant
    - 0x8: Magic number (0x207D98E5_E5100E51)
    - 0x12: ESI version number (0)
    - 0x18: Location of the manifest ROM (absolute address)

    - 0x800: Start of MMIO space for requests. Mapping is contained in the
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

  # Asserted for one cycle when the host requests a design reset via an MMIO
  # write to the header. Propagates up to the BSP which performs the reset.
  reset_request = Output(Bits(1))

  # Amount of register space each client gets. This is a GIANT HACK and needs to
  # be replaced by parameterizable services.
  # TODO: make the amount of register space each client gets a parameter.
  # Supporting this will require more address decode logic.

  RegisterSpace = 0x800
  RegisterSpaceBits = RegisterSpace.bit_length() - 1
  AddressMask = RegisterSpace - 1

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
    header_bundle_wire = Wire(esi.MMIO.read_write.type)
    table[0] = header_bundle_wire
    header = HeaderMMIO(manifest_loc)(clk=ports.clk,
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

    # Enforce the single-outstanding-transaction invariant in hardware: hold
    # off accepting a new command until the response to the previous command
    # has been consumed by the host. Snoop the response wire for the
    # completion pulse.
    resp_xact, _ = counted_output.snoop_xact()
    cmd_limiter = MaxOutstandingLimiter(cmd_channel.type.inner_type,
                                        max_outstanding=1)(
                                            clk=ports.clk,
                                            rst=ports.rst,
                                            in_=cmd_channel,
                                            complete=resp_xact,
                                            instance_name="cmd_rate_limiter",
                                        )
    cmd_channel = cmd_limiter.out

    # Get the selection index and the address to hand off to the clients.
    sel_bits, client_cmd_chan = ChannelMMIO.build_addr_read(
        cmd_channel, len(table), manifest_loc)

    # Build the demux/mux and assign the results of each appropriately.
    read_clients_clog2 = clog2(len(table))
    # Combine selection bits and command channel payload into a struct channel for the demux tree.
    TreeInType = StructType([
        ("sel", Bits(read_clients_clog2)),
        ("data", client_cmd_chan.type.inner_type),
    ])
    sel_bits_truncated = sel_bits.pad_or_truncate(read_clients_clog2)
    combined_cmd_chan = client_cmd_chan.transform(
        lambda cmd, _sel=sel_bits_truncated: TreeInType({
            "sel": _sel,
            "data": cmd
        }))
    demux_inst = ChannelDemuxTree_HalfStage_ReadyBlocking(
        client_cmd_chan.type.inner_type, len(table), branching_factor_log2=2)(
            clk=ports.clk,
            rst=ports.rst,
            inp=combined_cmd_chan,
            instance_name="client_cmd_demux",
        )
    client_cmd_channels = [demux_inst.get_out(i) for i in range(len(table))]
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

    # The header surfaces a reset request when the host writes the reset magic
    # number to slot 7. Propagate it up to the caller (the BSP).
    ports.reset_request = header.reset_request

  @staticmethod
  def build_addr_read(read_addr_chan: ChannelSignal, num_clients: int,
                      manifest_loc: int) -> Tuple[BitsSignal, ChannelSignal]:
    """Build a channel for the address read request. Returns the index to select
    the client and a channel for the masked address to be passed to the
    clients."""

    # Decoding the selection bits is very simple as of now. This might need to
    # change to support more flexibility in addressing. Not clear if what we're
    # doing now it sufficient or not.

    manifest_loc_const = UInt(32)(manifest_loc)

    cmd_ready_wire = Wire(Bits(1))
    cmd, cmd_valid = read_addr_chan.unwrap(cmd_ready_wire)
    is_manifest_read = cmd.offset >= manifest_loc_const
    sel_bits = NamedWire(Bits(32 - ChannelMMIO.RegisterSpaceBits), "sel_bits")
    # If reading the manifest, override the selection to select the manifest instead.
    sel_bits.assign(
        Mux(is_manifest_read,
            cmd.offset.as_bits()[ChannelMMIO.RegisterSpaceBits:],
            Bits(32 - ChannelMMIO.RegisterSpaceBits)(num_clients - 1)))
    regular_client_offset = (cmd.offset.as_bits() &
                             Bits(32)(ChannelMMIO.AddressMask)).as_uint()
    offset = Mux(is_manifest_read, regular_client_offset,
                 (cmd.offset - manifest_loc_const).as_uint(32))
    client_cmd = NamedWire(esi.MMIOReadWriteCmdType, "client_cmd")
    client_cmd.assign(
        esi.MMIOReadWriteCmdType({
            "write": cmd.write,
            "offset": offset,
            "data": cmd.data
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

    # Mux together the local register responses and the downstream data to
    # create the upstream response.
    upstream_resp = esi.ChannelMux([local_reg_resp_chan, downstream_data_chan])
    upstream_resp_chan_wire.assign(upstream_resp)


@modparams
def SliceReadGearbox(input_bitwidth: int,
                     output_bitwidth: int) -> type["SliceReadGearboxImpl"]:
  """Narrow one engine word to a single-message client element no wider than the
  word (``OUT <= IN``). The element sits in the word's low bits, so the datapath
  is a slice; ``valid_bytes`` is unused (a single element is never a partial
  word). Wider single elements use `ConcatReadGearbox`; packed list reads use
  `DepackReadGearbox`/`ShiftReadGearbox`."""

  if input_bitwidth <= 0 or input_bitwidth % 8 != 0:
    raise ValueError("engine word width must be a positive multiple of 8 bits")
  if not 0 < output_bitwidth <= input_bitwidth:
    raise ValueError("SliceReadGearbox requires 0 < output <= input")

  in_bytes = input_bitwidth // 8
  vb_width = clog2(in_bytes)

  class SliceReadGearboxImpl(Module):
    clk = Clock()
    rst = Reset()
    in_ = InputChannel(
        StructType([
            ("tag", esi.HostMem.TagType),
            ("data", Bits(input_bitwidth)),
            ("valid_bytes", UInt(vb_width)),
            ("last", Bits(1)),
        ]))
    out = OutputChannel(
        StructType([
            ("tag", esi.HostMem.TagType),
            ("data", Bits(output_bitwidth)),
            ("last", Bits(1)),
        ]))

    @generator
    def build(ports):
      up_ready = Wire(Bits(1), name="up_ready")
      up, up_valid = ports.in_.unwrap(up_ready)
      client_channel, client_ready = SliceReadGearboxImpl.out.type.wrap(
          {
              "tag": up.tag,
              "data": up.data[:output_bitwidth],
              "last": up.last,
          }, up_valid)
      up_ready.assign(client_ready)
      ports.out = client_channel

  return SliceReadGearboxImpl


@modparams
def ConcatReadGearbox(input_bitwidth: int,
                      output_bitwidth: int) -> type["ConcatReadGearboxImpl"]:
  """Concatenate ``ceil(OUT/IN)`` consecutive engine words into one client
  element wider than the word (``OUT > IN``). Serves single-message reads (any
  ``OUT > IN``; the low ``OUT`` bits of the concatenation are the element) and
  contiguous list reads whose element is a whole number of output_bitwidth
  (``OUT % IN == 0``, so elements never straddle). ``valid_bytes`` is unused:
  such lists have no partial words and a single element is one flit. Straddling
  lists use `ShiftReadGearbox`."""

  if input_bitwidth <= 0 or input_bitwidth % 8 != 0:
    raise ValueError("engine word width must be a positive multiple of 8 bits")
  if output_bitwidth <= input_bitwidth:
    raise ValueError("ConcatReadGearbox requires output > input")

  in_bytes = input_bitwidth // 8
  vb_width = clog2(in_bytes)

  class ConcatReadGearboxImpl(Module):
    clk = Clock()
    rst = Reset()
    in_ = InputChannel(
        StructType([
            ("tag", esi.HostMem.TagType),
            ("data", Bits(input_bitwidth)),
            ("valid_bytes", UInt(vb_width)),
            ("last", Bits(1)),
        ]))
    out = OutputChannel(
        StructType([
            ("tag", esi.HostMem.TagType),
            ("data", Bits(output_bitwidth)),
            ("last", Bits(1)),
        ]))

    @generator
    def build(ports):
      ready_for_upstream = Wire(Bits(1), name="ready_for_upstream")
      # Register the input for fmax; the ESI channel buffer keeps the handshake
      # elastic.
      in_reg = ports.in_.buffer(ports.clk, ports.rst, stages=1)
      up, upstream_valid = in_reg.unwrap(ready_for_upstream)
      upstream_data = up.data
      upstream_last = up.last
      upstream_xact = ready_for_upstream & upstream_valid

      # Registers accumulate `chunks` upstream words into one client element;
      # the output is their concatenation. For a list, elements stream back to
      # back and 'last' rides the final word of the burst's final element.
      chunks = ceil(output_bitwidth / input_bitwidth)
      counter_width = clog2(chunks)
      reg_ces = [Wire(Bits(1)) for _ in range(chunks)]
      regs = [
          upstream_data.reg(ports.clk,
                            ports.rst,
                            ce=reg_ces[idx],
                            name=f"chunk_reg_{idx}") for idx in range(chunks)
      ]
      client_data_bits = BitsSignal.concat(reversed(regs))[:output_bitwidth]

      # Pair-index counter: the word accepted this cycle is written to
      # chunk_reg[counter]. 'Counter' clears in preference to incrementing, so
      # mask the clear with the accept -- a consume and an accept on the same
      # cycle means the accepted word is chunk 0 of the *next* element, so the
      # index must land on 1, not 0. 'chunks' need not be a power of two, so
      # wrap explicitly rather than relying on the counter's natural rollover.
      counter = Wire(UInt(counter_width), name="chunk_counter")
      client_xact = Wire(Bits(1))
      set_client_valid = counter == UInt(counter_width)(chunks - 1)
      counter.assign(
          Counter(counter_width)(clk=ports.clk,
                                 rst=ports.rst,
                                 clear=(upstream_xact & set_client_valid) |
                                 (client_xact & ~upstream_xact),
                                 increment=upstream_xact,
                                 instance_name="chunk_counter").out)
      client_valid = ControlReg(ports.clk, ports.rst,
                                [set_client_valid & upstream_xact],
                                [client_xact])
      for idx, reg_ce in enumerate(reg_ces):
        reg_ce.assign(upstream_xact & (counter == UInt(counter_width)(idx)))
      # 'last' of the final engine word that completes this client flit.
      client_last = upstream_last.reg(ports.clk,
                                      ports.rst,
                                      ce=upstream_xact,
                                      name="last_reg")
      tag_reg = up.tag.reg(ports.clk,
                           ports.rst,
                           ce=upstream_xact,
                           name="tag_reg")

      client_channel, client_ready = ConcatReadGearboxImpl.out.type.wrap(
          {
              "tag": tag_reg,
              "data": client_data_bits,
              "last": client_last,
          }, client_valid)
      client_xact.assign(client_valid & client_ready)
      ready_for_upstream.assign(~client_valid | client_ready)
      ports.out = client_channel

  return ConcatReadGearboxImpl


@modparams
def DepackReadGearbox(input_bitwidth: int,
                      output_bitwidth: int) -> type["DepackReadGearboxImpl"]:
  """Unpack a byte-aligned element that divides the engine word
  (``OUT % 8 == 0`` and ``IN % OUT == 0``) from a contiguous list response. Each
  word holds ``IN/OUT`` gap-free elements that never straddle, so a counter
  drives a parts:1 element mux -- no shifter (e.g. 32b/64b, 64b/256b).
  ``valid_bytes`` locates the last element in the burst's (possibly partial)
  final word. Straddling relationships use `ShiftReadGearbox`."""

  if input_bitwidth % 8 != 0:
    raise ValueError("engine word width must be a multiple of 8 bits")
  if output_bitwidth == 0 or output_bitwidth % 8 != 0 \
      or input_bitwidth % output_bitwidth != 0:
    raise ValueError(
        "DepackReadGearbox requires a byte-aligned element that divides the "
        "engine word")

  in_bytes = input_bitwidth // 8
  # 'valid_bytes' is the real byte count minus 1; a word always has >= 1 byte.
  vb_width = clog2(in_bytes)
  count_width = clog2(in_bytes + 1)
  parts = input_bitwidth // output_bitwidth
  elem_bytes = output_bitwidth // 8

  class DepackReadGearboxImpl(Module):
    clk = Clock()
    rst = Reset()
    in_ = InputChannel(
        StructType([
            ("tag", esi.HostMem.TagType),
            ("data", Bits(input_bitwidth)),
            ("valid_bytes", UInt(vb_width)),
            ("last", Bits(1)),
        ]))
    out = OutputChannel(
        StructType([
            ("tag", esi.HostMem.TagType),
            ("data", Bits(output_bitwidth)),
            ("last", Bits(1)),
        ]))

    @generator
    def build(ports):
      client_ready = Wire(Bits(1), name="client_ready")
      up_ready = Wire(Bits(1), name="up_ready")
      # Register the input for fmax; the ESI channel buffer keeps the handshake
      # elastic and decouples the ready path.
      in_reg = ports.in_.buffer(ports.clk, ports.rst, stages=1)
      up, up_valid = in_reg.unwrap(up_ready)
      client_xact = up_valid & client_ready

      if parts == 1:
        # One element per word; nothing to select.
        last_in_word = Bits(1)(1)
        client_data = up.data
      else:
        idx_width = clog2(parts)
        idx = Reg(UInt(idx_width),
                  clk=ports.clk,
                  rst=ports.rst,
                  rst_value=0,
                  ce=client_xact,
                  name="idx")
        # (idx + 1) * elem_bytes == real valid bytes marks the word's last
        # element (the final word may hold fewer than `parts`); add 1 back to
        # the biased 'valid_bytes' to recover the real count.
        real_valid_bytes = (up.valid_bytes + UInt(1)(1)).as_uint(count_width)
        consumed = ((idx + UInt(1)(1)) *
                    UInt(count_width)(elem_bytes)).as_uint(count_width)
        last_in_word = consumed == real_valid_bytes
        # parts:1 element-select mux -- the entire datapath, no shifter.
        word_parts = Array(Bits(output_bitwidth), parts)([
            up.data[k * output_bitwidth:(k + 1) * output_bitwidth]
            for k in range(parts)
        ])
        client_data = word_parts[idx]
        idx.assign(
            Mux(last_in_word, (idx + UInt(1)(1)).as_uint(idx_width),
                UInt(idx_width)(0)))

      # Consume the buffered word as its last element leaves.
      up_ready.assign(client_xact & last_in_word)
      client_channel, client_ready_sig = DepackReadGearboxImpl.out.type.wrap(
          {
              "tag": up.tag,
              "data": client_data,
              "last": (up.last & last_in_word).as_bits(),
          }, up_valid)
      client_ready.assign(client_ready_sig)
      ports.out = client_channel

  return DepackReadGearboxImpl


@modparams
def ShiftReadGearbox(input_bitwidth: int,
                     output_bitwidth: int) -> type["ShiftReadGearboxImpl"]:
  """Universal fallback: unpack a contiguous, byte-packed element stream (a
  `read_list` response) for ANY ``(input_bitwidth, output_bitwidth)`` pair.

  Elements are packed at their natural byte stride ``stride = ceil(OUT/8)``
  bytes, so element k begins at wire bit ``k*stride*8`` and, in general,
  straddles engine-word boundaries at an arbitrary bit offset. A byte-addressed
  shift-register accumulator realigns each element across words. This is correct
  for every width relationship; `SliceReadGearbox`, `ConcatReadGearbox` and
  `DepackReadGearbox` are optimizations that avoid this barrel shifter for the
  regular (non-straddling) cases.

  Each input word carries ``valid_bytes`` (how many of its bytes are real) and
  ``last``. Both are framed to one whole `read_list` request rather than to the
  transport: `HostMemReadReqSplitter` drops the per-chunk framing of the reads
  it issues and re-derives these from the request's total length, so only the
  request's final word is ever partial. That length is ``num_elements *
  stride``, so tracking real bytes lets the gearbox emit exactly the right
  elements and place the list-terminating ``last`` on the final one -- no
  padding element is ever emitted."""

  if input_bitwidth % 8 != 0:
    raise ValueError("engine word width must be a multiple of 8 bits")
  if output_bitwidth <= 0:
    raise ValueError("client element width must be positive")
  in_bytes = input_bitwidth // 8
  stride_bytes = (output_bitwidth + 7) // 8
  stride_bits = stride_bytes * 8
  # Hold at most one partial element plus one freshly accepted word.
  buf_bytes = stride_bytes + in_bytes
  buf_bits = buf_bytes * 8
  # 'valid_bytes' is the real byte count minus 1; a word always has >= 1 byte.
  vb_width = clog2(in_bytes)
  cnt_width = clog2(buf_bytes + 1)
  # The append offset is only ever in [0, stride_bytes] (has_room), so the shift
  # index needs fewer bits than the full count -- see `build`.
  offset_width = clog2(stride_bytes + 1)

  class ShiftReadGearboxImpl(Module):
    clk = Clock()
    rst = Reset()
    in_ = InputChannel(
        StructType([
            ("tag", esi.HostMem.TagType),
            ("data", Bits(input_bitwidth)),
            ("valid_bytes", UInt(vb_width)),
            ("last", Bits(1)),
        ]))
    out = OutputChannel(
        StructType([
            ("tag", esi.HostMem.TagType),
            ("data", Bits(output_bitwidth)),
            ("last", Bits(1)),
        ]))

    @generator
    def build(ports):
      client_ready = Wire(Bits(1), name="client_ready")
      up_ready = Wire(Bits(1), name="up_ready")
      # Register the input for fmax; the ESI channel buffer keeps the handshake
      # elastic and decouples the ready path.
      in_reg = ports.in_.buffer(ports.clk, ports.rst, stages=1)
      up, up_valid = in_reg.unwrap(up_ready)

      from pycde.circt.dialects import comb

      # Byte-addressed accumulator: `buffer` holds `count` valid bytes packed
      # from bit 0 up; the element being emitted is buffer[0:OUT].
      buffer = Reg(Bits(buf_bits),
                   clk=ports.clk,
                   rst=ports.rst,
                   rst_value=0,
                   name="buffer")
      count = Reg(UInt(cnt_width),
                  clk=ports.clk,
                  rst=ports.rst,
                  rst_value=0,
                  name="count")
      saw_last = Wire(Bits(1), name="saw_last")

      # Accept a whole engine word only when there's room, and never while
      # draining a finished burst -- otherwise the next burst's bytes would mix
      # into this one's buffer.
      has_room = count <= UInt(cnt_width)(buf_bytes - in_bytes)
      up_ready.assign(has_room & ~saw_last)
      up_xact = up_ready & up_valid

      # Emit an element once a full stride slot is buffered.
      client_valid = count >= UInt(cnt_width)(stride_bytes)
      client_xact = client_valid & client_ready

      # The burst's final word sets `saw_last`; the emit that drains the buffer
      # to empty terminates the list. These never coincide: emitting needs a
      # slot buffered by a prior cycle's accept, so `after_emit == 0` on an
      # accept cycle is impossible.
      added = Mux(up_xact,
                  UInt(cnt_width)(0), (up.valid_bytes.as_uint(cnt_width) +
                                       UInt(1)(1)).as_uint(cnt_width))
      after_add = (count + added).as_uint(cnt_width)
      after_emit = (after_add -
                    UInt(cnt_width)(stride_bytes)).as_uint(cnt_width)
      set_saw_last = (up_xact & up.last).as_bits()
      is_final_slot = (after_emit == UInt(cnt_width)(0))
      burst_ending = saw_last | set_saw_last
      client_last = client_valid & burst_ending & is_final_slot
      final_emit = client_xact & client_last

      # Append the accepted word at bit offset count*8 (dynamic left shift);
      # then, if we emit this cycle, drop the consumed slot (constant right
      # shift by the stride). The append offset is <= stride_bytes when
      # accepting, so bound the shift index to that range: its high bits are
      # constant 0, which lets constant-propagation prune the upper barrel-
      # shifter stages (synthesis won't infer this bound from the count reg).
      append_off = count.as_bits()[0:offset_width]
      shamt = BitsSignal.concat([append_off,
                                 Bits(3)(0)]).pad_or_truncate(buf_bits)
      word_ext = up.data.pad_or_truncate(buf_bits)
      shifted_word = BitsSignal(
          comb.ShlOp(word_ext.value, shamt.value).result, Bits(buf_bits))
      appended = buffer | Mux(up_xact, Bits(buf_bits)(0), shifted_word)
      drained = appended[stride_bits:buf_bits].pad_or_truncate(buf_bits)
      buffer.assign(
          Mux(final_emit, Mux(client_xact, appended, drained),
              Bits(buf_bits)(0)))

      # count += accepted real bytes (biased 'valid_bytes' + 1); -= stride on
      # emit.
      count.assign(Mux(client_xact, after_add, after_emit))

      saw_last.assign(
          ControlReg(ports.clk, ports.rst, [set_saw_last], [final_emit]))

      tag_reg = up.tag.reg(ports.clk, ports.rst, ce=up_xact, name="tag_reg")
      client_channel, client_ready_sig = ShiftReadGearboxImpl.out.type.wrap(
          {
              "tag": tag_reg,
              "data": buffer[0:output_bitwidth],
              "last": client_last,
          }, client_valid)
      client_ready.assign(client_ready_sig)
      ports.out = client_channel

  return ShiftReadGearboxImpl


def select_read_gearbox(is_list: bool, input_bitwidth: int,
                        output_bitwidth: int):
  """Pick the read-gearbox module for a client of the given kind and width
  relationship. Every gearbox shares the {tag, data, valid_bytes, last} input
  (from `HostMemReadReqSplitter`) and the {tag, data, last} output, so callers
  wire them identically. `ShiftReadGearbox` is the correct-for-everything
  fallback; the others avoid its barrel shifter for regular relationships."""
  if not is_list:
    # A single element starts at bit 0 and never straddles at a bit offset.
    if output_bitwidth <= input_bitwidth:
      return SliceReadGearbox(input_bitwidth, output_bitwidth)
    return ConcatReadGearbox(input_bitwidth, output_bitwidth)
  if output_bitwidth > input_bitwidth:
    # Super-word list: a whole-word-multiple element never straddles.
    if output_bitwidth % input_bitwidth == 0:
      return ConcatReadGearbox(input_bitwidth, output_bitwidth)
    return ShiftReadGearbox(input_bitwidth, output_bitwidth)
  # Sub-word list: a byte-aligned element that divides the word never straddles.
  if input_bitwidth % output_bitwidth == 0 and output_bitwidth % 8 == 0:
    return DepackReadGearbox(input_bitwidth, output_bitwidth)
  return ShiftReadGearbox(input_bitwidth, output_bitwidth)


# Maximum size, in bytes, of a single upstream read request. Reads larger than
# this are split by the requester into multiple requests. The default is a
# conservative PCIe-derived cap (Max_Read_Request_Size tops out at 4096 bytes,
# but root ports often negotiate less); it mirrors kPcieMaxReadRequestBytes in
# the Cosim backend (cpp/lib/backends/Cosim.cpp).
DEFAULT_MAX_READ_REQUEST_BYTES = 64 * 4  # 64 double words

# Maximum size, in bytes, of a single upstream write transaction; an element
# whose write payload is wider is split into multiple <= this-size transactions.
# The default is a conservative PCIe-derived Max-Payload-Size cap.
DEFAULT_MAX_WRITE_PAYLOAD_BYTES = 256


@modparams
def HostMemReadReqSplitter(req_channel_type: Channel,
                           resp_channel_type: Channel, max_chunk_bytes: int):
  """Split oversized host memory read requests into request-sized chunks before
  arbitration and reassemble the per-chunk responses into a single logical
  burst.

  A burst read (`read_list`) can request many more bytes than a single upstream
  read request can carry. This module breaks such a request into
  `max_chunk_bytes`-sized (word-aligned) chunks addressed sequentially from the
  base. Splitting here -- *before* the requests
  are arbitrated onto the shared upstream read channel -- lets each client's
  chunks interleave with other clients' requests, so one large burst does not
  monopolize host memory bandwidth.

  On the response path the per-chunk end-of-list markers are dropped and a
  single burst-final `last` is re-derived from the total transfer length, so the
  gearbox and client see one contiguous response stream identical to an unsplit
  read.

  Only one logical request is in flight at a time (matching the read processor's
  one-outstanding-transaction-per-client model): a new request is not accepted
  until the current burst's chunks have all been issued and its responses have
  fully drained. This will be a performance limiter.
  TODO: make this able to issue >1 one read at a time.

  req_channel_type:  channel of the upstream read request {address, length
    (bytes), tag}.
  resp_channel_type: channel of the upstream response {tag, data, last}.
  max_chunk_bytes:   largest per-chunk byte count; must be > 0 and a multiple of
    the response word size.
  """
  assert max_chunk_bytes > 0

  req_struct = req_channel_type.inner_type
  resp_struct = resp_channel_type.inner_type
  req_fields = dict(req_struct.fields)
  addr_width = req_fields["address"].bitwidth
  length_width = req_fields["length"].bitwidth
  tag_type = req_fields["tag"]
  word_bytes = dict(resp_struct.fields)["data"].bitwidth // 8
  word_shift = clog2(word_bytes)
  words_width = length_width - word_shift
  # The response is augmented with a per-word 'valid_bytes': the number of real
  # bytes in the (possibly partial) final word, biased by -1. A burst word
  # always has >= 1 real byte, so encoding count-1 fits in one fewer bit.
  vb_width = clog2(word_bytes)
  resp_fields = dict(resp_struct.fields)
  resp_out_struct = StructType([
      ("tag", resp_fields["tag"]),
      ("data", resp_fields["data"]),
      ("valid_bytes", UInt(vb_width)),
      ("last", Bits(1)),
  ])
  resp_out_channel_type = Channel(resp_out_struct)

  class HostMemReadReqSplitterImpl(Module):
    clk = Clock()
    rst = Reset()
    req_in = Input(req_channel_type)
    req_out = Output(req_channel_type)
    resp_in = Input(resp_channel_type)
    resp_out = Output(resp_out_channel_type)

    @generator
    def build(ports):
      clk = ports.clk
      rst = ports.rst

      # Burst state shared by the request-splitting and response-reassembly
      # FSMs. One logical request is processed at a time.
      emit_busy = Wire(Bits(1), name="emit_busy")  # issuing chunk requests
      resp_busy = Wire(Bits(1), name="resp_busy")  # responses still draining
      cur_addr = Wire(UInt(addr_width), name="cur_addr")
      remaining = Wire(UInt(length_width), name="remaining")  # req bytes left
      tag_reg = Wire(tag_type, name="tag_reg")
      words_left = Wire(UInt(words_width), name="words_left")  # resp words left

      idle = (~emit_busy) & (~resp_busy)

      # --- Request intake and splitting ---
      req_ready = Wire(Bits(1))
      req_payload, req_valid = ports.req_in.unwrap(req_ready)
      accept = idle & req_valid
      req_ready.assign(accept)

      max_chunk = UInt(length_width)(max_chunk_bytes)
      chunk_len = Mux(remaining > max_chunk, remaining, max_chunk)
      last_chunk = remaining <= max_chunk

      # Round the emitted read length up to a whole word. The reader response
      # is word-granular and 'valid_bytes' still carries the real trailing byte
      # count, so total_words and the reassembled element count are unchanged;
      # this just keeps every read word-aligned for single-flit HostMem
      # transports that reject sub-word read lengths.
      if word_shift == 0:
        chunk_len_out = chunk_len
      else:
        chunk_words = (chunk_len + UInt(length_width)(word_bytes - 1)
                      ).as_bits()[word_shift:].as_uint(words_width)
        chunk_len_out = BitsSignal.concat(
            [chunk_words.as_bits(), Bits(word_shift)(0)]).as_uint(length_width)

      req_out_ch, req_out_ready = req_channel_type.wrap(
          req_struct({
              "address": cur_addr,
              "length": chunk_len_out,
              "tag": tag_reg,
          }), emit_busy)
      ports.req_out = req_out_ch
      chunk_xact = emit_busy & req_out_ready

      emit_busy.assign(
          ControlReg(clk,
                     rst, [accept], [chunk_xact & last_chunk],
                     name="emit_busy_reg"))

      # cur_addr: load base on accept, advance by the chunk on each issue.
      cur_addr_incr = (cur_addr +
                       chunk_len.as_uint(addr_width)).as_uint(addr_width)
      cur_addr.assign(
          Mux(accept, Mux(chunk_xact, cur_addr, cur_addr_incr),
              req_payload.address).reg(clk,
                                       rst,
                                       rst_value=0,
                                       ce=accept | chunk_xact,
                                       name="cur_addr_reg"))

      # remaining: load length on accept, subtract each issued chunk.
      remaining_dec = (remaining - chunk_len).as_uint(length_width)
      remaining.assign(
          Mux(accept, Mux(chunk_xact, remaining, remaining_dec),
              req_payload.length).reg(clk,
                                      rst,
                                      rst_value=0,
                                      ce=accept | chunk_xact,
                                      name="remaining_reg"))

      tag_reg.assign(req_payload.tag.reg(clk, rst, ce=accept, name="tag_reg_r"))

      # --- Response reassembly: re-derive the burst-final 'last' and the byte
      # count of the (possibly partial) final word. Elements need not tile
      # evenly into words, so count words with ceil(length / word_bytes). ---
      total_words = ((req_payload.length + UInt(length_width)(word_bytes - 1)
                     ).as_bits()[word_shift:]).as_uint(words_width)
      # Bytes valid in the final word = length - (total_words - 1) * word_bytes.
      words_before_last = (total_words -
                           UInt(words_width)(1)).as_uint(words_width)
      bytes_before_last = BitsSignal.concat(
          [words_before_last.as_bits(),
           Bits(word_shift)(0)]).as_uint(length_width)
      final_valid_bytes = (req_payload.length - bytes_before_last -
                           UInt(length_width)(1)).as_uint(vb_width).reg(
                               clk, rst, ce=accept, name="final_valid_bytes")
      resp_ready = Wire(Bits(1))
      resp_payload, resp_valid = ports.resp_in.unwrap(resp_ready)
      is_final_word = words_left == UInt(words_width)(1)
      resp_out_ch, resp_out_ready = resp_out_channel_type.wrap(
          resp_out_struct({
              "tag":
                  resp_payload.tag,
              "data":
                  resp_payload.data,
              "valid_bytes":
                  Mux(is_final_word,
                      UInt(vb_width)(word_bytes - 1), final_valid_bytes),
              "last":
                  is_final_word,
          }), resp_valid)
      ports.resp_out = resp_out_ch
      resp_ready.assign(resp_out_ready)
      resp_xact = resp_valid & resp_out_ready

      # words_left: load total on accept, decrement per received word.
      words_dec = (words_left - UInt(words_width)(1)).as_uint(words_width)
      words_left.assign(
          Mux(accept, Mux(resp_xact, words_left, words_dec),
              total_words).reg(clk,
                               rst,
                               rst_value=0,
                               ce=accept | resp_xact,
                               name="words_left_reg"))

      resp_busy.assign(
          ControlReg(clk,
                     rst, [accept], [resp_xact & is_final_word],
                     name="resp_busy_reg"))

  return HostMemReadReqSplitterImpl


def HostmemReadProcessor(
    read_width: int,
    hostmem_module,
    reqs: List[esi._OutputBundleSetter],
    max_read_request_bytes: int = DEFAULT_MAX_READ_REQUEST_BYTES):
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

      # Demux the upstream response frames {tag, data, last} to each client by
      # tag. Each client's stream then flows through a `HostMemReadReqSplitter`
      # (which annotates per-word 'valid_bytes' and the burst-final 'last') into
      # the leaf gearbox chosen by `select_read_gearbox`.
      demux = esi.TaggedDemux(len(reqs), upstream_resp_channel.type)(
          clk=ports.clk, rst=ports.rst, in_=upstream_resp_channel)

      word_bytes = read_width // 8
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
        client_type = resp_type.inner_type
        is_list = isinstance(client_type, Window)

        # A read_list response is a parallel window over
        # struct{tag, data: list<element>} (num_items=1), lowering to
        # struct{tag, data: element, data_size, last}; a single read carries the
        # element directly. Pull the element width out of whichever shape.
        if is_list:
          lowered = client_type.lowered_type
          lowered_fields = dict(lowered.fields)
          element_type = lowered_fields["data"]
          element_bits = element_type.bitwidth
          data_size_type = lowered_fields["data_size"]
          if element_bits == 0:
            raise ValueError("read_list element type cannot be zero-width.")
        else:
          if client_type.data.bitwidth == 0:
            raise ValueError("Client data type cannot be zero-width. Use a "
                             "single-bit type if no data is needed.")
          element_bits = client_type.data.bitwidth
        # Elements are packed contiguously in host memory at their natural byte
        # size, independent of the engine word width.
        elem_stride_bytes = (element_bits + 7) // 8

        # Both single-message and list reads flow demux -> splitter -> gearbox
        # with a uniform {tag, data, valid_bytes, last} interface. The splitter
        # chunks oversized requests (so even a wide single element is
        # request-chunked) and annotates each word with 'valid_bytes' plus the
        # burst-final 'last'; `select_read_gearbox` picks the leaf gearbox for
        # this (is_list, read_width, element_bits). 'splitter_resp' breaks the
        # request/response construction cycle (the client request is derived
        # from the gearbox's response bundle).
        max_chunk_bytes = (max_read_request_bytes // word_bytes) * word_bytes
        gearbox_mod = select_read_gearbox(is_list, read_width, element_bits)
        splitter_resp = Wire(gearbox_mod.in_.type)
        gearbox = gearbox_mod(clk=ports.clk, rst=ports.rst, in_=splitter_resp)

        if is_list:
          # Propagate 'last', then re-wrap the element as the response window.
          client_resp_channel = gearbox.out.transform(
              lambda m, lowered=lowered, element_type=element_type,
              data_size_type=data_size_type, client_type=client_type:
              client_type.wrap(
                  lowered({
                      "tag": m.tag,
                      "data": m.data.bitcast(element_type),
                      "data_size": data_size_type(0),
                      "last": m.last,
                  })))
          client_bundle, froms = client.type.pack(resp=client_resp_channel)
          client_req = froms["req"]
          logical_req = client_req.transform(
              lambda r, idx=idx, elem_stride_bytes=elem_stride_bytes:
              hostmem_module.UpstreamReadReq({
                  "address":
                      r.address,
                  "length": (r.length * UInt(64)
                             (elem_stride_bytes)).as_uint(32),
                  "tag":
                      idx,
              }))
        else:
          # Single-message read: one element; discard the 'last' burst marker.
          client_resp_channel = gearbox.out.transform(
              lambda m, client_type=client_type: client_type({
                  "tag": m.tag,
                  "data": m.data.bitcast(client_type.data)
              }))
          client_bundle, froms = client.type.pack(resp=client_resp_channel)
          client_req = froms["req"]
          logical_req = client_req.transform(
              lambda r, idx=idx, elem_stride_bytes=elem_stride_bytes:
              hostmem_module.UpstreamReadReq({
                  "address": r.address,
                  "length": UInt(32)(elem_stride_bytes),
                  # TODO: Change this once we support tag-rewriting.
                  "tag": idx,
              }))

        splitter = HostMemReadReqSplitter(
            logical_req.type, demuxed_upstream_channel.type,
            max_chunk_bytes)(clk=ports.clk,
                             rst=ports.rst,
                             req_in=logical_req,
                             resp_in=demuxed_upstream_channel)
        splitter_resp.assign(splitter.resp_out)
        tagged_client_req = splitter.req_out

        tagged_client_reqs.append(tagged_client_req)

        # Set the port for the client request.
        setattr(ports, HostmemReadProcessorImpl.reqPortMap[client],
                client_bundle)

      # Assign the multiplexed read request to the upstream request. Use the
      # list-aware, pipelined ChannelArbiter (vs. the combinational ChannelMux)
      # for a registered N:1 mux that closes timing at high client fan-in. Read
      # requests are single-flit, so list-awareness is a no-op here.
      # TODO: Don't release a request until the client is ready to accept
      # the response otherwise the system could deadlock.
      muxed_client_reqs = ChannelArbiter(tagged_client_reqs,
                                         ports.clk,
                                         ports.rst,
                                         telemetry=False)
      upstream_req_channel.assign(muxed_client_reqs)
      HostmemReadProcessorImpl.reqPortMap.clear()

  return HostmemReadProcessorImpl


@modparams
def TaggedWriteGearbox(input_bitwidth: int, output_bitwidth: int,
                       max_burst_bytes: int) -> type["TaggedWriteGearboxImpl"]:
  """Build a gearbox to convert the client data to upstream write chunks.
  Assumes a struct {address, tag, data} and only gearboxes the data. Tag is
  stored separately and the struct is re-assembled later on.

  'max_burst_bytes' caps a single contiguous upstream write transaction (a
  max-payload-size analog): when an element spans more than 'max_burst_bytes',
  its engine words are split into multiple <= 'max_burst_bytes' transactions by
  emitting the framing 'last' at each boundary. 0 disables the cap."""

  if output_bitwidth % 8 != 0:
    raise ValueError("Output bitwidth must be a multiple of 8.")
  input_pad_bits = 0
  if input_bitwidth % 8 != 0:
    input_pad_bits = 8 - (input_bitwidth % 8)
  input_padded_bitwidth = input_bitwidth + input_pad_bits

  # Number of engine words per capped transaction (0 = uncapped).
  max_burst_words = (max_burst_bytes //
                     (output_bitwidth // 8)) if max_burst_bytes else 0
  if max_burst_words:
    assert (max_burst_words & (max_burst_words - 1)) == 0, \
        "max_burst_bytes / (output_bitwidth // 8) must be a power of two"

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
            ("last", Bits(1)),
        ]))

    num_chunks = ceil(input_padded_bitwidth / output_bitwidth)

    @generator
    def build(ports):
      upstream_ready = Wire(Bits(1))
      ready_for_client = Wire(Bits(1))
      client_tag_and_data, client_valid = ports.in_.unwrap(ready_for_client)
      client_data = client_tag_and_data.data
      if input_pad_bits > 0:
        client_data = client_data.pad_or_truncate(input_padded_bitwidth)
      client_xact = ready_for_client & client_valid
      input_bitwidth_bytes = input_padded_bitwidth // 8
      output_bitwidth_bytes = output_bitwidth // 8

      # Determine if gearboxing is necessary and whether it needs to be
      # gearboxed up or just sliced down.
      if output_bitwidth == input_padded_bitwidth:
        upstream_data_bits = client_data
        upstream_valid = client_valid
        ready_for_client.assign(upstream_ready)
        tag = client_tag_and_data.tag
        address = client_tag_and_data.address
        valid_bytes = Bits(8)(input_bitwidth_bytes)
        last = Bits(1)(1)
      elif output_bitwidth > input_padded_bitwidth:
        upstream_data_bits = client_data.as_bits(output_bitwidth)
        upstream_valid = client_valid
        ready_for_client.assign(upstream_ready)
        tag = client_tag_and_data.tag
        address = client_tag_and_data.address
        valid_bytes = Bits(8)(input_bitwidth_bytes)
        last = Bits(1)(1)
      else:
        # Create registers equal to the number of upstream transactions needed
        # to complete the transmission.
        num_chunks = TaggedWriteGearboxImpl.num_chunks
        num_chunks_idx_bitwidth = clog2(num_chunks)
        if input_padded_bitwidth % output_bitwidth == 0:
          padding_numbits = 0
        else:
          padding_numbits = output_bitwidth - (input_padded_bitwidth %
                                               output_bitwidth)
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
        elem_end = counter.out == (num_chunks - 1)
        valid_bytes = Mux(elem_end,
                          Bits(8)(output_bitwidth_bytes),
                          Bits(8)((output_bitwidth - padding_numbits) // 8))
        if max_burst_words and num_chunks > max_burst_words:
          # Max-payload-size cap: end the upstream write transaction at the
          # element end OR every max_burst_words engine words, whichever comes
          # first, so a wide element's write is split into <= max_burst_bytes
          # transactions. Each word keeps its own sequential address; only the
          # transaction-framing 'last' changes.
          burst_shift = clog2(max_burst_words)
          burst_end = counter.out.as_bits()[:burst_shift].and_reduce()
          last = elem_end | burst_end
        else:
          last = elem_end

      upstream_channel, upstrm_ready_sig = TaggedWriteGearboxImpl.out.type.wrap(
          {
              "address": address,
              "tag": tag,
              "data": upstream_data_bits,
              "valid_bytes": valid_bytes,
              "last": last,
          }, upstream_valid)
      upstream_ready.assign(upstrm_ready_sig)
      ports.out = upstream_channel

  return TaggedWriteGearboxImpl


@modparams
def EmitEveryN(message_type: Type, N: int) -> type['EmitEveryNImpl']:
  """Emit (forward) one message for every N input messages. The emitted message
  is the last one of the N received. N must be >= 1."""

  if N < 1:
    raise ValueError("N must be >= 1")

  class EmitEveryNImpl(Module):
    clk = Clock()
    rst = Reset()
    in_ = InputChannel(message_type)
    out = OutputChannel(message_type)

    @generator
    def build(ports):
      ready_for_in = Wire(Bits(1))
      in_data, in_valid = ports.in_.unwrap(ready_for_in)
      xact = in_valid & ready_for_in

      # Fast path: N == 1 -> pass-through.
      if N == 1:
        out_chan, out_ready = EmitEveryNImpl.out.type.wrap(in_data, in_valid)
        ready_for_in.assign(out_ready)
        ports.out = out_chan
        return

      counter_width = clog2(N)
      counter_clear = Wire(Bits(1))
      counter = Counter(counter_width)(clk=ports.clk,
                                       rst=ports.rst,
                                       increment=xact,
                                       clear=counter_clear)

      # Capture last message of the group.
      last_msg = in_data.reg(ports.clk, ports.rst, ce=xact, name="last_msg")
      # Clear the counter.
      hit_last = (counter.out == UInt(counter_width)(N - 1)) & xact
      counter_clear.assign(hit_last)

      emit_accepted = Wire(Bits(1))
      out_valid = ControlReg(ports.clk, ports.rst, [hit_last], [emit_accepted])

      out_chan, out_ready = EmitEveryNImpl.out.type.wrap(last_msg, out_valid)
      # Stall input while waiting for downstream to accept the aggregated output.
      ready_for_in.assign(~(out_valid & ~out_ready))
      emit_accepted.assign(out_valid & out_ready)  # Output consumed downstream.

      ports.out = out_chan

  return EmitEveryNImpl


def HostMemWriteProcessor(
    write_width: int,
    hostmem_module,
    reqs: List[esi._OutputBundleSetter],
    max_write_payload_bytes: int = DEFAULT_MAX_WRITE_PAYLOAD_BYTES
) -> type["HostMemWriteProcessorImpl"]:
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
      clk = ports.clk
      rst = ports.rst

      # Width of the frame's 'data_size' field: log2 of the number of bytes per
      # engine word. It holds (valid_bytes - 1) for the final (possibly partial)
      # word of a write.
      size_width = clog2(write_width // 8)

      # If there's no write clients, just create a no-op write bundle
      if len(reqs) == 0:
        req, _ = Channel(hostmem_module.UpstreamWriteReq).wrap(
            {
                "address": 0,
                "tag": 0,
                "data": 0,
                "data_size": 0,
                "last": 0,
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
        input_flit_ack = Wire(upstream_ack_tag.type)

        if isinstance(client_type, Window):
          # Windowed (list) write: the client streams a list of elements to be
          # written to sequential addresses from a base. Lowered frame:
          # struct{address, tag, data: elem[num_items], data_size, last}. One
          # element per frame (num_items=1 here).
          bundle_sig, wfroms = req.type.pack(ackTag=input_flit_ack)
          windowed_req = wfroms["req"]
          lowered = client_type.lowered_type
          array_type = dict(lowered.fields)["data"]
          element_bits = array_type.element_type.bitwidth
          # Elements are packed contiguously in host memory at their natural
          # byte size, independent of the engine word width (matches the
          # read_list path). Each per-element write is byte-enabled via
          # data_size, so a sub-word element writes only its own bytes.
          elem_stride = (element_bits + 7) // 8

          gearbox_mod = TaggedWriteGearbox(element_bits, write_width,
                                           max_write_payload_bytes)
          gearbox_in_type = gearbox_mod.in_.type.inner_type

          # Unwrap the window frames; compute a base+offset address from a
          # per-burst element counter (reset after each burst's final element).
          ready_for_frame = Wire(Bits(1))
          frame_win, frame_valid = windowed_req.unwrap(ready_for_frame)
          frame = frame_win.unwrap()
          frame_xact = frame_valid & ready_for_frame
          elem_clear = Wire(Bits(1))
          elem_counter = Counter(64)(clk=ports.clk,
                                     rst=ports.rst,
                                     clear=elem_clear,
                                     increment=frame_xact)
          elem_clear.assign(frame_xact & frame["last"])
          elem_addr = (frame["address"] +
                       elem_counter.out * UInt(64)(elem_stride)).as_uint(64)
          gearbox_in_chan, gearbox_in_ready = Channel(gearbox_in_type).wrap(
              gearbox_in_type({
                  "tag": frame["tag"],
                  "address": elem_addr,
                  "data": frame["data"][0].bitcast(gearbox_in_type.data),
              }), frame_valid)
          ready_for_frame.assign(gearbox_in_ready)
          gearbox = gearbox_mod(clk=ports.clk,
                                rst=ports.rst,
                                in_=gearbox_in_chan)
        else:
          # Single-message write.
          write_req_bundle_type = esi.HostMem.write_req_bundle_type(
              client_type.data)
          bundle_sig, sfroms = write_req_bundle_type.pack(ackTag=input_flit_ack)
          gearbox_mod = TaggedWriteGearbox(client_type.data.bitwidth,
                                           write_width, max_write_payload_bytes)
          gearbox_in_type = gearbox_mod.in_.type.inner_type
          bitcast_client_req = sfroms["req"].transform(
              lambda m, git=gearbox_in_type: git({
                  "tag": m.tag,
                  "address": m.address,
                  "data": m.data.bitcast(git.data)
              }))
          gearbox = gearbox_mod(clk=ports.clk,
                                rst=ports.rst,
                                in_=bitcast_client_req)

        write_channels.append(
            gearbox.out.transform(
                lambda m, idx=idx: hostmem_module.UpstreamWriteReq({
                    "address":
                        m.address,
                    "tag":
                        idx,
                    "data":
                        m.data,
                    "data_size": (m.valid_bytes.as_uint() - UInt(8)
                                  (1)).as_bits()[:size_width],
                    "last":
                        m.last,
                })))

        # Count the number of acks received from hostmem for this client
        # and only send one back to the client per input.
        ack_every_n = EmitEveryN(upstream_ack_tag.type, gearbox_mod.num_chunks)(
            clk=clk, rst=rst, in_=demuxed_acks.get_out(idx))
        input_flit_ack.assign(ack_every_n.out)

        # Set the port for the client request.
        setattr(ports, HostMemWriteProcessorImpl.reqPortMap[req], bundle_sig)

      # Multiplex the write requests onto the single upstream channel with the
      # list-aware, pipelined ChannelArbiter (matching the read side). A real
      # windowed write (multi-word client flits) engages the arbiter's list-
      # awareness -- via the frame's 'last' -- to keep a client's words
      # contiguous; single-word (<= engine width) clients emit one message per
      # word, for which single-flit arbitration is correct.
      muxed_write_channel = ChannelArbiter(write_channels,
                                           ports.clk,
                                           ports.rst,
                                           telemetry=False)
      upstream_req_channel.assign(muxed_write_channel)

  return HostMemWriteProcessorImpl


@modparams
def ChannelHostMem(
    read_width: int,
    write_width: int,
    max_read_request_bytes: int = DEFAULT_MAX_READ_REQUEST_BYTES,
    max_write_payload_bytes: int = DEFAULT_MAX_WRITE_PAYLOAD_BYTES
) -> typing.Type['ChannelHostMemImpl']:

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
                    ("last", Bits(1)),
                ])),
        ]))

    if write_width % 8 != 0:
      raise ValueError("Write width must be a multiple of 8.")
    UpstreamWriteReq = StructType([
        ("address", UInt(64)),
        ("tag", UInt(8)),
        ("data", Bits(write_width)),
        ("data_size", Bits(clog2(write_width // 8))),
        ("last", Bits(1)),
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
      read_reqs = [
          req for req in bundles.to_client_reqs
          if req.port in ('read', 'read_list')
      ]
      read_proc_module = HostmemReadProcessor(read_width, ChannelHostMemImpl,
                                              read_reqs, max_read_request_bytes)
      read_proc = read_proc_module(clk=ports.clk, rst=ports.rst)
      ports.read = read_proc.upstream
      for req in read_reqs:
        req.assign(getattr(read_proc, read_proc_module.reqPortMap[req]))

      # The write side.
      write_reqs = [
          req for req in bundles.to_client_reqs if req.port == 'write'
      ]
      write_proc_module = HostMemWriteProcessor(write_width, ChannelHostMemImpl,
                                                write_reqs,
                                                max_write_payload_bytes)
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


def _resolve_engine_pair(path: str) -> Tuple[Callable, Callable]:
  """Resolve a dotted Python import path to a
  `(to_host_engine_gen, from_host_engine_gen)` tuple, used to override the
  default engine pair for a specific service request.

  The path may point at either:
    - a module-level 2-tuple attribute, e.g.
      `"mypkg.mymod.MyEnginePair"` where `MyEnginePair` is
      `(MyToHost, MyFromHost)`; or
    - a zero-arg factory callable returning such a tuple.
  """
  import importlib
  if not isinstance(path, str):
    raise TypeError(
        "Engine override path must be a dotted 'pkg.mod.attr' string; "
        f"got {type(path).__name__}")
  module_path, _, attr_path = path.rpartition(".")
  if not module_path or not attr_path:
    raise ValueError(
        "Engine override path must be a dotted 'pkg.mod.attr' string; "
        f"got {path!r}")
  obj = importlib.import_module(module_path)
  for part in attr_path.split("."):
    obj = getattr(obj, part)
  if callable(obj):
    obj = obj()
  if not (isinstance(obj, tuple) and len(obj) == 2):
    raise TypeError(
        f"Engine override {path!r} must resolve to a 2-tuple "
        f"(to_host_engine_gen, from_host_engine_gen); got {type(obj).__name__}")
  if not (callable(obj[0]) and callable(obj[1])):
    raise TypeError(
        f"Engine override {path!r} must resolve to a 2-tuple of callables; got "
        f"({type(obj[0]).__name__}, {type(obj[1]).__name__})")
  return obj


def ChannelEngineService(
    to_host_engine_gen: Callable,
    from_host_engine_gen: Callable) -> type['ChannelEngineService']:
  """Returns a channel service implementation which calls
  to_host_engine_gen(<client_type>) or from_host_engine_gen(<client_type>) to
  generate the to_host and from_host engines for each channel. Does not support
  engines which can service multiple clients at once.

  Individual service requests may override the default engine pair by passing
  `options={"engine": "pkg.mod.attr"}` at the service-request call site (e.g.
  `HostComms.some_bundle(AppID(...), options={"engine": "..."})`). The path
  is resolved by `_resolve_engine_pair` and must yield a
  `(to_host_engine_gen, from_host_engine_gen)` tuple with the same call shape
  as the defaults; the override applies to every channel of that request's
  bundle.
  """

  class ChannelEngineService(esi.ServiceImplementation):
    """Service implementation which services the clients via a per-channel DMA
    engine."""

    clk = Clock()
    rst = Reset()

    @generator
    def build(ports, bundles: esi._ServiceGeneratorBundles):
      clk = ports.clk
      rst = ports.rst

      def build_engine_appid(client_appid: List[esi.AppID],
                             channel_name: str) -> str:
        appid_strings = [str(appid) for appid in client_appid]
        return f"{'_'.join(appid_strings)}.{channel_name}"

      def build_engine(bc: BundledChannel,
                       bundle_to_host_gen: Callable,
                       bundle_from_host_gen: Callable,
                       input_channel=None) -> Type:
        idbase = build_engine_appid(bundle.client_name, bc.name)
        eng_appid = esi.AppID(idbase)
        # DMA engines require at least 1 byte of data; substitute Bits(8)
        # for zero-width (void) channel types so the engine never sees a
        # zero-length transfer.
        engine_client_type = bc.channel.inner_type
        is_void = (engine_client_type.bitwidth == 0)
        if is_void:
          engine_client_type = Bits(8)
        if bc.direction == ChannelDirection.FROM:
          engine_mod = bundle_to_host_gen(engine_client_type)
        else:
          engine_mod = bundle_from_host_gen(engine_client_type)
        eng_inputs = {
            "clk": ports.clk,
            "rst": ports.rst,
        }
        eng_details: Dict[str, object] = {"engine_inst": eng_appid}
        if input_channel is not None:
          # For void channels, widen the 0-bit input to the 8-bit
          # placeholder the engine expects.
          if is_void:
            input_channel = input_channel.transform(lambda _: Bits(8)(0))
          if (engine_mod.input_channel.type.signaling
              != input_channel.type.signaling):
            input_channel = input_channel.buffer(
                clk,
                rst,
                stages=1,
                output_signaling=engine_mod.input_channel.type.signaling)
          eng_inputs["input_channel"] = input_channel
        if hasattr(engine_mod, "mmio"):
          mmio_appid = esi.AppID(idbase + ".mmio")
          eng_inputs["mmio"] = esi.MMIO.read_write(mmio_appid)
          eng_details["mmio"] = mmio_appid
        if hasattr(engine_mod, "hostmem_write"):
          eng_inputs["hostmem_write"] = esi.HostMem.write_from_bundle(
              esi.AppID(idbase + ".hostmem_write"),
              engine_mod.hostmem_write.type)
        if hasattr(engine_mod, "hostmem_read"):
          eng_inputs["hostmem_read"] = esi.HostMem.read_from_bundle(
              esi.AppID(idbase + ".hostmem_read"), engine_mod.hostmem_read.type)
        engine = engine_mod(appid=eng_appid, **eng_inputs)
        engine_rec = bundles.emit_engine(engine, details=eng_details)
        engine_rec.add_record(bundle, {bc.name: {}})
        return engine

      for bundle in bundles.to_client_reqs:
        # Per-request engine override: if the client's service request carries
        # an `"engine"` option, use that engine pair instead of the defaults
        # for every channel of this bundle. This is purely a hardware-side
        # substitution.
        engine_override = bundle.options.get("engine")
        if engine_override is None:
          bundle_to_host_gen = to_host_engine_gen
          bundle_from_host_gen = from_host_engine_gen
        else:
          bundle_to_host_gen, bundle_from_host_gen = _resolve_engine_pair(
              engine_override)

        bundle_type = bundle.type
        to_channels = {}
        # Create a DMA engine for each channel headed TO the client (from the host).
        for bc in bundle_type.channels:
          if bc.direction == ChannelDirection.TO:
            engine = build_engine(bc, bundle_to_host_gen, bundle_from_host_gen)
            out_chan = engine.output_channel
            # For void channels, narrow the 8-bit placeholder back to 0-bit.
            if bc.channel.inner_type.bitwidth == 0:
              out_chan = out_chan.transform(lambda _: Bits(0)(0))
            to_channels[bc.name] = out_chan

        client_bundle_sig, froms = bundle_type.pack(**to_channels)
        bundle.assign(client_bundle_sig)

        # Create a DMA engine for each channel headed FROM the client (to the host).
        for bc in bundle_type.channels:
          if bc.direction == ChannelDirection.FROM:
            build_engine(bc, bundle_to_host_gen, bundle_from_host_gen,
                         froms[bc.name])

  return ChannelEngineService
