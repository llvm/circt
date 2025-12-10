# ===- esitester.py - accelerator for testing ESI functionality -----------===//
#
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//
#
#  This accelerator is intended to eventually grow into a full ESI accelerator
#  test image. It will be used both to test system functionality and system
#  performance. The corresponding software appliciation in the ESI runtime and
#  the ESI cosim. Where this should live longer-term is a unclear.
#
# ===----------------------------------------------------------------------===//

# REQUIRES: esi-runtime, esi-cosim, rtl-sim, esitester
# RUN: rm -rf %t
# RUN: mkdir %t && cd %t

# Test cosim without DMA.
# RUN: %PYTHON% %s %t cosim 2>&1
# RUN: esi-cosim.py --source %t -- esitester -v cosim env callback -i 5 | FileCheck %s
# RUN: esi-cosim.py --source %t -- esitester cosim env streaming_add | FileCheck %s --check-prefix=STREAMING_ADD
# RUN: esi-cosim.py --source %t -- esitester cosim env streaming_add -t | FileCheck %s --check-prefix=STREAMING_ADD
# RUN: ESI_COSIM_MANIFEST_MMIO=1 esi-cosim.py --source %t -- esiquery cosim env info
# RUN: esi-cosim.py --source %t -- esiquery cosim env telemetry | FileCheck %s --check-prefix=TELEMETRY

# Now test the DMA engines.
# RUN: %PYTHON% %s %t cosim_dma 2>&1
# RUN: esi-cosim.py --source %t -- esitester cosim env hostmem
# RUN: esi-cosim.py --source %t -- esitester cosim env dma -w -r
# RUN: esi-cosim.py --source %t -- esiquery cosim env telemetry | FileCheck %s --check-prefix=TELEMETRY

import sys
from typing import Type

import pycde.esi as esi
from pycde import Clock, Module, Reset, System, generator, modparams
from pycde.bsp import get_bsp
from pycde.common import AppID, Constant, InputChannel, Output, OutputChannel
from pycde.constructs import ControlReg, Counter, NamedWire, Reg, Wire
from pycde.module import Metadata
from pycde.types import Bits, Channel, ChannelSignaling, UInt

# CHECK: [CONNECT] connecting to backend

# STREAMING_ADD: Streaming add test results:
# STREAMING_ADD:   input[0]=222709 + 5 = 222714 (expected 222714)
# STREAMING_ADD:   input[1]=894611 + 5 = 894616 (expected 894616)
# STREAMING_ADD:   input[2]=772894 + 5 = 772899 (expected 772899)
# STREAMING_ADD:   input[3]=429150 + 5 = 429155 (expected 429155)
# STREAMING_ADD:   input[4]=629806 + 5 = 629811 (expected 629811)
# STREAMING_ADD: Streaming add test passed

# TELEMETRY: ********************************
# TELEMETRY: * Telemetry
# TELEMETRY: ********************************

# TELEMETRY: fromhostdma[32].fromHostCycles: 0
# TELEMETRY: fromhostdma[64].fromHostCycles: 0
# TELEMETRY: fromhostdma[128].fromHostCycles: 0
# TELEMETRY: fromhostdma[256].fromHostCycles: 0
# TELEMETRY: fromhostdma[512].fromHostCycles: 0
# TELEMETRY: fromhostdma[534].fromHostCycles: 0
# TELEMETRY: fromhostdma_0[512].fromHostCycles: 0
# TELEMETRY: fromhostdma_1[512].fromHostCycles: 0
# TELEMETRY: fromhostdma_2[512].fromHostCycles: 0
# TELEMETRY: readmem[32].addrCmdCycles: 0
# TELEMETRY: readmem[32].addrCmdIssued: 0
# TELEMETRY: readmem[32].addrCmdResponses: 0
# TELEMETRY: readmem[32].lastReadLSB: 0
# TELEMETRY: readmem[64].addrCmdCycles: 0
# TELEMETRY: readmem[64].addrCmdIssued: 0
# TELEMETRY: readmem[64].addrCmdResponses: 0
# TELEMETRY: readmem[64].lastReadLSB: 0
# TELEMETRY: readmem[128].addrCmdCycles: 0
# TELEMETRY: readmem[128].addrCmdIssued: 0
# TELEMETRY: readmem[128].addrCmdResponses: 0
# TELEMETRY: readmem[128].lastReadLSB: 0
# TELEMETRY: readmem[256].addrCmdCycles: 0
# TELEMETRY: readmem[256].addrCmdIssued: 0
# TELEMETRY: readmem[256].addrCmdResponses: 0
# TELEMETRY: readmem[256].lastReadLSB: 0
# TELEMETRY: readmem[512].addrCmdCycles: 0
# TELEMETRY: readmem[512].addrCmdIssued: 0
# TELEMETRY: readmem[512].addrCmdResponses: 0
# TELEMETRY: readmem[512].lastReadLSB: 0
# TELEMETRY: readmem[534].addrCmdCycles: 0
# TELEMETRY: readmem[534].addrCmdIssued: 0
# TELEMETRY: readmem[534].addrCmdResponses: 0
# TELEMETRY: readmem[534].lastReadLSB: 0
# TELEMETRY: readmem_0[512].addrCmdCycles: 0
# TELEMETRY: readmem_0[512].addrCmdIssued: 0
# TELEMETRY: readmem_0[512].addrCmdResponses: 0
# TELEMETRY: readmem_0[512].lastReadLSB: 0
# TELEMETRY: readmem_1[512].addrCmdCycles: 0
# TELEMETRY: readmem_1[512].addrCmdIssued: 0
# TELEMETRY: readmem_1[512].addrCmdResponses: 0
# TELEMETRY: readmem_1[512].lastReadLSB: 0
# TELEMETRY: readmem_2[512].addrCmdCycles: 0
# TELEMETRY: readmem_2[512].addrCmdIssued: 0
# TELEMETRY: readmem_2[512].addrCmdResponses: 0
# TELEMETRY: readmem_2[512].lastReadLSB: 0
# TELEMETRY: tohostdma[32].toHostCycles: 0
# TELEMETRY: tohostdma[32].totalWrites: 0
# TELEMETRY: tohostdma[64].toHostCycles: 0
# TELEMETRY: tohostdma[64].totalWrites: 0
# TELEMETRY: tohostdma[128].toHostCycles: 0
# TELEMETRY: tohostdma[128].totalWrites: 0
# TELEMETRY: tohostdma[256].toHostCycles: 0
# TELEMETRY: tohostdma[256].totalWrites: 0
# TELEMETRY: tohostdma[512].toHostCycles: 0
# TELEMETRY: tohostdma[512].totalWrites: 0
# TELEMETRY: tohostdma[534].toHostCycles: 0
# TELEMETRY: tohostdma[534].totalWrites: 0
# TELEMETRY: tohostdma_0[512].toHostCycles: 0
# TELEMETRY: tohostdma_0[512].totalWrites: 0
# TELEMETRY: tohostdma_1[512].toHostCycles: 0
# TELEMETRY: tohostdma_1[512].totalWrites: 0
# TELEMETRY: tohostdma_2[512].toHostCycles: 0
# TELEMETRY: tohostdma_2[512].totalWrites: 0
# TELEMETRY: writemem[32].addrCmdCycles: 0
# TELEMETRY: writemem[32].addrCmdIssued: 0
# TELEMETRY: writemem[32].addrCmdResponses: 0
# TELEMETRY: writemem[64].addrCmdCycles: 0
# TELEMETRY: writemem[64].addrCmdIssued: 0
# TELEMETRY: writemem[64].addrCmdResponses: 0
# TELEMETRY: writemem[128].addrCmdCycles: 0
# TELEMETRY: writemem[128].addrCmdIssued: 0
# TELEMETRY: writemem[128].addrCmdResponses: 0
# TELEMETRY: writemem[256].addrCmdCycles: 0
# TELEMETRY: writemem[256].addrCmdIssued: 0
# TELEMETRY: writemem[256].addrCmdResponses: 0
# TELEMETRY: writemem[512].addrCmdCycles: 0
# TELEMETRY: writemem[512].addrCmdIssued: 0
# TELEMETRY: writemem[512].addrCmdResponses: 0
# TELEMETRY: writemem[534].addrCmdCycles: 0
# TELEMETRY: writemem[534].addrCmdIssued: 0
# TELEMETRY: writemem[534].addrCmdResponses: 0
# TELEMETRY: writemem_0[512].addrCmdCycles: 0
# TELEMETRY: writemem_0[512].addrCmdIssued: 0
# TELEMETRY: writemem_0[512].addrCmdResponses: 0
# TELEMETRY: writemem_1[512].addrCmdCycles: 0
# TELEMETRY: writemem_1[512].addrCmdIssued: 0
# TELEMETRY: writemem_1[512].addrCmdResponses: 0
# TELEMETRY: writemem_2[512].addrCmdCycles: 0
# TELEMETRY: writemem_2[512].addrCmdIssued: 0
# TELEMETRY: writemem_2[512].addrCmdResponses: 0


class CallbackTest(Module):
  """Call a function on the host when an MMIO write is received at offset
    0x10."""

  # CHECK: 0
  # CHECK: 1
  # CHECK: 2
  # CHECK: 3
  # CHECK: 4

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    clk = ports.clk
    rst = ports.rst

    mmio_bundle = esi.MMIO.read_write(appid=AppID("cmd"))
    data_resp_chan = Wire(Channel(Bits(64)))
    mmio_cmd_chan = mmio_bundle.unpack(data=data_resp_chan)["cmd"]
    cb_trigger, mmio_cmd_chan_fork = mmio_cmd_chan.fork(clk=clk, rst=rst)

    data_resp_chan.assign(
        mmio_cmd_chan_fork.transform(lambda cmd: Bits(64)(cmd.data)))

    cb_trigger_ready = Wire(Bits(1))
    cb_trigger_cmd, cb_trigger_valid = cb_trigger.unwrap(cb_trigger_ready)
    trigger = cb_trigger_valid & (cb_trigger_cmd.offset == UInt(32)(0x10))
    data_reg = cb_trigger_cmd.data.reg(clk, rst, ce=trigger)
    cb_chan, cb_trigger_ready_sig = Channel(Bits(64)).wrap(
        data_reg, trigger.reg(clk, rst))
    cb_trigger_ready.assign(cb_trigger_ready_sig)
    esi.CallService.call(AppID("cb"), cb_chan, Bits(0))


class LoopbackInOutAdd(Module):
  """Exposes a function which adds the 'add_amt' constant to the argument."""

  clk = Clock()
  rst = Reset()

  add_amt = Constant(UInt(16), 11)

  @generator
  def construct(ports):
    loopback = Wire(Channel(UInt(16), signaling=ChannelSignaling.FIFO))
    args = esi.FuncService.get_call_chans(AppID("add"),
                                          arg_type=UInt(24),
                                          result=loopback)

    ready = Wire(Bits(1))
    data, valid = args.unwrap(ready)
    plus7 = data + LoopbackInOutAdd.add_amt.value
    data_chan, data_ready = Channel(UInt(16), ChannelSignaling.ValidReady).wrap(
        plus7.as_uint(16), valid)
    data_chan_buffered = data_chan.buffer(ports.clk, ports.rst, 1,
                                          ChannelSignaling.FIFO)
    ready.assign(data_ready)
    loopback.assign(data_chan_buffered)


class StreamingAdder(Module):
  """Exposes a function which has an argument of struct {add_amt, list<uint32>}.
  It then adds add_amt to each element of list and returns the resulting list.
  """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    from pycde.types import StructType, List, Window

    # Define the argument type: struct { add_amt: UInt(32), list: List<UInt(32)> }
    arg_struct_type = StructType([("add_amt", UInt(32)),
                                  ("input", List(UInt(32)))])

    # Create a windowed version of the argument struct for streaming
    arg_window_type = Window.default_of(arg_struct_type)

    # Result is also a List
    result_type = List(UInt(32))
    result_window_type = Window.default_of(result_type)

    result_chan = Wire(Channel(result_window_type))
    args = esi.FuncService.get_call_chans(AppID("streaming_add"),
                                          arg_type=arg_window_type,
                                          result=result_chan)

    # Unwrap the argument channel
    ready = Wire(Bits(1))
    arg_data, arg_valid = args.unwrap(ready)

    # Unwrap the window to get the struct/union
    arg_unwrapped = arg_data.unwrap()

    # Extract add_amt and list from the struct
    add_amt = arg_unwrapped["add_amt"]
    input_int = arg_unwrapped["input"]

    result_int = (add_amt + input_int).as_uint(32)
    result_window = result_window_type.wrap(
        result_window_type.lowered_type({
            "data": result_int,
            "last": arg_unwrapped.last
        }))

    # Wrap the result into a channel
    result_chan_internal, result_ready = Channel(result_window_type).wrap(
        result_window, arg_valid)
    ready.assign(result_ready)
    result_chan.assign(result_chan_internal)


@modparams
def MMIOAdd(add_amt: int) -> Type[Module]:

  class MMIOAdd(Module):
    """Exposes an MMIO address space wherein MMIO reads return the <address
        offset into its space> + add_amt."""

    metadata = Metadata(
        name="MMIOAdd",
        misc={"add_amt": add_amt},
    )

    add_amt_const = Constant(UInt(32), add_amt)

    @generator
    def build(ports):
      mmio_read_bundle = esi.MMIO.read(appid=AppID("mmio_client", add_amt))

      address_chan_wire = Wire(Channel(UInt(32)))
      address, address_valid = address_chan_wire.unwrap(1)
      response_data = (address.as_uint() + add_amt).as_bits(64)
      response_chan, response_ready = Channel(Bits(64)).wrap(
          response_data, address_valid)

      address_chan = mmio_read_bundle.unpack(data=response_chan)["offset"]
      address_chan_wire.assign(address_chan)

  return MMIOAdd


@modparams
def AddressCommand(width: int):

  class AddressCommand(Module):
    """Constructs an module which takes MMIO commands and issues host memory
        commands based on those commands. Tracks the number of cycles to issue
        addresses and get all of the expected responses.

        MMIO offsets:
            0x10: Starting address for host memory operations.
            0x18: Number of flits to read/write.
            0x20: Start read/write operation.
        """

    clk = Clock()
    rst = Reset()

    # Number of flits left to issue.
    flits_left = Output(UInt(64))
    # Signal to start the operation.
    command_go = Output(Bits(1))

    # Channel which issues hostmem addresses. Must be transformed into
    # read/write requests by the instantiator.
    hostmem_cmd_address = OutputChannel(UInt(64))

    # Channel which indicates when the read/write operation is done.
    hostmem_cmd_done = InputChannel(Bits(0))

    @generator
    def construct(ports):
      # MMIO command channel setup.
      cmd_chan_wire = Wire(Channel(esi.MMIOReadWriteCmdType))
      resp_ready_wire = Wire(Bits(1))
      cmd, cmd_valid = cmd_chan_wire.unwrap(resp_ready_wire)
      mmio_xact = cmd_valid & resp_ready_wire

      # Write enables.
      start_addr_we = (mmio_xact & cmd.write & (cmd.offset == UInt(32)(0x10)))
      flits_we = mmio_xact & cmd.write & (cmd.offset == UInt(32)(0x18))
      start_op_we = mmio_xact & cmd.write & (cmd.offset == UInt(32)(0x20))
      ports.command_go = start_op_we

      # Registers for start address and number of flits.
      start_addr = cmd.data.as_uint().reg(
          clk=ports.clk,
          rst=ports.rst,
          rst_value=0,
          ce=start_addr_we,
          name="start_addr",
      )
      flits_total = cmd.data.as_uint().reg(
          clk=ports.clk,
          rst=ports.rst,
          rst_value=0,
          ce=flits_we,
          name="flits_total",
      )

      # Response counter.
      responses_incr = Wire(Bits(1))
      responses_cnt = Counter(64)(
          clk=ports.clk,
          rst=ports.rst,
          clear=start_op_we,
          increment=responses_incr,
          instance_name="addr_cmd_responses_cnt",
      )

      operation_done = responses_cnt.out.as_uint() == flits_total
      operation_active = ControlReg(
          clk=ports.clk,
          rst=ports.rst,
          asserts=[start_op_we],
          resets=[operation_done],
          name="operation_active",
      )
      # Cycle counter while active.
      cycles_cnt = Counter(64)(
          clk=ports.clk,
          rst=ports.rst,
          clear=start_op_we,
          increment=operation_active,
          instance_name="addr_cmd_cycle_counter",
      )
      # Latch final cycle count at completion.
      final_cycles = Reg(
          UInt(64),
          clk=ports.clk,
          rst=ports.rst,
          rst_value=0,
          ce=operation_done,
          name="addr_cmd_cycles",
      )
      final_cycles.assign(cycles_cnt.out.as_uint())

      # Issue counter.
      issue_incr = Wire(Bits(1))
      issue_cnt = Counter(64)(
          clk=ports.clk,
          rst=ports.rst,
          clear=start_op_we,
          increment=issue_incr,
      )

      # Increment by number of bytes per flit, rounded up to nearest 32
      # bits (double word).
      incr_bytes = UInt(64)(((width + 31) // 32) * 4)

      # Generate current address.
      current_addr = (start_addr +
                      (issue_cnt.out.as_uint() * incr_bytes)).as_uint(64)

      # Valid when active and still have flits to issue.
      addr_valid = operation_active & (issue_cnt.out.as_uint() < flits_total)
      addr_chan, addr_ready = Channel(UInt(64),
                                      ChannelSignaling.ValidReady).wrap(
                                          current_addr, addr_valid)
      issue_xact = addr_valid & addr_ready
      issue_incr.assign(issue_xact)

      # Consume hostmem_cmd_done (Bits(0) channel) for completed responses.
      _, done_valid = ports.hostmem_cmd_done.unwrap(Bits(1)(1))
      responses_incr.assign(done_valid)

      # flits_left = total - responses received.
      flits_left_val = (flits_total - responses_cnt.out.as_uint()).as_uint(64)
      ports.flits_left = flits_left_val  # direct assignment

      # Drive output channel.
      ports.hostmem_cmd_address = addr_chan  # direct assignment

      # MMIO read response: return flits_left.
      response_data = flits_left_val.as_bits(64)
      response_chan, response_ready = Channel(Bits(64)).wrap(
          response_data, cmd_valid)
      resp_ready_wire.assign(response_ready)

      mmio_rw = esi.MMIO.read_write(appid=AppID("cmd", width))
      mmio_rw_cmd_chan = mmio_rw.unpack(data=response_chan)["cmd"]
      cmd_chan_wire.assign(mmio_rw_cmd_chan)

      # Report telemetry.
      esi.Telemetry.report_signal(
          ports.clk,
          ports.rst,
          esi.AppID("addrCmdCycles"),
          final_cycles,
      )
      esi.Telemetry.report_signal(
          ports.clk,
          ports.rst,
          esi.AppID("addrCmdIssued"),
          issue_cnt.out,
      )
      esi.Telemetry.report_signal(
          ports.clk,
          ports.rst,
          esi.AppID("addrCmdResponses"),
          responses_cnt.out,
      )

  return AddressCommand


@modparams
def ReadMem(width: int):

  class ReadMem(Module):
    """Host memory read test module.

        Function:
          Issues a sequence of host memory read requests using an internal
          address/control submodule which is configured via MMIO writes. Each
          read returns 'width' bits; the low 64 bits of the most recent
          response are latched and exported as telemetry (lastReadLSB).

        Flit width:
          'width' is the number of payload data bits per read flit. The address
          stride between successive requests is ceil(width/32) 32-bit words
          (= ceil(width/32) * 4 bytes). Non–power‑of‑two widths are supported
          and packed into the minimum whole 32‑bit word count.

        MMIO command interface:
          0x10  Write: Starting base address for the read operation.
          0x18  Write: Number of flits (read transactions) to perform.
          0x20  Write: Start the operation (assert once to launch).
          Reads return the current flits_left (remaining responses).

        Operation:
          After 0x20 is written, sequential addresses are generated:
            addr = start_addr + i * ceil(width/32)   (i = 0 .. flits-1)
          Each address produces one host memory read request.

        Telemetry (AppID -> signal):
          addrCmdCycles     Total cycles elapsed during the active window.
          addrCmdIssued     Count of host memory commands issued.
          addrCmdResponses  Count of host memory responses received.
          lastReadLSB       Low 64 bits of the most recently received read data.

        Notes:
          Backpressure on the read response channel naturally throttles issue.
          Completion occurs when responses == requested flits.
        """

    clk = Clock()
    rst = Reset()

    width_bits = Constant(UInt(32), width)

    @generator
    def construct(ports):
      clk = ports.clk
      rst = ports.rst

      address_cmd_resp = Wire(Channel(Bits(0)))
      addresses = AddressCommand(width)(
          clk=clk,
          rst=rst,
          hostmem_cmd_done=address_cmd_resp,
          instance_name="address_command",
      )

      read_cmd_chan = addresses.hostmem_cmd_address.transform(
          lambda addr: esi.HostMem.ReadReqType({
              "tag": UInt(8)(0),
              "address": addr
          }))
      read_responses = esi.HostMem.read(
          appid=AppID("host"),
          data_type=Bits(width),
          req=read_cmd_chan,
      )
      # Signal completion to AddressCommand (each response -> Bits(0)).
      address_cmd_resp.assign(read_responses.transform(lambda resp: Bits(0)(0)))
      # Snoop the response channel to capture the low 64 bits without consuming it.
      read_resp_valid, _, read_resp_data = read_responses.snoop()
      last_read_lsb = Reg(
          UInt(64),
          clk=ports.clk,
          rst=ports.rst,
          rst_value=0,
          ce=read_resp_valid,
          name="last_read_lsb",
      )
      last_read_lsb.assign(read_resp_data.data.as_uint(64))
      esi.Telemetry.report_signal(
          ports.clk,
          ports.rst,
          esi.AppID("lastReadLSB"),
          last_read_lsb,
      )

  return ReadMem


@modparams
def WriteMem(width: int) -> Type[Module]:

  class WriteMem(Module):
    """Host memory write test module.

        Function:
          Issues sequential host memory write requests produced by an internal
          address/control submodule configured via MMIO writes. Data for each
          flit is the current free‑running 32‑bit cycle counter value
          (zero‑extended or truncated to 'width').

        Flit width:
          'width' is the number of payload data bits per write flit. The address
          stride between successive writes is ceil(width/32) 32‑bit words
          (= ceil(width/32) * 4 bytes). Wider payloads span multiple words;
          narrower payloads still consume one word of address space per flit.

        MMIO command interface:
          0x10  Write: Starting base address for the write operation.
          0x18  Write: Number of flits (write transactions) to perform.
          0x20  Write: Start the operation (assert once to launch).
          Reads return the current flits_left (remaining responses).

        Data pattern:
          data[i] = cycle_counter sampled when the write command for flit i is formed.

        Telemetry (AppID -> signal):
          addrCmdCycles     Total cycles elapsed during the active window.
          addrCmdIssued     Count of host memory commands issued.
          addrCmdResponses  Count of host memory responses received.

        Notes:
          No additional telemetry beyond the above signals is generated here.
          Completion occurs when write responses == requested flits.
        """

    clk = Clock()
    rst = Reset()

    width_bits = Constant(UInt(32), width)

    @generator
    def construct(ports):
      clk = ports.clk
      rst = ports.rst

      cycle_counter_reset = Wire(Bits(1))
      cycle_counter = Counter(32)(
          clk=clk,
          rst=rst,
          clear=Bits(1)(0),
          increment=Bits(1)(1),
      )

      address_cmd_resp = Wire(Channel(Bits(0)))
      addresses = AddressCommand(width)(
          clk=clk,
          rst=rst,
          hostmem_cmd_done=address_cmd_resp,
          instance_name="address_command",
      )
      cycle_counter_reset.assign(addresses.command_go)

      write_cmd_chan = addresses.hostmem_cmd_address.transform(
          lambda addr: esi.HostMem.write_req_channel_type(UInt(width))({
              "tag": UInt(8)(0),
              "address": addr,
              "data": cycle_counter.out.as_uint(width),
          }))

      write_responses = esi.HostMem.write(
          appid=AppID("host"),
          req=write_cmd_chan,
      )
      # Signal completion to AddressCommand (each response -> Bits(0)).
      address_cmd_resp.assign(
          write_responses.transform(lambda resp: Bits(0)(0)))

  return WriteMem


@modparams
def ToHostDMATest(width: int):
  """Construct a module that sends the write count over a channel to the host
    the specified number of times. Exercises any DMA engine."""

  class ToHostDMATest(Module):
    """Transmit a 32-bit cycle counter value to the host a programmed number of times.

        Functionality:
          A free-running 32-bit counter advances only on successful channel
          handshakes. A write to MMIO offset 0x0 programs 'write_count' (number
          of messages to send). Each message’s payload is the counter value
          constrained to 'width':
            width < 32  -> lower 'width' bits (truncated)
            width >= 32 -> zero-extended to 'width' bits
          One message is emitted per handshake until 'write_count' messages have
          been transferred. A new write to 0x0 re-arms after completion.

        Width:
          Selects how the 32-bit counter is represented on the output channel
          (truncate or zero-extend as above).

        MMIO command interface:
          0x0 Write: Set write_count (messages to transmit). Starts a new
                    sequence if idle/completed.
          0x0 Read: Returns constant 0.

        Telemetry:
          totalWrites (AppID "totalWrites"): Count of messages successfully sent.
          toHostCycles (AppID "toHostCycles"): Cycle count from write_count programming
            (start) through completion (inclusive of active cycles).

        Notes:
          Backpressure governs pacing. Completion when totalWrites == write_count.
        """

    clk = Clock()
    rst = Reset()

    width_bits = Constant(UInt(32), width)

    @generator
    def construct(ports):
      count_reached = Wire(Bits(1))
      count_valid = Wire(Bits(1))
      out_xact = Wire(Bits(1))
      cycle_counter = Counter(32)(
          clk=ports.clk,
          rst=ports.rst,
          clear=Bits(1)(0),
          increment=out_xact,
      )

      write_cntr_incr = ~count_reached & count_valid & out_xact
      write_counter = Counter(32)(
          clk=ports.clk,
          rst=ports.rst,
          clear=count_reached,
          increment=write_cntr_incr,
      )
      num_writes = write_counter.out

      # Get the MMIO space for commands.
      cmd_chan_wire = Wire(Channel(esi.MMIOReadWriteCmdType))
      resp_ready_wire = Wire(Bits(1))
      cmd, cmd_valid = cmd_chan_wire.unwrap(resp_ready_wire)
      mmio_xact = cmd_valid & resp_ready_wire
      response_data = Bits(64)(0)
      response_chan, response_ready = Channel(response_data.type).wrap(
          response_data, cmd_valid)
      resp_ready_wire.assign(response_ready)

      # write_count is the specified number of times to send the cycle count.
      write_count_ce = mmio_xact & cmd.write & (cmd.offset == UInt(32)(0))
      write_count = cmd.data.as_uint().reg(clk=ports.clk,
                                           rst=ports.rst,
                                           rst_value=0,
                                           ce=write_count_ce)
      count_reached.assign(num_writes == write_count)
      count_valid.assign(
          ControlReg(
              clk=ports.clk,
              rst=ports.rst,
              asserts=[write_count_ce],
              resets=[count_reached],
          ))

      mmio_rw = esi.MMIO.read_write(appid=AppID("cmd"))
      mmio_rw_cmd_chan = mmio_rw.unpack(data=response_chan)["cmd"]
      cmd_chan_wire.assign(mmio_rw_cmd_chan)

      # Output channel.
      out_channel, out_channel_ready = Channel(UInt(width)).wrap(
          cycle_counter.out.as_uint(width), count_valid)
      out_xact.assign(out_channel_ready & count_valid)
      esi.ChannelService.to_host(name=AppID("out"), chan=out_channel)

      total_write_counter = Counter(64)(
          clk=ports.clk,
          rst=ports.rst,
          clear=Bits(1)(0),
          increment=write_cntr_incr,
      )
      esi.Telemetry.report_signal(
          ports.clk,
          ports.rst,
          esi.AppID("totalWrites"),
          total_write_counter.out,
      )

      # Cycle telemetry: count cycles while sequence active.
      tohost_cycle_cnt = Counter(64)(
          clk=ports.clk,
          rst=ports.rst,
          clear=write_count_ce,
          increment=count_valid,
          instance_name="tohost_cycle_counter",
      )
      tohost_final_cycles = Reg(
          UInt(64),
          clk=ports.clk,
          rst=ports.rst,
          rst_value=0,
          ce=count_reached,
          name="tohost_cycles",
      )
      tohost_final_cycles.assign(tohost_cycle_cnt.out.as_uint())
      esi.Telemetry.report_signal(
          ports.clk,
          ports.rst,
          esi.AppID("toHostCycles"),
          tohost_final_cycles,
      )

  return ToHostDMATest


@modparams
def FromHostDMATest(width: int):
  """Construct a module that receives the write count over a channel from the
    host the specified number of times. Exercises any DMA engine."""

  class FromHostDMATest(Module):
    """Receive test data from the host a programmed number of times.

        Functionality:
          A write to MMIO offset 0x0 programs 'read_count', the number of messages
          to accept from the host. The input channel (AppID "in") is marked ready
          while the number of received messages is less than 'read_count'. Each
          received width-bit payload is latched; the most recent value is exposed
          on MMIO reads.

        Width:
          'width' is the payload bit width of each received message. The latched
          value is widened/truncated to 64 bits for MMIO read-back (lower 64 bits
          if width > 64).

        MMIO command interface:
          0x0 Write: Set read_count (number of messages to receive). Clears the
              internal receive counter.
          0x0 Read: Returns the last received value (Bits(64), derived from the
              width-bit payload).

        Telemetry:
          fromHostCycles (AppID "fromHostCycles"): Cycle count from read_count programming
            (start) through completion of the programmed receive sequence.

        Notes:
          Completion is when received messages == programmed read_count; another
          write to 0x0 re-arms for a new sequence.
        """

    clk = Clock()
    rst = Reset()

    width_bits = Constant(UInt(32), width)

    @generator
    def build(ports):
      last_read = Wire(UInt(width))

      # Get the MMIO space for commands.
      cmd_chan_wire = Wire(Channel(esi.MMIOReadWriteCmdType))
      resp_ready_wire = Wire(Bits(1))
      cmd, cmd_valid = cmd_chan_wire.unwrap(resp_ready_wire)
      mmio_xact = cmd_valid & resp_ready_wire
      response_data = last_read.as_bits(64)
      response_chan, response_ready = Channel(response_data.type).wrap(
          response_data, cmd_valid)
      resp_ready_wire.assign(response_ready)

      # read_count is the specified number of times to recieve data.
      read_count_ce = mmio_xact & cmd.write & (cmd.offset == UInt(32)(0))
      read_count = cmd.data.as_uint().reg(clk=ports.clk,
                                          rst=ports.rst,
                                          rst_value=0,
                                          ce=read_count_ce)
      in_data_xact = NamedWire(Bits(1), "in_data_xact")
      read_counter = Counter(32)(
          clk=ports.clk,
          rst=ports.rst,
          clear=read_count_ce,
          increment=in_data_xact,
      )

      mmio_rw = esi.MMIO.read_write(appid=AppID("cmd"))
      mmio_rw_cmd_chan = mmio_rw.unpack(data=response_chan)["cmd"]
      cmd_chan_wire.assign(mmio_rw_cmd_chan)

      in_chan = esi.ChannelService.from_host(name=AppID("in"), type=UInt(width))
      in_ready = NamedWire(read_counter.out < read_count, "in_ready")
      in_data, in_valid = in_chan.unwrap(in_ready)
      NamedWire(in_data, "in_data")
      in_data_xact.assign(in_valid & in_ready)

      last_read.assign(
          in_data.reg(
              clk=ports.clk,
              rst=ports.rst,
              ce=in_data_xact,
              name="last_read",
          ))

      # Cycle telemetry: detect completion and count active cycles.
      fromhost_count_reached = Wire(Bits(1))
      fromhost_count_reached.assign(read_counter.out == read_count)
      fromhost_cycle_valid = ControlReg(
          clk=ports.clk,
          rst=ports.rst,
          asserts=[read_count_ce],
          resets=[fromhost_count_reached],
          name="fromhost_cycle_active",
      )
      fromhost_cycle_cnt = Counter(64)(
          clk=ports.clk,
          rst=ports.rst,
          clear=read_count_ce,
          increment=fromhost_cycle_valid,
          instance_name="fromhost_cycle_counter",
      )
      fromhost_final_cycles = Reg(
          UInt(64),
          clk=ports.clk,
          rst=ports.rst,
          rst_value=0,
          ce=fromhost_count_reached,
          name="fromhost_cycles",
      )
      fromhost_final_cycles.assign(fromhost_cycle_cnt.out.as_uint())
      esi.Telemetry.report_signal(
          ports.clk,
          ports.rst,
          esi.AppID("fromHostCycles"),
          fromhost_final_cycles,
      )

  return FromHostDMATest


class EsiTester(Module):
  """Top-level ESI test harness module.

    Contains submodules:
      CallbackTest            (single instance) – host callback via MMIO write (offset 0x10).
      LoopbackInOutAdd        (single instance) – function service adding constant 11.
      MMIOAdd(add_amt)        instances for add_amt in {4, 9, 14} – MMIO read returns offset + add_amt.
      ReadMem(width)          for widths: 32, 64, 128, 256, 512, 534 – host memory read tests.
      WriteMem(width)         for widths: 32, 64, 128, 256, 512, 534 – host memory write tests.
      ToHostDMATest(width)    for widths: 32, 64, 128, 256, 512, 534 – DMA to host, cycle & count telemetry.
      FromHostDMATest(width)  for widths: 32, 64, 128, 256, 512, 534 – DMA from host, cycle telemetry.

    Width set used across Read/Write/DMA tests:
      widths = [32, 64, 128, 256, 512, 534]

    Purpose:
      Aggregates all functional, MMIO, host memory, and DMA tests into one image
      for comprehensive accelerator validation and telemetry collection.
    """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    CallbackTest(
        clk=ports.clk,
        rst=ports.rst,
        instance_name="cb_test",
        appid=AppID("cb_test"),
    )
    LoopbackInOutAdd(
        clk=ports.clk,
        rst=ports.rst,
        instance_name="loopback",
        appid=AppID("loopback"),
    )
    StreamingAdder(
        clk=ports.clk,
        rst=ports.rst,
        instance_name="streaming_adder",
        appid=AppID("streaming_adder"),
    )

    for i in range(4, 18, 5):
      MMIOAdd(i)(instance_name=f"mmio_add_{i}", appid=AppID("mmio_add", i))

    for width in [32, 64, 128, 256, 512, 534]:
      ReadMem(width)(
          instance_name=f"readmem_{width}",
          appid=esi.AppID("readmem", width),
          clk=ports.clk,
          rst=ports.rst,
      )
      WriteMem(width)(
          instance_name=f"writemem_{width}",
          appid=AppID("writemem", width),
          clk=ports.clk,
          rst=ports.rst,
      )
      ToHostDMATest(width)(
          instance_name=f"tohostdma_{width}",
          appid=AppID("tohostdma", width),
          clk=ports.clk,
          rst=ports.rst,
      )
      FromHostDMATest(width)(
          instance_name=f"fromhostdma_{width}",
          appid=AppID("fromhostdma", width),
          clk=ports.clk,
          rst=ports.rst,
      )

    for i in range(3):
      ReadMem(512)(
          instance_name=f"readmem_{i}",
          appid=esi.AppID(f"readmem_{i}", 512),
          clk=ports.clk,
          rst=ports.rst,
      )
      WriteMem(512)(
          instance_name=f"writemem_{i}",
          appid=AppID(f"writemem_{i}", 512),
          clk=ports.clk,
          rst=ports.rst,
      )
      ToHostDMATest(512)(
          instance_name=f"tohostdma_{i}",
          appid=AppID(f"tohostdma_{i}", 512),
          clk=ports.clk,
          rst=ports.rst,
      )
      FromHostDMATest(512)(
          instance_name=f"fromhostdma_{i}",
          appid=AppID(f"fromhostdma_{i}", 512),
          clk=ports.clk,
          rst=ports.rst,
      )


if __name__ == "__main__":
  bsp = get_bsp(sys.argv[2])
  s = System(bsp(EsiTester), name="EsiTester", output_directory=sys.argv[1])
  s.compile()
  s.package()
