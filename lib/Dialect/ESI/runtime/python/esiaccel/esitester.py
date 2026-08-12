# ===- esitester.py - accelerator for testing ESI functionality -----------===//
#
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//
#
#  This design is used for testing ESI functionality. It is distribed in the
#  esiaccel package for BSP developers to exercise new BSPs, boards, and
#  features. It is compatible with the distributed esitester application.
#
#  Importantly, it is not a standalone application -- merely a collection of
#  test modules and top level. The user must write a main function which builds
#  the system using this module as a library.
#
# ===----------------------------------------------------------------------===//

import sys
from typing import Type

import pycde.esi as esi
from pycde import Clock, Module, Reset, System, generator, modparams
from esiaccel.bsp import get_bsp
from esiaccel.components.mmio import MmioRegistry, mmio_write_we
from pycde.common import AppID, Constant, Input, InputChannel, Output, OutputChannel
from pycde.constructs import ControlReg, Counter, Mux, NamedWire, Reg, Wire
from pycde.module import Metadata
from pycde.signals import BitsSignal
from pycde.testing import print_info
from pycde.types import Array, Bits, Channel, ChannelSignaling, UInt

# Fixed 64-bit seed for the hostmem burst data pattern. WriteMem fills every
# byte of each element with (seed ^ index) tiled across bytes and XORed with a
# distinct per-position mask, and ReadMem folds every received byte into a
# checksum, so the host tests verify the full-width data landed at / was fetched
# from the right bytes -- independent of the backend's engine word width.
_ESITESTER_SEQ_SEED = 0x5A5A5A5A5A5A5A5A

# ---------------------------------------------------------------------------
# Reusable building blocks for timing-friendly MMIO-controlled test modules.
#
# Both helpers keep the BSP's MMIO mux and the consumer-facing channel inside
# this user module isolated from each other's combinational paths:
#
#   * `MmioRegistry` drives the MMIO bundle's `cmd_ready` and response `valid`
#     from a single `resp_pending` ControlReg. The BSP MMIO mux therefore
#     only sees FF outputs, and the user module's internal write-enable
#     strobes are 1-cycle-late registered pulses gated by `write_cmd_xact_r`.
#
#   * `IterationGate` exposes a counter+limit "run for N iterations" widget
#     whose `active` output is a ControlReg (FF) and whose only consumer-
#     driven input (`iter_xact`) terminates at the counter's clock enable.
#     The consumer's channel-ready signal never re-emerges combinationally
#     from this module. It also always reports `cycles` telemetry for the
#     active window.
#
# Use both together to build a test that drives a single channel with N
# iterations under MMIO control without exposing the BSP arbitration mux to
# wide internal combinational logic.
# ---------------------------------------------------------------------------


@modparams
def IterationGate(count_width: int):
  """Run an internal counter for `limit` iterations gated by `iter_xact`,
    and always report cycle telemetry for the active window.

    `start_pulse` asserts `active` (and clears the counters) for one cycle.
    `iter_xact` is the per-iteration handshake strobe; it feeds only a
    Counter clock enable. `active` clears the cycle after `iter_count`
    reaches `limit`.

    The consumer's channel-ready signal can flow into `iter_xact` without
    re-emerging combinationally through any output: only Counters (FFs)
    consume it.

    Telemetry reported under this instance's AppID:
        cycles : ui64
            Cycles `active` was asserted between `start_pulse` and
            `count_reached`; latched on `count_reached` so the host always
            reads the final value of the most recent run.

    Ports:
        start_pulse   : Input  Bits(1)
        limit         : Input  UInt(count_width)
        iter_xact     : Input  Bits(1)
        active        : Output Bits(1)            -- ControlReg, FF output
        count_reached : Output Bits(1)            -- iter_count == limit
        iter_count    : Output UInt(count_width)
        iters_left    : Output UInt(count_width)
    """

  class IterationGate(Module):
    clk = Clock()
    rst = Reset()

    start_pulse = Input(Bits(1))
    limit = Input(UInt(count_width))
    iter_xact = Input(Bits(1))

    active = Output(Bits(1))
    count_reached = Output(Bits(1))
    iter_count = Output(UInt(count_width))
    iters_left = Output(UInt(count_width))

    @generator
    def construct(ports):
      clk = ports.clk
      rst = ports.rst

      # `count_reached_sig` fires combinationally on the last
      # `iter_xact` (when `iter_count` is about to become
      # `limit`), so `active` drops the cycle the consumer would
      # have issued the (limit+1)th transaction. Without this,
      # there is a 2-cycle race between `iter_count` reaching
      # `limit` and the ControlReg dropping `active`, during
      # which the consumer issues one extra transaction -- and
      # for `limit=1` the host-visible counters jump straight
      # from 0 to 2, so any host poll for "== 1" never catches.
      # `count_reached` is combinational on `iter_xact` (which is
      # consumer-driven), but it only feeds the ControlReg reset
      # (a flop) and telemetry, so the consumer's ready signal
      # does not re-emerge combinationally on `active`.
      count_reached_wire = Wire(Bits(1))
      active_r = ControlReg(
          clk=clk,
          rst=rst,
          asserts=[ports.start_pulse],
          resets=[count_reached_wire],
          name="active_r",
      )
      counter = Counter(count_width)(
          clk=clk,
          rst=rst,
          clear=ports.start_pulse,
          increment=ports.iter_xact,
          instance_name="iter_counter",
      )
      last_iter = (counter.out.as_uint() == (
          ports.limit - UInt(count_width)(1)).as_uint(count_width)).as_bits(1)
      count_reached_sig = ports.iter_xact & last_iter
      count_reached_wire.assign(count_reached_sig)

      ports.active = active_r
      ports.count_reached = count_reached_sig
      ports.iter_count = counter.out.as_uint()
      # Elements remaining in the *active* window: 0 when no run is in flight
      # (after `limit` is set but before `start_pulse`, or after completion),
      # else `limit - iter_count`.
      remaining = (ports.limit - counter.out.as_uint()).as_uint(count_width)
      ports.iters_left = Mux(active_r, UInt(count_width)(0), remaining)

      cycles_cnt = Counter(64)(
          clk=clk,
          rst=rst,
          clear=ports.start_pulse,
          increment=active_r,
          instance_name="cycle_counter",
      )
      final_cycles = Reg(
          UInt(64),
          clk=clk,
          rst=rst,
          rst_value=0,
          ce=count_reached_sig,
          name="final_cycles",
      )
      final_cycles.assign(cycles_cnt.out.as_uint())
      esi.Telemetry.report_signal(clk, rst, AppID("cycles"), final_cycles)

  return IterationGate


class CallbackTest(Module):
  """Call a function on the host when an MMIO write is received at offset
    0x10."""

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


@modparams
def StreamingAdder(numItems: int):
  """Creates a StreamingAdder module parameterized by the number of items per
  window frame. The module exposes a function which has an argument of struct
  {add_amt, list<uint32>}. It then adds add_amt to each element of the list in
  parallel (numItems at a time) and returns the resulting list.
  """

  class StreamingAdder(Module):
    clk = Clock()
    rst = Reset()

    @generator
    def construct(ports):
      from pycde.types import StructType, List, Window

      # Define the argument type: struct { add_amt: UInt(32), list: List<UInt(32)> }
      arg_struct_type = StructType([("add_amt", UInt(32)),
                                    ("input", List(UInt(32)))])

      # Create a windowed version with numItems parallel elements
      arg_window_type = Window(
          "arg_window", arg_struct_type,
          [Window.Frame(None, ["add_amt", ("input", numItems)])])

      # Result is also a List with numItems parallel elements
      result_struct_type = StructType([("data", List(UInt(32)))])
      result_window_type = Window("result_window", result_struct_type,
                                  [Window.Frame(None, [("data", numItems)])])

      result_chan = Wire(Channel(result_window_type))
      args = esi.FuncService.get_call_chans(AppID("streaming_add"),
                                            arg_type=arg_window_type,
                                            result=result_chan)

      # Unwrap the argument channel
      ready = Wire(Bits(1))
      arg_data, arg_valid = args.unwrap(ready)

      # Unwrap the window to get the lowered struct
      # Lowered type: struct { add_amt, input: array[numItems], input_size, last }
      arg_unwrapped = arg_data.unwrap()

      # Extract add_amt and input array from the struct
      add_amt = arg_unwrapped["add_amt"]
      input_arr = arg_unwrapped["input"]

      # Perform all additions in parallel
      result_arr = [
          (add_amt + input_arr[i]).as_uint(32) for i in range(numItems)
      ]

      # Build the result lowered type
      # Lowered type: struct { data: array[numItems], data_size, last }
      lowered_val = result_window_type.lowered_type({
          "data": result_arr,
          "data_size": arg_unwrapped["input_size"],
          "last": arg_unwrapped["last"]
      })

      result_window = result_window_type.wrap(lowered_val)

      # Wrap the result into a channel
      result_chan_internal, result_ready = Channel(result_window_type).wrap(
          result_window, arg_valid)
      ready.assign(result_ready)
      result_chan.assign(result_chan_internal)

  return StreamingAdder


class CoordTranslator(Module):
  """Exposes a function which takes a struct of {x_translation, y_translation,
  coords: list<struct{x, y}>} and adds the translation to each coordinate,
  returning the translated list of coordinates.
  """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    from pycde.types import StructType, List, Window

    # Define the coordinate type: struct { x: UInt(32), y: UInt(32) }
    coord_type = StructType([("x", UInt(32)), ("y", UInt(32))])

    # Define the argument type: struct { x_translation, y_translation, coords }
    arg_struct_type = StructType([("x_translation", UInt(32)),
                                  ("y_translation", UInt(32)),
                                  ("coords", List(coord_type))])

    # Create a windowed version of the argument struct for streaming
    arg_window_type = Window.default_of(arg_struct_type)

    # Result is also a List of coordinates
    result_type = List(coord_type)
    result_window_type = Window.default_of(result_type)

    result_chan = Wire(Channel(result_window_type))
    args = esi.FuncService.get_call_chans(AppID("translate_coords"),
                                          arg_type=arg_window_type,
                                          result=result_chan)

    # Unwrap the argument channel
    ready = Wire(Bits(1))
    arg_data, arg_valid = args.unwrap(ready)

    # Unwrap the window to get the struct/union
    arg_unwrapped = arg_data.unwrap()

    # Extract translations and coordinates from the struct
    x_translation = arg_unwrapped["x_translation"]
    y_translation = arg_unwrapped["y_translation"]
    input_coord = arg_unwrapped["coords"]

    # Add translations to each coordinate
    result_x = (x_translation + input_coord["x"]).as_uint(32)
    result_y = (y_translation + input_coord["y"]).as_uint(32)

    # Create the result coordinate struct
    result_coord = coord_type({"x": result_x, "y": result_y})

    result_window = result_window_type.wrap(
        result_window_type.lowered_type({
            "data": result_coord,
            "last": arg_unwrapped.last
        }))

    # Wrap the result into a channel
    result_chan_internal, result_ready = Channel(result_window_type).wrap(
        result_window, arg_valid)
    ready.assign(result_ready)
    result_chan.assign(result_chan_internal)


class SerialCoordTranslator(Module):
  """Like CoordTranslator, but uses the serial (bulk-transfer) list encoding.

  Input wire format is a window with two frames:
    - "header": {x_translation, y_translation, coords_count}
    - "data":   {coords[1]}  (one coordinate per frame)

  Output wire format is also a window with two frames:
    - "header": {coords_count}
    - "data":   {coords[1]}  (one coordinate per frame)

  In bulk-transfer encoding, the sender may transmit multiple header/data
  sequences to extend a list. A common pattern is to set coords_count=64 and
  re-send a new header every 64 items; the final header has coords_count=0.
  This module passes the header count through and translates each coordinate.
  """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    from pycde.types import List, StructType, Window

    clk = ports.clk
    rst = ports.rst

    bulk_count_width = 16
    items_per_frame = 1

    coord_type = StructType([("x", Bits(32)), ("y", Bits(32))])

    # ----- Input window type (serial/bulk transfer) -----
    arg_struct_type = StructType([
        ("x_translation", Bits(32)),
        ("y_translation", Bits(32)),
        ("coords", List(coord_type)),
    ])
    arg_window_type = Window(
        "serial_coord_args",
        arg_struct_type,
        [
            Window.Frame(
                "header",
                [
                    "x_translation",
                    "y_translation",
                    ("coords", 0, bulk_count_width),
                ],
            ),
            Window.Frame(
                "data",
                [("coords", items_per_frame, 0)],
            ),
        ],
    )

    # ----- Output window type (serial/bulk transfer) -----
    result_struct_type = StructType([("coords", List(coord_type))])
    result_window_type = Window(
        "serial_coord_result",
        result_struct_type,
        [
            Window.Frame(
                "header",
                [("coords", 0, bulk_count_width)],
            ),
            Window.Frame(
                "data",
                [("coords", items_per_frame, 0)],
            ),
        ],
    )

    result_chan = Wire(Channel(result_window_type))
    args = esi.FuncService.get_call_chans(
        AppID("translate_coords_serial"),
        arg_type=arg_window_type,
        result=result_chan,
    )

    # Unwrap the argument channel.
    in_ready = Wire(Bits(1))
    in_window, in_valid = args.unwrap(in_ready)
    in_union = in_window.unwrap()

    hdr_frame = in_union["header"]
    data_frame = in_union["data"]

    hdr_x = hdr_frame["x_translation"].as_uint(32)
    hdr_y = hdr_frame["y_translation"].as_uint(32)
    hdr_count_bits = hdr_frame["coords_count"]
    hdr_count = hdr_count_bits.as_uint(bulk_count_width)

    out_hdr_struct_ty = result_window_type.lowered_type.header
    out_data_struct_ty = result_window_type.lowered_type.data

    # Output channel (built below) drives readiness/backpressure.
    out_ready_wire = Wire(Bits(1))
    handshake = in_valid & out_ready_wire

    # Track which frame we're currently expecting.
    in_is_header = Reg(
        Bits(1),
        clk=clk,
        rst=rst,
        rst_value=1,
        ce=handshake,
        name="in_is_header",
    )
    # Only log the frame count when the handshake is for a header frame.
    hdr_handshake = handshake & in_is_header
    hdr_handshake.when_true(
        lambda: print_info("Received frame count=%d", hdr_count_bits))

    # Latch the most recent header count for re-use when emitting the output
    # header (do not rely on union extracts during data frames).
    hdr_is_zero = hdr_count == UInt(bulk_count_width)(0)
    footer_handshake = hdr_handshake & hdr_is_zero
    start_handshake = hdr_handshake & ~hdr_is_zero
    message_active = ControlReg(
        clk,
        rst,
        asserts=[start_handshake],
        resets=[footer_handshake],
        name="message_active",
    )
    count_reg = Reg(
        UInt(bulk_count_width),
        clk=clk,
        rst=rst,
        rst_value=0,
        ce=hdr_handshake,
        name="coords_count",
    )
    count_reg.assign(hdr_count)

    data_handshake = handshake & ~in_is_header
    data_count = Counter(bulk_count_width)(
        clk=clk,
        rst=rst,
        clear=hdr_handshake,
        increment=data_handshake,
        instance_name="data_count",
    ).out

    # Latch translations only on the first header of a message.
    x_translation_reg = Reg(
        UInt(32),
        clk=clk,
        rst=rst,
        rst_value=0,
        ce=start_handshake & ~message_active,
        name="x_translation",
    )
    y_translation_reg = Reg(
        UInt(32),
        clk=clk,
        rst=rst,
        rst_value=0,
        ce=start_handshake & ~message_active,
        name="y_translation",
    )
    x_translation_reg.assign(hdr_x)
    y_translation_reg.assign(hdr_y)

    # Next-state logic for header/data tracking.
    count_minus_one = (count_reg -
                       UInt(bulk_count_width)(1)).as_uint(bulk_count_width)
    data_last = data_count == count_minus_one
    next_is_header = Mux(in_is_header, data_last, hdr_is_zero)
    in_is_header.assign(next_is_header)

    # Build output frames.
    out_hdr_struct = out_hdr_struct_ty(
        {"coords_count": hdr_count.as_bits(bulk_count_width)})

    in_coord = data_frame["coords"][0]
    in_x = in_coord["x"].as_uint(32)
    in_y = in_coord["y"].as_uint(32)
    translated_x = (x_translation_reg + in_x).as_uint(32)
    translated_y = (y_translation_reg + in_y).as_uint(32)
    out_coord = coord_type({
        "x": translated_x.as_bits(32),
        "y": translated_y.as_bits(32),
    })
    out_data_struct = out_data_struct_ty({"coords": [out_coord]})

    out_union_hdr = result_window_type.lowered_type(("header", out_hdr_struct))
    out_union_data = result_window_type.lowered_type(("data", out_data_struct))
    out_union = Mux(in_is_header, out_union_data, out_union_hdr)
    out_window = result_window_type.wrap(out_union)

    out_chan, out_ready = Channel(result_window_type).wrap(out_window, in_valid)
    out_ready_wire.assign(out_ready)

    in_ready.assign(out_ready)
    result_chan.assign(out_chan)


class AutoSerialCoordTranslator(Module):
  """Like CoordTranslator, but exposes the function with the serial
  (bulk-transfer) list encoding on both the argument and result. Internally,
  the serial input is converted to the parallel one-item-per-message form via
  `ListWindowToParallel`, the per-coordinate translation is applied, and the
  parallel result is converted back to the serial wire form via
  `ListWindowToSerial`.

  This exercises the automatic serial<->parallel conversion modules instead of
  building the frame state machine by hand (as `SerialCoordTranslator` does).
  """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    from pycde.types import StructType, List, Window
    from pycde.esi import ListWindowToParallel, ListWindowToSerial

    bulk_count_width = 16
    items_per_frame = 1
    # Intentionally tiny FIFO so the host's coord lists (which can be much
    # larger than this) get split across many bulk transfers, exercising the
    # multi-burst code paths in `ListWindowToSerial` (drain-on-full bursts
    # interleaved with the producer, plus the count==0 terminator).
    fifo_depth = 4

    # ---- Externally-visible (serial) function arg/result types. ----
    # NOTE: use Bits for coord/translation fields. The window lowering for
    # bulk-transfer encoding currently strips signedness from union variant
    # fields, which causes type mismatches when the underlying struct uses
    # UInt; SerialCoordTranslator hits the same constraint.
    coord_type = StructType([("x", Bits(32)), ("y", Bits(32))])

    arg_struct_type = StructType([("x_translation", Bits(32)),
                                  ("y_translation", Bits(32)),
                                  ("coords", List(coord_type))])
    arg_window_type = Window.serial_of(arg_struct_type, bulk_count_width,
                                       items_per_frame)

    result_type = List(coord_type)
    result_window_type = Window.serial_of(result_type, bulk_count_width,
                                          items_per_frame)

    # Result channel back to FuncService is the serial output of the
    # parallel->serial converter (assigned at the end).
    result_chan = Wire(Channel(result_window_type))
    args = esi.FuncService.get_call_chans(AppID("translate_coords_auto_serial"),
                                          arg_type=arg_window_type,
                                          result=result_chan)

    # ---- Convert the serial argument stream into a parallel one. ----
    s2p = ListWindowToParallel(arg_window_type)(clk=ports.clk,
                                                rst=ports.rst,
                                                serial_in=args)
    parallel_arg = s2p.parallel_out

    # ---- Apply the per-coordinate translation. ----
    par_ready = Wire(Bits(1))
    par_window, par_valid = parallel_arg.unwrap(par_ready)
    par_struct = par_window.unwrap()

    x_translation = par_struct["x_translation"].as_uint(32)
    y_translation = par_struct["y_translation"].as_uint(32)
    input_coord = par_struct["coords"]
    last_bit = par_struct["last"]

    result_x = (x_translation +
                input_coord["x"].as_uint(32)).as_uint(32).as_bits(32)
    result_y = (y_translation +
                input_coord["y"].as_uint(32)).as_uint(32).as_bits(32)
    result_coord = coord_type({"x": result_x, "y": result_y})

    parallel_result_window_type = Window.default_of(result_type)
    parallel_result_struct = parallel_result_window_type.lowered_type({
        "data": result_coord,
        "last": last_bit,
    })
    parallel_result_window = parallel_result_window_type.wrap(
        parallel_result_struct)

    parallel_result_chan, parallel_result_ready = Channel(
        parallel_result_window_type).wrap(parallel_result_window, par_valid)
    par_ready.assign(parallel_result_ready)

    # ---- Convert the parallel result stream back into a serial one. ----
    p2s = ListWindowToSerial(parallel_result_window_type, bulk_count_width,
                             items_per_frame,
                             fifo_depth)(clk=ports.clk,
                                         rst=ports.rst,
                                         parallel_in=parallel_result_chan)
    result_chan.assign(p2s.serial_out)


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
def BurstCommand(width: int):
  """MMIO-controlled single-burst command surface. Replaces AddressCommand's
    per-flit address stream: exposes one {address, tag, length} burst request
    per command -- for a ``read_list`` or a windowed (list) write -- and tracks
    completion.

    MMIO register map (8-byte stride):
      0x00 Read : flits_left (elements remaining in the active command).
      0x08 Write: base address.
      0x10 Write: list length (flits).
      0x18 Write: start.
    """

  class BurstCommand(Module):
    clk = Clock()
    rst = Reset()

    # Remaining elements (for MMIO read-back).
    flits_left = Output(UInt(64))
    # Single {address, tag, length} burst request, held valid until
    # accepted.
    burst_req = OutputChannel(esi.HostMem.read_req_burst_type())
    # One Bits(0) completion token per received element / write ack.
    hostmem_cmd_done = InputChannel(Bits(0))

    @generator
    def construct(ports):
      clk = ports.clk
      rst = ports.rst

      # Register map (RO < RW < WO, 8-byte stride):
      #   0x00  flits_left   (RO, client-updated read-back)
      #   0x08  start_addr   (RW, host-written base address)
      #   0x10  flits_total  (RW, host-written list length)
      #   0x18  start        (WO, host write triggers the operation)
      flits_left_data = Wire(Bits(64))
      mmio = MmioRegistry(num_ro=1, num_rw=2, num_wo=1)(
          clk=clk,
          rst=rst,
          read_reg_ce=Array(Bits(1), 3)([Bits(1)(1),
                                         Bits(1)(0),
                                         Bits(1)(0)]),
          read_reg_data=Array(Bits(64),
                              3)([flits_left_data,
                                  Bits(64)(0),
                                  Bits(64)(0)]),
          instance_name="mmio",
          appid=AppID("mmio", width),
      )

      start_addr = mmio.read_reg_value[1].as_uint()
      flits_total = mmio.read_reg_value[2].as_uint()
      start_op_we = mmio_write_we(mmio, 0x18)

      # Response side: count completed elements; auto-reports cycles.
      _, done_valid = ports.hostmem_cmd_done.unwrap(Bits(1)(1))
      resp_gate = IterationGate(64)(
          clk=clk,
          rst=rst,
          start_pulse=start_op_we,
          limit=flits_total,
          iter_xact=done_valid,
          instance_name="resp_gate",
          appid=AppID("addrCmdResp"),
      )
      ports.flits_left = resp_gate.iters_left
      # The RO flits_left register mirrors the live remaining count.
      flits_left_data.assign(resp_gate.iters_left.as_bits(64))

      # Single burst request, held valid from start until accepted.
      burst_accepted = Wire(Bits(1))
      burst_pending = ControlReg(
          clk=clk,
          rst=rst,
          asserts=[start_op_we],
          resets=[burst_accepted],
          name="burst_pending",
      )
      burst_req_t = esi.HostMem.read_req_burst_type()
      burst_chan, burst_ready = Channel(burst_req_t).wrap(
          burst_req_t({
              "address": start_addr,
              "tag": UInt(8)(0),
              "length": flits_total,
          }),
          burst_pending,
      )
      burst_accepted.assign((burst_pending & burst_ready).as_bits())
      ports.burst_req = burst_chan

      # Issue side: one issued command per accepted burst.
      issue_cnt = Counter(64)(
          clk=clk,
          rst=rst,
          clear=start_op_we,
          increment=burst_accepted,
      )

      esi.Telemetry.report_signal(clk, rst, esi.AppID("addrCmdIssued"),
                                  issue_cnt.out)
      esi.Telemetry.report_signal(clk, rst, esi.AppID("addrCmdResponses"),
                                  resp_gate.iter_count)

  return BurstCommand


@modparams
def ReadMem(width: int):

  class ReadMem(Module):
    """Host memory burst (list) read test module.

        Issues a single ``read_list`` burst of 'flits' elements starting at the
        base address (both configured via MMIO) and receives the elements back
        as a windowed list (num_items=1 -> one element per frame). The low 64
        bits of the most recent element are exported as telemetry (lastReadLSB),
        and every byte of every element is folded into a byte-position-sensitive
        integrity checksum (readChecksum).

        MMIO command interface (via BurstCommand):
          0x00  Read : remaining element count (flits_left).
          0x08  Write: base address for the read.
          0x10  Write: number of list elements (flits) to read.
          0x18  Write: start the operation.

        Telemetry (AppID -> signal):
          addrCmdIssued     Count of burst commands issued (1 per command).
          addrCmdResponses  Count of list elements received.
          lastReadLSB       Low 64 bits of the most recent element.
        """

    clk = Clock()
    rst = Reset()

    width_bits = Constant(UInt(32), width)

    @generator
    def construct(ports):
      clk = ports.clk
      rst = ports.rst

      done_wire = Wire(Channel(Bits(0)))
      cmd = BurstCommand(width)(
          clk=clk,
          rst=rst,
          hostmem_cmd_done=done_wire,
          instance_name="burst_command",
      )

      # One read_list request per command; elements come back as a
      # windowed list.
      read_responses = esi.HostMem.read_list(
          appid=AppID("host"),
          req=cmd.burst_req,
          element_type=Bits(width),
          num_items=1,
      )
      # Each received element -> one completion token to BurstCommand.
      done_wire.assign(read_responses.transform(lambda resp: Bits(0)(0)))
      # Snoop each received element without consuming it.
      read_resp_valid_snoop, read_resp_data = read_responses.snoop_xact()
      read_elem = read_resp_data.unwrap()["data"][0]
      read_elem_lsb = read_elem.as_uint(64)
      last_read_lsb = Reg(
          UInt(64),
          clk=ports.clk,
          rst=ports.rst,
          rst_value=0,
          ce=read_resp_valid_snoop,
          name="last_read_lsb",
      )
      last_read_lsb.assign(read_elem_lsb)
      esi.Telemetry.report_signal(
          ports.clk,
          ports.rst,
          esi.AppID("lastReadLSB"),
          last_read_lsb,
      )
      # Byte-position-sensitive integrity checksum over all `width` bits: fold
      # each 64-bit chunk of the element into 64 bits with a per-chunk rotate
      # (so word/byte misplacement doesn't cancel), then XOR across elements.
      num_chunks = (width + 63) // 64
      elem_fold = Bits(64)(0)
      for c in range(num_chunks):
        hi = min(64 * c + 64, width)
        chunk = read_elem[64 * c:hi]
        if hi - 64 * c < 64:
          chunk = chunk.as_uint().as_uint(64).as_bits()
        r = (8 * c) % 64
        if r != 0:
          chunk = BitsSignal.concat([chunk[0:64 - r], chunk[64 - r:64]])
        elem_fold = elem_fold ^ chunk
      read_checksum = Wire(UInt(64))
      read_checksum.assign((read_checksum.as_bits() ^ elem_fold).as_uint().reg(
          ports.clk,
          ports.rst,
          rst_value=0,
          ce=read_resp_valid_snoop,
          name="read_checksum",
      ))
      esi.Telemetry.report_signal(
          ports.clk,
          ports.rst,
          esi.AppID("readChecksum"),
          read_checksum,
      )

  return ReadMem


@modparams
def WriteMem(width: int) -> Type[Module]:

  class WriteMem(Module):
    """Host memory burst (list) write test module.

        Issues a single windowed (list) write of 'flits' elements to sequential
        addresses starting at the base address (both configured via MMIO). The
        elements are streamed as a windowed list (num_items=1 -> one element per
        frame, 'last' on the final); each element's payload is a byte-level
        pattern derived from its frame index (see _ESITESTER_SEQ_SEED).

        MMIO command interface (via BurstCommand):
          0x00  Read : remaining element count (flits_left).
          0x08  Write: base address for the write.
          0x10  Write: number of list elements (flits) to write.
          0x18  Write: start the operation.

        Telemetry (AppID -> signal):
          addrCmdIssued     Count of burst commands issued (1 per command).
          addrCmdResponses  Count of write acks received.
          addrCmdResp/cycles  Active-window cycle count.
        """

    clk = Clock()
    rst = Reset()

    width_bits = Constant(UInt(32), width)

    @generator
    def construct(ports):
      clk = ports.clk
      rst = ports.rst

      done_wire = Wire(Channel(Bits(0)))
      cmd = BurstCommand(width)(
          clk=clk,
          rst=rst,
          hostmem_cmd_done=done_wire,
          instance_name="burst_command",
      )

      # Windowed (list) write: consume the single {base, tag, length}
      # burst request and stream 'length' elements to sequential
      # addresses (num_items=1 -> one element per frame).
      write_win = esi.HostMem.write_window(Bits(width), 1)
      lowered = write_win.lowered_type

      streaming = Wire(Bits(1))
      frame_xact = Wire(Bits(1))
      burst_ready = (~streaming).as_bits()
      burst, burst_valid = cmd.burst_req.unwrap(burst_ready)
      burst_accept = (burst_valid & ~streaming).as_bits()
      base_addr = burst.address.reg(
          clk=clk,
          rst=rst,
          rst_value=0,
          ce=burst_accept,
          name="base_addr",
      )
      total = burst.length.reg(
          clk=clk,
          rst=rst,
          rst_value=0,
          ce=burst_accept,
          name="total",
      )

      frame_counter = Counter(64)(
          clk=clk,
          rst=rst,
          clear=burst_accept,
          increment=frame_xact,
      )
      is_last = (frame_counter.out == (total -
                                       UInt(64)(1)).as_uint(64)).as_bits(1)
      last_accept = (frame_xact & is_last).as_bits()
      streaming.assign(
          ControlReg(
              clk=clk,
              rst=rst,
              asserts=[burst_accept],
              resets=[last_accept],
              name="streaming",
          ))

      # Byte-level data pattern: tile (seed ^ frame index) across the element's
      # bytes and XOR each byte with a distinct per-position mask so every byte
      # is unique. A read that fetches the wrong bytes is then caught at byte
      # granularity by the hostmembw data-integrity check.
      seq64 = frame_counter.out.as_bits() ^ Bits(64)(_ESITESTER_SEQ_SEED)
      num_bytes = (width + 7) // 8
      elem_bytes = [
          seq64[8 * (j % 8):8 * (j % 8) + 8] ^ Bits(8)((j * 0x9D) & 0xFF)
          for j in range(num_bytes)
      ]
      element = BitsSignal.concat(list(reversed(elem_bytes)))[0:width]
      frame_val = lowered({
          "address": base_addr,
          "tag": UInt(8)(0),
          "data": [element],
          "data_size": Bits(0)(0),
          "last": is_last,
      })
      frame_chan, frame_ready = Channel(write_win).wrap(
          write_win.wrap(frame_val), streaming)
      frame_xact.assign((streaming & frame_ready).as_bits())

      write_responses = esi.HostMem.write(
          appid=AppID("host"),
          req=frame_chan,
      )
      # Each write ack -> one completion token to BurstCommand.
      done_wire.assign(write_responses.transform(lambda resp: Bits(0)(0)))

  return WriteMem


@modparams
def ToHostDMATest(width: int):
  """Construct a module that sends the write count over a channel to the host
    the specified number of times. Exercises any DMA engine."""

  class ToHostDMATest(Module):
    """Transmit patterned values to the host a programmed number of times.

        A write to MMIO offset 0x0 programs `write_count`. Each message carries
        a byte-level pattern derived from its per-command transfer index (see
        ``_ESITESTER_SEQ_SEED``). The index advances on a successful channel
        handshake and resets for every command. The payload's final byte is
        truncated when `width` is not a multiple of eight.
        """

    clk = Clock()
    rst = Reset()

    width_bits = Constant(UInt(32), width)

    @generator
    def construct(ports):
      count_reached = Wire(Bits(1))
      count_valid = Wire(Bits(1))
      out_xact = Wire(Bits(1))

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

      # Output one byte-level pattern per command transfer. This lets the host
      # verify the complete payload, including every byte of wide messages.
      sequence_counter = Counter(64)(
          clk=ports.clk,
          rst=ports.rst,
          clear=write_count_ce,
          increment=out_xact,
      )
      seq64 = sequence_counter.out.as_bits() ^ Bits(64)(_ESITESTER_SEQ_SEED)
      num_bytes = (width + 7) // 8
      payload_bytes = [
          seq64[8 * (j % 8):8 * (j % 8) + 8] ^ Bits(8)((j * 0x9D) & 0xFF)
          for j in range(num_bytes)
      ]
      payload = BitsSignal.concat(list(reversed(payload_bytes)))[0:width]
      out_channel, out_channel_ready = Channel(UInt(width)).wrap(
          payload.as_uint(width), count_valid)
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
                    fromHostChecksum (AppID "fromHostChecksum"): Byte-position-sensitive
                        fold of all payloads accepted during the programmed receive sequence.

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

      # Fold every received payload so the host can verify all transferred
      # bytes, rather than only the low 64 bits of the final payload.
      in_bits = in_data.as_bits()
      num_chunks = (width + 63) // 64
      item_fold = Bits(64)(0)
      for c in range(num_chunks):
        hi = min(64 * c + 64, width)
        chunk = in_bits[64 * c:hi]
        if hi - 64 * c < 64:
          chunk = chunk.as_uint().as_uint(64).as_bits()
        r = (8 * c) % 64
        if r != 0:
          chunk = BitsSignal.concat([chunk[0:64 - r], chunk[64 - r:64]])
        item_fold = item_fold ^ chunk
      from_host_checksum = Wire(UInt(64))
      checksum_next = Mux(
          read_count_ce,
          (from_host_checksum.as_bits() ^ item_fold).as_uint(),
          UInt(64)(0),
      )
      from_host_checksum.assign(
          checksum_next.reg(
              clk=ports.clk,
              rst=ports.rst,
              rst_value=0,
              ce=read_count_ce | in_data_xact,
              name="from_host_checksum",
          ))
      esi.Telemetry.report_signal(
          ports.clk,
          ports.rst,
          esi.AppID("fromHostChecksum"),
          from_host_checksum,
      )

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


# Factory returning the same (to_host, from_host) engine pair the cosim_dma
# BSP wires in by default. Used below to exercise the per-request engine
# override on `ChannelService` requests: the resolver imports this path,
# calls it, and the returned pair replaces the default pair for just that
# one request's channel. Kept module-scope so it is importable as
# 'esiaccel.esitester._one_item_buffers_pair' from a service-request
# `options={"engine": ...}` value.
def _one_item_buffers_pair():
  from .bsp.dma import OneItemBuffersToHost, OneItemBuffersFromHost
  return (OneItemBuffersToHost, OneItemBuffersFromHost)


class ChannelTest(Module):
  """Test the ChannelService with a to_host producer and a from_host loopback.

  The 'producer' to_host port sends incrementing UInt(32) values. The number
  of values to send is specified via an MMIO write to offset 0x0. Reading MMIO
  returns the remaining count.

  The 'loopback_in'/'loopback_out' pair forwards from_host data back to_host."""

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    clk = ports.clk
    rst = ports.rst

    # MMIO interface for triggering the producer.
    cmd_chan_wire = Wire(Channel(esi.MMIOReadWriteCmdType))

    # State: remaining count and current value.
    remaining = Reg(UInt(32), clk=clk, rst=rst, rst_value=0)
    cur_value = Reg(UInt(32), clk=clk, rst=rst, rst_value=0)

    # Handle MMIO commands.
    cmd_ready = Wire(Bits(1))
    cmd, cmd_valid = cmd_chan_wire.unwrap(cmd_ready)
    is_write = cmd.write & cmd_valid
    # On write to offset 0x0, load the count and reset the current value.
    load_count = is_write & (cmd.offset == UInt(32)(0))

    # to_host: send incrementing values while remaining > 0.
    has_data = remaining != UInt(32)(0)
    data_chan, data_ready = Channel(UInt(32)).wrap(cur_value, has_data)
    sent = data_ready & has_data

    # Compute next state: load from MMIO takes priority, then decrement on send.
    next_remaining = Mux(
        load_count, Mux(sent, remaining, (remaining - UInt(32)(1)).as_uint(32)),
        cmd.data.as_uint(32))
    next_cur_value = Mux(
        load_count, Mux(sent, cur_value, (cur_value + UInt(32)(1)).as_uint(32)),
        UInt(32)(0))
    remaining.assign(next_remaining)
    cur_value.assign(next_cur_value)

    # MMIO read response: return remaining count.
    response_chan, response_ready = Channel(Bits(64)).wrap(
        remaining.as_bits(64), cmd_valid)
    cmd_ready.assign(response_ready)

    mmio_rw = esi.MMIO.read_write(appid=AppID("cmd"))
    mmio_rw_cmd_chan = mmio_rw.unpack(data=response_chan)["cmd"]
    cmd_chan_wire.assign(mmio_rw_cmd_chan)

    # Per-request engine override: on cosim_dma this resolves to the same
    # (OneItemBuffersToHost, OneItemBuffersFromHost) pair the BSP already
    # uses by default, so this exercises the resolver + substitution path
    # end-to-end without altering the observed runtime behavior.
    esi.ChannelService.to_host(
        AppID("producer"),
        data_chan,
        options={"engine": "esiaccel.esitester._one_item_buffers_pair"})

    # from_host -> to_host loopback.
    loopback_in = esi.ChannelService.from_host(AppID("loopback_in"), UInt(32))
    esi.ChannelService.to_host(AppID("loopback_out"), loopback_in)


class EsiTester(Module):
  """Top-level ESI test harness module.

    Contains submodules:
      CallbackTest            (single instance) – host callback via MMIO write (offset 0x10).
      LoopbackInOutAdd        (single instance) – function service adding constant 11.
      ChannelTest             (single instance) – ChannelService to_host and from_host loopback.
      MMIOAdd(add_amt)        instances for add_amt in {4, 9, 14} – MMIO read returns offset + add_amt.
      ReadMem(width)          for widths: 24, 32, 64, 72, 128, 256, 512, 534 – host memory read tests.
      WriteMem(width)         for widths: 24, 32, 64, 72, 128, 256, 512, 534 – host memory write tests.
      ToHostDMATest(width)    for widths: 24, 32, 64, 72, 128, 256, 512, 534 – DMA to host, cycle & count telemetry.
      FromHostDMATest(width)  for widths: 24, 32, 64, 72, 128, 256, 512, 534 – DMA from host, cycle telemetry.

    Width set used across Read/Write/DMA tests:
      widths = [24, 32, 64, 72, 128, 256, 512, 534]

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
    ChannelTest(
        clk=ports.clk,
        rst=ports.rst,
        instance_name="channel_test",
        appid=AppID("channel_test"),
    )
    StreamingAdder(1)(
        clk=ports.clk,
        rst=ports.rst,
        instance_name="streaming_adder",
        appid=AppID("streaming_adder"),
    )
    CoordTranslator(
        clk=ports.clk,
        rst=ports.rst,
        instance_name="coord_translator",
        appid=AppID("coord_translator"),
    )
    SerialCoordTranslator(
        clk=ports.clk,
        rst=ports.rst,
        instance_name="coord_translator_serial",
        appid=AppID("coord_translator_serial"),
    )
    AutoSerialCoordTranslator(
        clk=ports.clk,
        rst=ports.rst,
        instance_name="coord_translator_auto_serial",
        appid=AppID("coord_translator_auto_serial"),
    )

    for i in range(4, 18, 5):
      MMIOAdd(i)(instance_name=f"mmio_add_{i}", appid=AppID("mmio_add", i))

    for width in [24, 32, 64, 72, 128, 256, 512, 534]:
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
