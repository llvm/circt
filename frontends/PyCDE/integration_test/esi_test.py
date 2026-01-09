# REQUIRES: esi-runtime, esi-cosim, rtl-sim
# RUN: rm -rf %t
# RUN: mkdir %t && cd %t
# RUN: %PYTHON% %s %t 2>&1
# RUN: esi-cosim.py --source %t -- %PYTHON% %S/test_software/esi_test.py cosim env

import pycde
from pycde import (AppID, Clock, Module, Reset, modparams, generator)
from pycde.bsp import get_bsp
from pycde.common import Constant, Input, Output
from pycde.constructs import ControlReg, Mux, Reg, Wire
from pycde.esi import (ChannelService, CallService, FuncService, MMIO,
                       MMIOReadWriteCmdType)
from pycde.testing import print_info
from pycde.types import (Bits, Channel, ChannelSignaling, StructType, UInt,
                         Window)
from pycde.handshake import Func

import sys


class LoopbackInOutAdd(Module):
  """Loopback the request from the host, adding 7 to the first 15 bits."""
  clk = Clock()
  rst = Reset()

  add_amt = Constant(UInt(16), 11)

  @generator
  def construct(ports):
    loopback = Wire(Channel(UInt(16), signaling=ChannelSignaling.FIFO))
    args = FuncService.get_call_chans(AppID("add"),
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

    xact, snooped_data = data_chan_buffered.snoop_xact()
    xact.when_true(
        lambda: print_info("LoopbackInOutAdd received: %p", snooped_data))


class CallbackTest(Module):
  """Call a function on the host when an MMIO write is received at offset
    0x10."""
  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    clk = ports.clk
    rst = ports.rst

    mmio_bundle = MMIO.read_write(appid=AppID("cmd"))
    data_resp_chan = Wire(Channel(Bits(64)))
    mmio_cmd_chan = mmio_bundle.unpack(data=data_resp_chan)["cmd"]
    cb_trigger, mmio_cmd_chan_fork = mmio_cmd_chan.fork(clk=clk, rst=rst)

    data_resp_chan.assign(mmio_cmd_chan_fork.transform(lambda cmd: cmd.data))

    cb_trigger_ready = Wire(Bits(1))
    cb_trigger_cmd, cb_trigger_valid = cb_trigger.unwrap(cb_trigger_ready)
    trigger = cb_trigger_valid & (cb_trigger_cmd.offset == UInt(32)(0x10))
    data_reg = cb_trigger_cmd.data.reg(clk, rst, ce=trigger)
    cb_chan, cb_trigger_ready_sig = Channel(UInt(64)).wrap(
        data_reg.as_uint(), trigger.reg(clk, rst))
    cb_trigger_ready.assign(cb_trigger_ready_sig)
    resp_chan = CallService.call(AppID("cb"), cb_chan, UInt(64))
    # TODO: Fix snoop_xact to work with unconumed channels.
    _, _ = resp_chan.unwrap(Bits(1)(1))
    xact, snooped_data = resp_chan.snoop_xact()
    xact.when_true(lambda: print_info("Callback received: %p", snooped_data))


@modparams
def MMIOClient(add_amt: int):

  class MMIOClient(Module):
    """A module which requests an MMIO address space and upon an MMIO read
    request, returns the <address offset into its space> + add_amt."""

    @generator
    def build(ports):
      mmio_read_bundle = MMIO.read(appid=AppID("mmio_client", add_amt))

      address_chan_wire = Wire(Channel(UInt(32)))
      address, address_valid = address_chan_wire.unwrap(1)
      response_data = (address + add_amt).as_bits(64)
      response_chan, response_ready = Channel(Bits(64)).wrap(
          response_data, address_valid)

      address_chan = mmio_read_bundle.unpack(data=response_chan)['offset']
      address_chan_wire.assign(address_chan)

  return MMIOClient


class MMIOReadWriteClient(Module):
  clk = Clock()
  rst = Reset()

  @generator
  def build(ports):
    mmio_read_write_bundle = MMIO.read_write(appid=AppID("mmio_rw_client"))

    cmd_chan_wire = Wire(Channel(MMIOReadWriteCmdType))
    resp_ready_wire = Wire(Bits(1))
    cmd, cmd_valid = cmd_chan_wire.unwrap(resp_ready_wire)

    add_amt = Reg(UInt(64),
                  name="add_amt",
                  clk=ports.clk,
                  rst=ports.rst,
                  rst_value=0,
                  ce=cmd_valid & cmd.write & (cmd.offset == 0x8).as_bits())
    add_amt.assign(cmd.data.as_uint())
    response_data = Mux(
        cmd.write,
        (cmd.offset + add_amt).as_bits(64),
        Bits(64)(0),
    )
    response_chan, response_ready = Channel(Bits(64)).wrap(
        response_data, cmd_valid)
    resp_ready_wire.assign(response_ready)

    cmd_chan = mmio_read_write_bundle.unpack(data=response_chan)['cmd']
    cmd_chan_wire.assign(cmd_chan)


class ConstProducer(Module):
  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    const = UInt(32)(42)
    xact = Wire(Bits(1))
    valid = ~ControlReg(ports.clk, ports.rst, [xact], [Bits(1)(0)])
    ch, ready = Channel(UInt(32)).wrap(const, valid)
    xact.assign(ready & valid)
    ChannelService.to_host(AppID("const_producer"), ch)


class JoinAddFunc(Func):
  # This test is broken since the DC dialect flow is broken. Leaving the code
  # here in case it gets fixed in the future.
  # https://github.com/llvm/circt/issues/7949 is the latest layer of the onion.

  a = Input(UInt(32))
  b = Input(UInt(32))
  x = Output(UInt(32))

  @generator
  def construct(ports):
    ports.x = (ports.a + ports.b).as_uint(32)


class Join(Module):
  # This test is broken since the JoinAddFunc function is broken.
  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    a = ChannelService.from_host(AppID("join_a"), UInt(32))
    b = ChannelService.from_host(AppID("join_b"), UInt(32))
    f = JoinAddFunc(clk=ports.clk, rst=ports.rst, a=a, b=b)
    ChannelService.to_host(AppID("join_x"), f.x)


# Define the struct with four fields
FourFieldStruct = StructType({
    "a": Bits(32),
    "b": Bits(32),
    "c": Bits(32),
    "d": Bits(32),
})

# Create a window that divides the struct into two frames
windowed_struct = Window(
    "four_field_window", FourFieldStruct,
    [Window.Frame("frame1", ["a", "b"]),
     Window.Frame("frame2", ["c", "d"])])


class WindowToStructFunc(Module):
  """Exposes a function that accepts a windowed struct (four fields split into
  two frames) and returns the reassembled struct without windowing.

  The input struct has four UInt(32) fields: a, b, c, d.
  The window divides these into two frames:
    - Frame 1: fields a and b
    - Frame 2: fields c and d

  Frames arrive in-order. The function reads both frames, reassembles the
  complete struct, and outputs it.
  """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):

    # Result is the complete struct (no windowing)
    result_chan = Wire(Channel(FourFieldStruct))
    args = FuncService.get_call_chans(AppID("struct_from_window"),
                                      arg_type=windowed_struct,
                                      result=result_chan)

    # State register to track which frame we're expecting (0 = frame1, 1 = frame2)
    expecting_frame2 = Reg(Bits(1),
                           name="expecting_frame2",
                           clk=ports.clk,
                           rst=ports.rst,
                           rst_value=0)

    # Registers to hold the values from frame1
    a_reg = Reg(Bits(32),
                name="a_reg",
                clk=ports.clk,
                rst=ports.rst,
                rst_value=0)
    b_reg = Reg(Bits(32),
                name="b_reg",
                clk=ports.clk,
                rst=ports.rst,
                rst_value=0)

    # Unwrap the incoming channel
    ready = Wire(Bits(1))
    window_data, window_valid = args.unwrap(ready)

    # Unwrap the window to get the union of frames
    frame_union = window_data.unwrap()

    # Extract data from both frames (only one is valid at a time based on state)
    # Access the frame structs through the union - the data is reinterpreted
    # based on which frame we're expecting
    frame1_data = frame_union["frame1"]
    frame2_data = frame_union["frame2"]

    # When we receive frame1, store a and b
    got_frame1 = window_valid & ~expecting_frame2
    a_reg.assign(Mux(got_frame1, a_reg, frame1_data.a))
    b_reg.assign(Mux(got_frame1, b_reg, frame1_data.b))

    # When we receive frame2, we can output the complete struct
    got_frame2 = window_valid & expecting_frame2

    # Update state: after receiving frame1, expect frame2; after frame2, expect frame1
    expecting_frame2.assign(
        Mux(window_valid, expecting_frame2, ~expecting_frame2))

    # Output the reassembled struct when we have frame2
    output_struct = FourFieldStruct({
        "a": a_reg,
        "b": b_reg,
        "c": frame2_data["c"],
        "d": frame2_data["d"]
    })
    result_internal, result_ready = Channel(FourFieldStruct).wrap(
        output_struct, got_frame2)

    # We're ready to accept when either:
    # - We're waiting for frame1 (always ready)
    # - We're waiting for frame2 and downstream is ready
    ready.assign(~expecting_frame2 | result_ready)
    result_chan.assign(result_internal)


class StructToWindowFunc(Module):
  """Exposes a function that accepts a complete struct and returns it as a
  windowed struct split into two frames.

  This is the inverse of WindowedStructFunc.

  The input struct has four Bits(32) fields: a, b, c, d.
  The output window divides these into two frames:
    - Frame 1: fields a and b
    - Frame 2: fields c and d

  The function reads the complete struct, then outputs two frames in order.
  """

  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    # Result is the windowed struct
    result_chan = Wire(Channel(windowed_struct))
    args = FuncService.get_call_chans(AppID("struct_to_window"),
                                      arg_type=FourFieldStruct,
                                      result=result_chan)

    # State register to track which frame we're sending (0 = frame1, 1 = frame2)
    sending_frame2 = Reg(Bits(1),
                         name="sending_frame2",
                         clk=ports.clk,
                         rst=ports.rst,
                         rst_value=0)

    # Register to indicate we have a valid struct to send
    have_struct = Reg(Bits(1),
                      name="have_struct",
                      clk=ports.clk,
                      rst=ports.rst,
                      rst_value=0)

    # Registers to hold the input struct fields
    a_reg = Reg(Bits(32),
                name="a_reg",
                clk=ports.clk,
                rst=ports.rst,
                rst_value=0)
    b_reg = Reg(Bits(32),
                name="b_reg",
                clk=ports.clk,
                rst=ports.rst,
                rst_value=0)
    c_reg = Reg(Bits(32),
                name="c_reg",
                clk=ports.clk,
                rst=ports.rst,
                rst_value=0)
    d_reg = Reg(Bits(32),
                name="d_reg",
                clk=ports.clk,
                rst=ports.rst,
                rst_value=0)

    # Unwrap the incoming channel
    ready = Wire(Bits(1))
    struct_data, struct_valid = args.unwrap(ready)

    # Get the lowered type (a union of frame structs)
    lowered_type = windowed_struct.lowered_type

    # Create frame1 and frame2 data
    frame1_struct = lowered_type.frame1({"a": a_reg, "b": b_reg})
    frame2_struct = lowered_type.frame2({"c": c_reg, "d": d_reg})

    # Select which frame to output based on state
    frame1_union = lowered_type(("frame1", frame1_struct))
    frame2_union = lowered_type(("frame2", frame2_struct))

    # Mux between frames based on state
    output_union = Mux(sending_frame2, frame1_union, frame2_union)
    output_window = windowed_struct.wrap(output_union)

    # Output is valid when we have a struct to send
    output_valid = have_struct
    result_internal, result_ready = Channel(windowed_struct).wrap(
        output_window, output_valid)

    # Compute state transitions
    frame_sent = output_valid & result_ready
    store_struct = struct_valid & ~have_struct
    done_sending = frame_sent & sending_frame2

    # Store the incoming struct when we receive it and aren't busy
    a_reg.assign(Mux(store_struct, a_reg, struct_data["a"]))
    b_reg.assign(Mux(store_struct, b_reg, struct_data["b"]))
    c_reg.assign(Mux(store_struct, c_reg, struct_data["c"]))
    d_reg.assign(Mux(store_struct, d_reg, struct_data["d"]))

    # have_struct: set when storing, clear when done sending both frames
    have_struct.assign(
        Mux(store_struct, Mux(done_sending, have_struct,
                              Bits(1)(0)),
            Bits(1)(1)))

    # sending_frame2: set after sending frame1, clear after sending frame2
    sending_frame2.assign(
        Mux(frame_sent & ~sending_frame2,
            Mux(done_sending, sending_frame2,
                Bits(1)(0)),
            Bits(1)(1)))

    # We're ready to accept a new struct when we don't have one
    ready.assign(~have_struct)
    result_chan.assign(result_internal)


class Top(Module):
  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    CallbackTest(clk=ports.clk, rst=ports.rst, appid=AppID("callback"))
    LoopbackInOutAdd(clk=ports.clk, rst=ports.rst, appid=AppID("loopback"))
    for i in range(4, 18, 5):
      MMIOClient(i)()
    MMIOReadWriteClient(clk=ports.clk, rst=ports.rst)
    ConstProducer(clk=ports.clk, rst=ports.rst)
    WindowToStructFunc(clk=ports.clk, rst=ports.rst)
    StructToWindowFunc(clk=ports.clk, rst=ports.rst)

    # Disable broken test.
    # Join(clk=ports.clk, rst=ports.rst)


if __name__ == "__main__":
  bsp = get_bsp(sys.argv[2] if len(sys.argv) > 2 else None)
  s = pycde.System(bsp(Top, core_freq=20_000_000),
                   name="ESILoopback",
                   output_directory=sys.argv[1])
  s.compile()
  s.package()
