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
from pycde.esi import ChannelService, FuncService, MMIO, MMIOReadWriteCmdType
from pycde.types import (Bits, Channel, ChannelSignaling, UInt)
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


class Top(Module):
  clk = Clock()
  rst = Reset()

  @generator
  def construct(ports):
    LoopbackInOutAdd(clk=ports.clk, rst=ports.rst, appid=AppID("loopback"))
    for i in range(4, 18, 5):
      MMIOClient(i)()
    MMIOReadWriteClient(clk=ports.clk, rst=ports.rst)
    ConstProducer(clk=ports.clk, rst=ports.rst)

    # Disable broken test.
    # Join(clk=ports.clk, rst=ports.rst)


if __name__ == "__main__":
  bsp = get_bsp(sys.argv[2] if len(sys.argv) > 2 else None)
  s = pycde.System(bsp(Top), name="ESILoopback", output_directory=sys.argv[1])
  s.compile()
  s.package()
