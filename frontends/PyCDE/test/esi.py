# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1 | FileCheck %s

import pycde
from pycde import (Clock, Input, InputChannel, OutputChannel, module, generator,
                   types)
from pycde import esi
from pycde.common import Output
from pycde.pycde_types import ChannelType
from pycde.value import BitVectorValue, ChannelValue


@module
class Producer:
  clk = Input(types.i1)
  int_out = OutputChannel(types.i32)

  @generator
  def construct(ports):
    chan = esi.HostComms.from_host(types.i32, "loopback_in")
    ports.int_out = chan


@module
class Consumer:
  clk = Input(types.i1)
  int_in = InputChannel(types.i32)

  @generator
  def construct(ports):
    esi.HostComms.to_host(ports.int_in, "loopback_out")


# @module
# class LoopbackTop:
#   clk = Clock(types.i1)
#   rst = Input(types.i1)

#   @generator
#   def construct(ports):
#     p = Producer(clk=ports.clk)
#     Consumer(clk=ports.clk, int_in=p.int_out)
#     # Use Cosim to implement the standard 'HostComms' service.
#     esi.Cosim(esi.HostComms, ports.clk, ports.rst)


@esi.ServiceImplementation(esi.HostComms)
class MultiplexerService:
  clk = Clock()
  rst = Input(types.i1)

  # Underlying channel is an untyped, 256-bit LI channel.
  trunk_in = Input(types.i256)
  trunk_in_valid = Input(types.i1)
  trunk_in_ready = Output(types.i1)
  trunk_out = Output(types.i256)
  trunk_out_valid = Output(types.i1)
  trunk_out_ready = Input(types.i1)

  @generator
  def generate(ports, channels):

    print("generating")
    input_reqs = channels.to_server_reqs
    if len(input_reqs) > 1:
      raise Exception("Multiple to_server requests not supported")
    MultiplexerService.unwrap_and_pad(ports, input_reqs[0][1])

    output_reqs = channels.to_client_reqs
    if len(output_reqs) > 1:
      raise Exception("Multiple to_client requests not supported")
    idx, req_name, req_type = output_reqs[0]
    output_chan, ready = MultiplexerService.slice_and_wrap(ports, req_type)
    channels.set_output(idx, output_chan)
    ports.trunk_in_ready = ready

  @staticmethod
  def slice_and_wrap(ports, channel_type: ChannelType):
    assert (channel_type.inner_type.width <= 256)
    sliced = ports.trunk_in[:channel_type.inner_type.width]
    return channel_type.wrap(sliced, ports.trunk_in_valid)

  @staticmethod
  def unwrap_and_pad(ports, input_channel: ChannelValue):
    """
    Unwrap the input channel and pad it to 256 bits.
    """
    (data, valid) = input_channel.unwrap(ports.trunk_out_ready)
    assert isinstance(data, BitVectorValue)
    assert len(data) <= 256
    ports.trunk_out = data.pad_or_truncate(256)
    ports.trunk_out_valid = valid


@module
class MultiplexerTop:
  clk = Clock(types.i1)
  rst = Input(types.i1)

  trunk_in = Input(types.i256)
  trunk_in_valid = Input(types.i1)
  trunk_in_ready = Output(types.i1)
  trunk_out = Output(types.i256)
  trunk_out_valid = Output(types.i1)
  trunk_out_ready = Input(types.i1)

  @generator
  def construct(ports):
    m = MultiplexerService(clk=ports.clk,
                           rst=ports.rst,
                           trunk_in=ports.trunk_in,
                           trunk_in_valid=ports.trunk_in_valid,
                           trunk_out_ready=ports.trunk_out_ready)

    ports.trunk_in_ready = m.trunk_in_ready
    ports.trunk_out = m.trunk_out
    ports.trunk_out_valid = m.trunk_out_valid

    p = Producer(clk=ports.clk)
    Consumer(clk=ports.clk, int_in=p.int_out)


s = pycde.System([MultiplexerTop], name="EsiSys")

s.generate()
s.print()
try:
  s.run_passes()
finally:
  s.print()

# CHECK-LABEL: msft.module @Top {} (%clk: i1, %rst: i1) attributes {fileName = "Top.sv"} {
# CHECK:         %Producer.int_out = msft.instance @Producer @Producer(%clk)  : (i1) -> !esi.channel<i32>
# CHECK:         msft.instance @Consumer @Consumer(%clk, %Producer.int_out)  : (i1, !esi.channel<i32>) -> ()
# CHECK:         esi.service.instance @HostComms impl as "cosim"(%clk, %rst) : (i1, i1) -> ()
# CHECK:         msft.output
# CHECK-LABEL: msft.module @Producer {} (%clk: i1) -> (int_out: !esi.channel<i32>) attributes {fileName = "Producer.sv"} {
# CHECK:         [[R0:%.+]] = esi.service.req.to_client <@HostComms::@from_host>(["loopback_in"]) : !esi.channel<i32>
# CHECK:         msft.output [[R0]] : !esi.channel<i32>
# CHECK-LABEL: msft.module @Consumer {} (%clk: i1, %int_in: !esi.channel<i32>) attributes {fileName = "Consumer.sv"} {
# CHECK:         esi.service.req.to_server %int_in -> <@HostComms::@to_host>(["loopback_out"]) : !esi.channel<i32>
# CHECK:         msft.output
# CHECK-LABEL: esi.service.decl @HostComms {
# CHECK:         esi.service.to_server @to_host : !esi.channel<!esi.any>
# CHECK:         esi.service.to_client @from_host : !esi.channel<!esi.any>
