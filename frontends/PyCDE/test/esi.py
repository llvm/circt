# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1 | FileCheck %s

from pycde import (Clock, Input, InputChannel, OutputChannel, module, generator,
                   types)
from pycde import esi
from pycde.common import Output
from pycde.constructs import Wire
from pycde.pycde_types import ChannelType
from pycde.testing import unittestmodule
from pycde.value import BitVectorValue, ChannelValue


@esi.ServiceDecl
class HostComms:
  to_host = esi.ToServer(types.any)
  from_host = esi.FromServer(types.any)
  req_resp = esi.ToFromServer(to_server_type=types.i16,
                              to_client_type=types.i32)


@module
class Producer:
  clk = Input(types.i1)
  int_out = OutputChannel(types.i32)

  @generator
  def construct(ports):
    chan = HostComms.from_host("loopback_in", types.i32)
    ports.int_out = chan


@module
class Consumer:
  clk = Input(types.i1)
  int_in = InputChannel(types.i32)

  @generator
  def construct(ports):
    HostComms.to_host(ports.int_in, "loopback_out")


# CHECK-LABEL: msft.module @LoopbackTop {} (%clk: i1, %rst: i1)
# CHECK:         %Producer.int_out = msft.instance @Producer @Producer(%clk)  : (i1) -> !esi.channel<i32>
# CHECK:         msft.instance @Consumer @Consumer(%clk, %Producer.int_out)  : (i1, !esi.channel<i32>) -> ()
# CHECK:         esi.service.instance svc @HostComms impl as "cosim"(%clk, %rst) : (i1, i1) -> ()
# CHECK:         msft.output
# CHECK-LABEL: msft.module @Producer {} (%clk: i1) -> (int_out: !esi.channel<i32>)
# CHECK:         [[R0:%.+]] = esi.service.req.to_client <@HostComms::@from_host>(["loopback_in"]) : !esi.channel<i32>
# CHECK:         msft.output [[R0]] : !esi.channel<i32>
# CHECK-LABEL: msft.module @Consumer {} (%clk: i1, %int_in: !esi.channel<i32>)
# CHECK:         esi.service.req.to_server %int_in -> <@HostComms::@to_host>(["loopback_out"]) : !esi.channel<i32>
# CHECK:         msft.output
# CHECK-LABEL: esi.service.decl @HostComms {
# CHECK:         esi.service.to_server @to_host : !esi.channel<!esi.any>
# CHECK:         esi.service.to_client @from_host : !esi.channel<!esi.any>


@unittestmodule(print=True)
class LoopbackTop:
  clk = Clock(types.i1)
  rst = Input(types.i1)

  @generator
  def construct(ports):
    p = Producer(clk=ports.clk)
    Consumer(clk=ports.clk, int_in=p.int_out)
    # Use Cosim to implement the standard 'HostComms' service.
    esi.Cosim(HostComms, ports.clk, ports.rst)


# CHECK-LABEL: msft.module @LoopbackInOutTop {} (%clk: i1, %rst: i1)
# CHECK:         esi.service.instance svc @HostComms impl as "cosim"(%clk, %rst) : (i1, i1) -> ()
# CHECK:         %0 = esi.service.req.inout %chanOutput -> <@HostComms::@req_resp>(["loopback_inout"]) : !esi.channel<i16> -> !esi.channel<i32>
# CHECK:         %rawOutput, %valid = esi.unwrap.vr %0, %ready : i32
# CHECK:         %1 = comb.extract %rawOutput from 0 : (i32) -> i16
# CHECK:         %chanOutput, %ready = esi.wrap.vr %1, %valid : i16
# CHECK:         msft.output
@unittestmodule(print=True)
class LoopbackInOutTop:
  clk = Clock(types.i1)
  rst = Input(types.i1)

  @generator
  def construct(ports):
    # Use Cosim to implement the standard 'HostComms' service.
    esi.Cosim(HostComms, ports.clk, ports.rst)

    loopback = Wire(types.channel(types.i16))
    from_host = HostComms.req_resp(loopback, "loopback_inout")
    ready = Wire(types.i1)
    wide_data, valid = from_host.unwrap(ready)
    data = wide_data[0:16]
    data_chan, data_ready = loopback.type.wrap(data, valid)
    ready.assign(data_ready)
    loopback.assign(data_chan)


@esi.ServiceImplementation(HostComms)
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

    input_reqs = channels.to_server_reqs
    if len(input_reqs) > 1:
      raise Exception("Multiple to_server requests not supported")
    MultiplexerService.unwrap_and_pad(ports, input_reqs[0])

    output_reqs = channels.to_client_reqs
    if len(output_reqs) > 1:
      raise Exception("Multiple to_client requests not supported")
    output_req = output_reqs[0]
    output_chan, ready = MultiplexerService.slice_and_wrap(
        ports, output_req.type)
    output_req.assign(output_chan)
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


# CHECK-LABEL: hw.module @MultiplexerTop<__INST_HIER: none = "INSTANTIATE_WITH_INSTANCE_PATH">(%clk: i1, %rst: i1, %trunk_in: i256, %trunk_in_valid: i1, %trunk_out_ready: i1) -> (trunk_in_ready: i1, trunk_out: i256, trunk_out_valid: i1) attributes {output_file = #hw.output_file<"MultiplexerTop.sv", includeReplicatedOps>} {
# CHECK:         %c0_i224 = hw.constant 0 : i224
# CHECK:         [[r0:%.+]] = comb.concat %c0_i224, %Consumer.loopback_out : i224, i32
# CHECK:         [[r1:%.+]] = comb.extract %trunk_in from 0 {sv.namehint = "trunk_in_0upto32"} : (i256) -> i32
# CHECK:         %Producer.int_out, %Producer.int_out_valid, %Producer.loopback_in_ready = hw.instance "Producer" sym @Producer @Producer<__INST_HIER: none = #hw.param.expr.str.concat<#hw.param.decl.ref<"__INST_HIER">, ".Producer">>(clk: %clk: i1, loopback_in: [[r1]]: i32, loopback_in_valid: %trunk_in_valid: i1, int_out_ready: %Consumer.int_in_ready: i1) -> (int_out: i32, int_out_valid: i1, loopback_in_ready: i1)
# CHECK:         %Consumer.loopback_out, %Consumer.loopback_out_valid, %Consumer.int_in_ready = hw.instance "Consumer" sym @Consumer @Consumer<__INST_HIER: none = #hw.param.expr.str.concat<#hw.param.decl.ref<"__INST_HIER">, ".Consumer">>(clk: %clk: i1, int_in: %Producer.int_out: i32, int_in_valid: %Producer.int_out_valid: i1, loopback_out_ready: %trunk_out_ready: i1) -> (loopback_out: i32, loopback_out_valid: i1, int_in_ready: i1)
# CHECK:         hw.output %Producer.loopback_in_ready, [[r0]], %Consumer.loopback_out_valid : i1, i256, i1
# CHECK-LABEL: hw.module @Producer<__INST_HIER: none = "INSTANTIATE_WITH_INSTANCE_PATH">(%clk: i1, %loopback_in: i32, %loopback_in_valid: i1, %int_out_ready: i1) -> (int_out: i32, int_out_valid: i1, loopback_in_ready: i1) attributes {output_file = #hw.output_file<"Producer.sv", includeReplicatedOps>} {
# CHECK:         hw.output %loopback_in, %loopback_in_valid, %int_out_ready : i32, i1, i1
# CHECK-LABEL: hw.module @Consumer<__INST_HIER: none = "INSTANTIATE_WITH_INSTANCE_PATH">(%clk: i1, %int_in: i32, %int_in_valid: i1, %loopback_out_ready: i1) -> (loopback_out: i32, loopback_out_valid: i1, int_in_ready: i1) attributes {output_file = #hw.output_file<"Consumer.sv", includeReplicatedOps>} {
# CHECK:         hw.output %int_in, %int_in_valid, %loopback_out_ready : i32, i1, i1


@unittestmodule(run_passes=True, print_after_passes=True, emit_outputs=True)
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
