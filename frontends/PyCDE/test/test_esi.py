# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1 | FileCheck %s

from pycde import (Clock, Input, InputChannel, OutputChannel, Module, generator,
                   types)
from pycde import esi
from pycde.common import AppID, Output, RecvBundle, SendBundle
from pycde.constructs import Wire
from pycde.types import (Bits, Bundle, BundledChannel, Channel,
                         ChannelDirection, ChannelSignaling, UInt, ClockType)
from pycde.testing import unittestmodule
from pycde.signals import BitVectorSignal, ChannelSignal

TestBundle = Bundle([
    BundledChannel("resp", ChannelDirection.TO, Bits(16)),
    BundledChannel("req", ChannelDirection.FROM, Bits(24))
])

TestFromBundle = Bundle(
    [BundledChannel("ch1", ChannelDirection.FROM, Bits(32))])


@esi.ServiceDecl
class HostComms:
  req_resp = TestBundle
  from_host = TestFromBundle


# CHECK-LABEL: hw.module @LoopbackInOutTop(in %clk : !seq.clock, in %rst : i1)
# CHECK:         esi.service.instance #esi.appid<"cosim"[0]> svc @HostComms impl as "cosim"(%clk, %rst) : (!seq.clock, i1) -> ()
# CHECK:         %bundle, %req = esi.bundle.pack %chanOutput : !esi.bundle<[!esi.channel<i16> to "resp", !esi.channel<i24> from "req"]>
# CHECK:         esi.service.req.to_server %bundle -> <@HostComms::@req_resp>(#esi.appid<"loopback_inout"[0]>) : !esi.bundle<[!esi.channel<i16> to "resp", !esi.channel<i24> from "req"]>
# CHECK:         %rawOutput, %valid = esi.unwrap.vr %req, %ready : i24
# CHECK:         [[R0:%.+]] = comb.extract %rawOutput from 0 : (i24) -> i16
# CHECK:         %chanOutput, %ready = esi.wrap.vr [[R0]], %valid : i16
@unittestmodule(print=True)
class LoopbackInOutTop(Module):
  clk = Clock()
  rst = Input(types.i1)

  @generator
  def construct(self):
    # Use Cosim to implement the 'HostComms' service.
    esi.Cosim(HostComms, self.clk, self.rst)

    loopback = Wire(types.channel(types.i16))
    call_bundle, froms = TestBundle.pack(resp=loopback)
    from_host = froms['req']
    HostComms.req_resp(call_bundle, AppID("loopback_inout", 0))

    ready = Wire(types.i1)
    wide_data, valid = from_host.unwrap(ready)
    data = wide_data[0:16]
    data_chan, data_ready = loopback.type.wrap(data, valid)
    ready.assign(data_ready)
    loopback.assign(data_chan)


class Producer(Module):
  clk = Clock()
  int_out = OutputChannel(types.i32)

  @generator
  def construct(ports):
    b, chans = TestFromBundle.pack()
    HostComms.from_host(b, AppID("producer", 0))
    ports.int_out = chans['ch1']


# TODO: fixme
# @unittestmodule(run_passes=True, print_after_passes=True, emit_outputs=True)
class PureTest(esi.PureModule):

  @generator
  def construct(ports):
    clk = esi.PureModule.input_port("clk", ClockType())
    rst = esi.PureModule.input_port("rst", Bits(1))
    esi.Cosim(HostComms, clk, rst)

    p2 = Producer(clk=clk, instance_name="prod2")
    esi.PureModule.output_port("p2_int", p2.int_out)
    esi.PureModule.param("FOO", Bits(5))
    esi.PureModule.param("STR")


ExStruct = types.struct({
    'a': Bits(4),
    'b': UInt(32),
})

Bundle1 = Bundle([
    BundledChannel("req", ChannelDirection.TO, types.channel(types.i32)),
    BundledChannel("resp", ChannelDirection.FROM, types.channel(types.i1)),
])
# CHECK: Bundle<[('req', ChannelDirection.TO, Channel<Bits<32>, ValidReady>), ('resp', ChannelDirection.FROM, Channel<Bits<1>, ValidReady>)]>
print(Bundle1)
# CHECK: Channel<Bits<1>, ValidReady>
print(Bundle1.resp)


# CHECK-LABEL:  hw.module @SendBundleTest(in %s1_in : !esi.channel<i32>, out b_send : !esi.bundle<[!esi.channel<i32> to "req", !esi.channel<i1> from "resp"]>, out i1_out : !esi.channel<i1>) attributes {output_file = #hw.output_file<"SendBundleTest.sv", includeReplicatedOps>} {
# CHECK-NEXT:     %bundle, %resp = esi.bundle.pack %s1_in : !esi.bundle<[!esi.channel<i32> to "req", !esi.channel<i1> from "resp"]>
# CHECK-NEXT:     hw.output %bundle, %resp : !esi.bundle<[!esi.channel<i32> to "req", !esi.channel<i1> from "resp"]>, !esi.channel<i1>
@unittestmodule()
class SendBundleTest(Module):
  b_send = SendBundle(Bundle1)
  s1_in = InputChannel(types.i32)
  i1_out = OutputChannel(types.i1)

  @generator
  def build(self):
    self.b_send, from_chans = Bundle1.pack(req=self.s1_in)
    self.i1_out = from_chans.resp


# CHECK-LABEL:  hw.module @RecvBundleTest(in %b_recv : !esi.bundle<[!esi.channel<i32> to "req", !esi.channel<i1> from "resp"]>, in %i1_in : !esi.channel<i1>, out s1_out : !esi.channel<i32>) attributes {output_file = #hw.output_file<"RecvBundleTest.sv", includeReplicatedOps>} {
# CHECK-NEXT:     %req = esi.bundle.unpack %i1_in from %b_recv : !esi.bundle<[!esi.channel<i32> to "req", !esi.channel<i1> from "resp"]>
# CHECK-NEXT:     hw.output %req : !esi.channel<i32>
@unittestmodule()
class RecvBundleTest(Module):
  b_recv = RecvBundle(Bundle1)
  s1_out = OutputChannel(types.i32)
  i1_in = InputChannel(types.i1)

  @generator
  def build(self):
    to_channels = self.b_recv.unpack(resp=self.i1_in)
    self.s1_out = to_channels['req']
