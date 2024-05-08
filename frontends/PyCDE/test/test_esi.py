# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1 | FileCheck %s

from pycde import (Clock, Input, InputChannel, OutputChannel, Module, generator,
                   types)
from pycde import esi
from pycde.common import AppID, RecvBundle, SendBundle
from pycde.constructs import Wire
from pycde.esi import MMIO
from pycde.module import Metadata
from pycde.types import (Bits, Bundle, BundledChannel, Channel,
                         ChannelDirection, UInt, ClockType)
from pycde.testing import unittestmodule

TestBundle = Bundle([
    BundledChannel("resp", ChannelDirection.FROM, Bits(16)),
    BundledChannel("req", ChannelDirection.TO, Bits(24))
])

TestFromBundle = Bundle([BundledChannel("ch1", ChannelDirection.TO, Bits(32))])


@esi.ServiceDecl
class HostComms:
  req_resp = TestBundle
  from_host = TestFromBundle


# CHECK: esi.manifest.sym @LoopbackInOutTop name "LoopbackInOut" {{.*}}version "0.1" {bar = "baz", foo = 1 : i64}


# CHECK-LABEL: hw.module @LoopbackInOutTop(in %clk : !seq.clock, in %rst : i1)
# CHECK:         esi.service.instance #esi.appid<"cosim"[0]> svc @HostComms impl as "cosim"(%clk, %rst) : (!seq.clock, i1) -> ()
# CHECK:         [[B0:%.+]] = esi.service.req <@HostComms::@req_resp>(#esi.appid<"loopback_inout"[0]>) : !esi.bundle<[!esi.channel<i16> from "resp", !esi.channel<i24> to "req"]>
# CHECK:         %req = esi.bundle.unpack %chanOutput from [[B0]] : !esi.bundle<[!esi.channel<i16> from "resp", !esi.channel<i24> to "req"]>
# CHECK:         %rawOutput, %valid = esi.unwrap.vr %req, %ready : i24
# CHECK:         [[R0:%.+]] = comb.extract %rawOutput from 0 : (i24) -> i16
# CHECK:         %chanOutput, %ready = esi.wrap.vr [[R0]], %valid : i16
@unittestmodule(print=True)
class LoopbackInOutTop(Module):
  clk = Clock()
  rst = Input(types.i1)

  metadata = Metadata(
      name="LoopbackInOut",
      version="0.1",
      misc={
          "foo": 1,
          "bar": "baz"
      },
  )

  @generator
  def construct(self):
    # Use Cosim to implement the 'HostComms' service.
    esi.Cosim(HostComms, self.clk, self.rst)

    loopback = Wire(types.channel(types.i16))
    call_bundle = HostComms.req_resp(AppID("loopback_inout", 0))
    froms = call_bundle.unpack(resp=loopback)
    from_host = froms['req']

    ready = Wire(types.i1)
    wide_data, valid = from_host.unwrap(ready)
    data = wide_data[0:16]
    data_chan, data_ready = loopback.type.wrap(data, valid)
    ready.assign(data_ready)
    loopback.assign(data_chan)


CallBundle = Bundle([
    BundledChannel("result", ChannelDirection.FROM, Bits(16)),
    BundledChannel("arg", ChannelDirection.TO, Bits(24))
])


# CHECK-LABEL:  hw.module @LoopbackCall(in %clk : !seq.clock, in %rst : i1) attributes {output_file = #hw.output_file<"LoopbackCall.sv", includeReplicatedOps>} {
# CHECK-NEXT:     [[R0:%.+]] = esi.service.req <@_FuncService::@call>(#esi.appid<"loopback">) : !esi.bundle<[!esi.channel<i24> to "arg", !esi.channel<i16> from "result"]>
# CHECK-NEXT:     %arg = esi.bundle.unpack %chanOutput from [[R0]] : !esi.bundle<[!esi.channel<i24> to "arg", !esi.channel<i16> from "result"]>
# CHECK-NEXT:     %rawOutput, %valid = esi.unwrap.vr %arg, %ready : i24
# CHECK-NEXT:     [[R1:%.+]] = comb.extract %rawOutput from 0 : (i24) -> i16
# CHECK-NEXT:     %chanOutput, %ready = esi.wrap.vr [[R1]], %valid : i16
# CHECK-NEXT:     hw.output
# CHECK-NEXT:   }
# CHECK-NEXT:   esi.service.std.func @_FuncService
@unittestmodule(print=True)
class LoopbackCall(Module):
  clk = Clock()
  rst = Input(Bits(1))

  metadata = Metadata(
      name="LoopbackCall",
      version="0.1",
  )

  @generator
  def construct(self):
    loopback = Wire(types.channel(types.i16))
    args = esi.FuncService.get_call_chans(name=AppID("loopback"),
                                          arg_type=Bits(24),
                                          result=loopback)

    ready = Wire(types.i1)
    wide_data, valid = args.unwrap(ready)
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


# CHECK-LABEL:  hw.module @MMIOReq()
# CHECK-NEXT:     %c0_i32 = hw.constant 0 : i32
# CHECK-NEXT:     %false = hw.constant false
# CHECK-NEXT:     [[B:%.+]] = esi.service.req <@MMIO::@read>(#esi.appid<"mmio_req">) : !esi.bundle<[!esi.channel<i32> to "offset", !esi.channel<i32> from "data"]>
# CHECK-NEXT:     %chanOutput, %ready = esi.wrap.vr %c0_i32, %false : i32
# CHECK-NEXT:     %offset = esi.bundle.unpack %chanOutput from [[B]] : !esi.bundle<[!esi.channel<i32> to "offset", !esi.channel<i32> from "data"]>
@unittestmodule(esi_sys=True)
class MMIOReq(Module):

  @generator
  def build(ports):
    c32 = Bits(32)(0)
    c1 = Bits(1)(0)

    read_bundle = MMIO.read(AppID("mmio_req"))

    data, _ = Channel(Bits(32)).wrap(c32, c1)
    _ = read_bundle.unpack(data=data)
