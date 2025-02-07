# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1 | FileCheck %s

from pycde import (Clock, Input, InputChannel, Output, OutputChannel, Module,
                   Reset, generator, types)
from pycde import esi
from pycde.common import AppID, Constant, RecvBundle, SendBundle
from pycde.constructs import Wire
from pycde.esi import HostMem, MMIO
from pycde.module import Metadata
from pycde.support import _obj_to_attribute, optional_dict_to_dict_attr
from pycde.types import (Bits, Bundle, BundledChannel, Channel,
                         ChannelDirection, ChannelSignaling, UInt, StructType,
                         ClockType)
from pycde.testing import unittestmodule

# CHECK: Channel<UInt<4>, ValidReady>
print(Channel(UInt(4)))

# CHECK: Channel<UInt<4>, FIFO>
print(Channel(UInt(4), ChannelSignaling.FIFO))

# CHECK: Channel<UInt<4>, ValidReady(1)>
print(Channel(UInt(4), ChannelSignaling.ValidReady, 1))

TestBundle = Bundle([
    BundledChannel("resp", ChannelDirection.FROM, Bits(16)),
    BundledChannel("req", ChannelDirection.TO, Bits(24))
])

TestFromBundle = Bundle([BundledChannel("ch1", ChannelDirection.TO, Bits(32))])

# CHECK: foo
print(AppID("foo"))

# CHECK: {{^}}#esi.appid<"foo">{{$}}
print(_obj_to_attribute(AppID("foo")))

# CHECK: {bar = 6 : i64, foo = 5 : i64}
print(optional_dict_to_dict_attr({"foo": 5, "bar": 6}))
# CHECK: {}
print(optional_dict_to_dict_attr(None))


@esi.ServiceDecl
class HostComms:
  req_resp = TestBundle
  from_host = TestFromBundle


# CHECK: esi.manifest.sym @LoopbackInOutTop name "LoopbackInOut" {{.*}}version "0.1" {bar = "baz", foo = 1 : i64}
# CHECK: esi.manifest.constants @LoopbackInOutTop {c1 = 54 : ui8}


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

  c1 = Constant(UInt(8), 54)

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


# CHECK-LABEL:  hw.module @SendBundleTest(in %clk : !seq.clock, in %rst : i1, in %s1_in : !esi.channel<i32>, out b_send : !esi.bundle<[!esi.channel<i32> to "req", !esi.channel<i1> from "resp"]>, out i1_out : !esi.channel<i1>) attributes {output_file = #hw.output_file<"SendBundleTest.sv", includeReplicatedOps>} {
# CHECK-NEXT:     [[B0:%.+]] = esi.buffer %clk, %rst, %s1_in {stages = 4 : i64} : i32
# CHECK-NEXT:     %bundle, %resp = esi.bundle.pack [[B0]] : !esi.bundle<[!esi.channel<i32> to "req", !esi.channel<i1> from "resp"]>
# CHECK-NEXT:     hw.output %bundle, %resp : !esi.bundle<[!esi.channel<i32> to "req", !esi.channel<i1> from "resp"]>, !esi.channel<i1>
@unittestmodule()
class SendBundleTest(Module):
  clk = Clock()
  rst = Reset()
  b_send = SendBundle(Bundle1)
  s1_in = InputChannel(types.i32)
  i1_out = OutputChannel(types.i1)

  @generator
  def build(self):
    s1_buffered = self.s1_in.buffer(self.clk, self.rst, 4)
    self.b_send, from_chans = Bundle1.pack(req=s1_buffered)
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


# CHECK-LABEL:  hw.module @ChannelTransform(in %s1_in : !esi.channel<i32>, out s2_out : !esi.channel<i8>)
# CHECK-NEXT:     %valid, %ready, %data = esi.snoop.vr %s1_in : !esi.channel<i32>
# CHECK-NEXT:     %rawOutput, %valid_0 = esi.unwrap.vr %s1_in, %ready_1 : i32
# CHECK-NEXT:     [[R0:%.+]] = comb.extract %rawOutput from 0 : (i32) -> i8
# CHECK-NEXT:     %chanOutput, %ready_1 = esi.wrap.vr [[R0]], %valid_0 : i8
# CHECK-NEXT:     hw.output %chanOutput : !esi.channel<i8>
@unittestmodule()
class ChannelTransform(Module):
  s1_in = InputChannel(Bits(32))
  s2_out = OutputChannel(Bits(8))

  @generator
  def build(self):
    valid, ready, data = self.s1_in.snoop()
    self.s2_out = self.s1_in.transform(lambda x: x[0:8])


# CHECK-LABEL:  hw.module @CoerceBundle(in %b_in : !esi.bundle<[!esi.channel<i8> from "resp", !esi.channel<i32> to "req"]>, out b_out : !esi.bundle<[!esi.channel<i8> from "result", !esi.channel<i32> to "arg"]>)
# CHECK-NEXT:     %req = esi.bundle.unpack %result from %b_in : !esi.bundle<[!esi.channel<i8> from "resp", !esi.channel<i32> to "req"]>
# CHECK-NEXT:     %bundle, %result = esi.bundle.pack %req : !esi.bundle<[!esi.channel<i8> from "result", !esi.channel<i32> to "arg"]>
# CHECK-NEXT:     hw.output %bundle : !esi.bundle<[!esi.channel<i8> from "result", !esi.channel<i32> to "arg"]>
@unittestmodule()
class CoerceBundle(Module):
  b_in = Input(
      Bundle([
          BundledChannel("resp", ChannelDirection.FROM, Channel(Bits(8))),
          BundledChannel("req", ChannelDirection.TO, Channel(Bits(32))),
      ]))
  b_out = Output(
      Bundle([
          BundledChannel("result", ChannelDirection.FROM, Channel(Bits(8))),
          BundledChannel("arg", ChannelDirection.TO, Channel(Bits(32))),
      ]))

  @generator
  def build(ports):
    ports.b_out = ports.b_in.coerce(CoerceBundle.b_out.type)


# CHECK-LABEL:  hw.module @CoerceBundleTransform(in %b_in : !esi.bundle<[!esi.channel<i32> to "req", !esi.channel<i8> from "resp"]>, out b_out : !esi.bundle<[!esi.channel<i24> to "arg", !esi.channel<i16> from "result"]>)
# CHECK-NEXT:     %rawOutput, %valid = esi.unwrap.vr %result, %ready : i16
# CHECK-NEXT:     [[R0:%.+]] = comb.extract %rawOutput from 0 : (i16) -> i8
# CHECK-NEXT:     %chanOutput, %ready = esi.wrap.vr [[R0]], %valid : i8
# CHECK-NEXT:     %req = esi.bundle.unpack %chanOutput from %b_in : !esi.bundle<[!esi.channel<i32> to "req", !esi.channel<i8> from "resp"]>
# CHECK-NEXT:     %rawOutput_0, %valid_1 = esi.unwrap.vr %req, %ready_3 : i32
# CHECK-NEXT:     [[R1:%.+]] = comb.extract %rawOutput_0 from 0 : (i32) -> i24
# CHECK-NEXT:     %chanOutput_2, %ready_3 = esi.wrap.vr [[R1]], %valid_1 : i24
# CHECK-NEXT:     %bundle, %result = esi.bundle.pack %chanOutput_2 : !esi.bundle<[!esi.channel<i24> to "arg", !esi.channel<i16> from "result"]>
# CHECK-NEXT:     hw.output %bundle : !esi.bundle<[!esi.channel<i24> to "arg", !esi.channel<i16> from "result"]>
@unittestmodule()
class CoerceBundleTransform(Module):
  b_in = Input(
      Bundle([
          BundledChannel("req", ChannelDirection.TO, Channel(Bits(32))),
          BundledChannel("resp", ChannelDirection.FROM, Channel(Bits(8))),
      ]))
  b_out = Output(
      Bundle([
          BundledChannel("arg", ChannelDirection.TO, Channel(Bits(24))),
          BundledChannel("result", ChannelDirection.FROM, Channel(Bits(16))),
      ]))

  @generator
  def build(ports):
    ports.b_out = ports.b_in.coerce(CoerceBundleTransform.b_out.type,
                                    lambda x: x[0:24], lambda x: x[0:8])


# CHECK-LABEL:  hw.module @MMIOReq()
# CHECK-NEXT:     %c0_i64 = hw.constant 0 : i64
# CHECK-NEXT:     %false = hw.constant false
# CHECK-NEXT:     [[B:%.+]] = esi.service.req <@MMIO::@read>(#esi.appid<"mmio_req">) : !esi.bundle<[!esi.channel<ui32> to "offset", !esi.channel<i64> from "data"]>
# CHECK-NEXT:     %chanOutput, %ready = esi.wrap.vr %c0_i64, %false : i64
# CHECK-NEXT:     %offset = esi.bundle.unpack %chanOutput from [[B]] : !esi.bundle<[!esi.channel<ui32> to "offset", !esi.channel<i64> from "data"]>
@unittestmodule(esi_sys=True)
class MMIOReq(Module):

  @generator
  def build(ports):
    c64 = Bits(64)(0)
    c1 = Bits(1)(0)

    read_bundle = MMIO.read(AppID("mmio_req"))

    data, _ = Channel(Bits(64)).wrap(c64, c1)
    _ = read_bundle.unpack(data=data)


# CHECK-LABEL:  hw.module @HostMemReq()
# CHECK-NEXT:     [[R0:%.+]] = hwarith.constant 0 : ui64
# CHECK-NEXT:     %false = hw.constant false
# CHECK-NEXT:     [[R2:%.+]] = hwarith.constant 0 : ui8
# CHECK-NEXT:     [[R3:%.+]] = hw.struct_create ([[R0]], [[R2]]) : !hw.struct<address: ui64, tag: ui8>
# CHECK-NEXT:     %chanOutput, %ready = esi.wrap.vr [[R3]], %false : !hw.struct<address: ui64, tag: ui8>
# CHECK-NEXT:     [[R1:%.+]] = esi.service.req <@_HostMem::@read>(#esi.appid<"host_mem_read_req">) : !esi.bundle<[!esi.channel<!hw.struct<address: ui64, tag: ui8>> from "req", !esi.channel<!hw.struct<tag: ui8, data: ui256>> to "resp"]>
# CHECK-NEXT:     %resp = esi.bundle.unpack %chanOutput from [[R1]] : !esi.bundle<[!esi.channel<!hw.struct<address: ui64, tag: ui8>> from "req", !esi.channel<!hw.struct<tag: ui8, data: ui256>> to "resp"]>
# CHECK-NEXT:     [[R4:%.+]] = hwarith.constant 0 : ui8
# CHECK-NEXT:     [[R5:%.+]] = hwarith.constant 0 : ui256
# CHECK-NEXT:     [[R6:%.+]] = hw.struct_create ([[R0]], [[R4]], [[R5]]) : !hw.struct<address: ui64, tag: ui8, data: ui256>
# CHECK-NEXT:     %chanOutput_0, %ready_1 = esi.wrap.vr [[R6]], %false : !hw.struct<address: ui64, tag: ui8, data: ui256>
# CHECK-NEXT:     [[R7:%.+]] = esi.service.req <@_HostMem::@write>(#esi.appid<"host_mem_write_req">) : !esi.bundle<[!esi.channel<!hw.struct<address: ui64, tag: ui8, data: ui256>> from "req", !esi.channel<ui8> to "ackTag"]>
# CHECK-NEXT:     %ackTag = esi.bundle.unpack %chanOutput_0 from [[R7]] : !esi.bundle<[!esi.channel<!hw.struct<address: ui64, tag: ui8, data: ui256>> from "req", !esi.channel<ui8> to "ackTag"]>
# CHECK:        esi.service.std.hostmem @_HostMem
@unittestmodule(esi_sys=True)
class HostMemReq(Module):

  @generator
  def build(ports):
    u64 = UInt(64)(0)
    c1 = Bits(1)(0)

    read_address, _ = Channel(esi.HostMem.ReadReqType).wrap(
        esi.HostMem.ReadReqType({
            "tag": 0,
            "address": u64
        }), c1)

    _ = HostMem.read(appid=AppID("host_mem_read_req"),
                     req=read_address,
                     data_type=UInt(256))

    write_req, _ = esi.HostMem.wrap_write_req(tag=UInt(8)(0),
                                              data=UInt(256)(0),
                                              address=u64,
                                              valid=c1)
    _ = HostMem.write(appid=AppID("host_mem_write_req"), req=write_req)
