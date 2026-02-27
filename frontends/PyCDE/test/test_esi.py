# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1 | FileCheck %s

from pycde import (Clock, Input, InputChannel, Output, OutputChannel, Module,
                   Reset, generator)
from pycde import esi
from pycde.common import AppID, Constant, RecvBundle, SendBundle
from pycde.constructs import Wire
from pycde.esi import HostMem, MMIO
from pycde.module import Metadata
from pycde.support import _obj_to_attribute, optional_dict_to_dict_attr
from pycde.types import (Bit, Bits, Bundle, BundledChannel, Channel,
                         ChannelDirection, ChannelSignaling, ClockType, List,
                         StructType, UInt, Window)
from pycde.testing import unittestmodule

BIT = Bit
I16 = Bits(16)
I24 = Bits(24)
I32 = Bits(32)

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
  rst = Input(BIT)

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

    loopback = Wire(Channel(I16))
    call_bundle = HostComms.req_resp(AppID("loopback_inout", 0))
    froms = call_bundle.unpack(resp=loopback)
    from_host = froms['req']

    ready = Wire(BIT)
    wide_data, valid = from_host.unwrap(ready)
    data = wide_data[0:16]
    data_chan, data_ready = loopback.type.wrap(data, valid)
    ready.assign(data_ready)
    loopback.assign(data_chan)


CallBundle = Bundle([
    BundledChannel("renamed_result", ChannelDirection.FROM, Bits(16)),
    BundledChannel("args", ChannelDirection.TO, Bits(24))
])


# CHECK-LABEL: hw.module @LoopbackCoercedCall(in %clk : !seq.clock, in %rst : i1, out call : !esi.bundle<[!esi.channel<i16> from "renamed_result", !esi.channel<i24> to "args"]>)
# CHECK:         [[REQ:%.+]] = esi.service.req <@_FuncService::@call>(#esi.appid<"loopback_coerced">) : !esi.bundle<[!esi.channel<i24> to "arg", !esi.channel<i16> from "result"]>
# CHECK:         %arg = esi.bundle.unpack %renamed_result from [[REQ]] : !esi.bundle<[!esi.channel<i24> to "arg", !esi.channel<i16> from "result"]>
# CHECK:         %bundle, %renamed_result = esi.bundle.pack %arg : !esi.bundle<[!esi.channel<i16> from "renamed_result", !esi.channel<i24> to "args"]>
@unittestmodule()
class LoopbackCoercedCall(Module):
  clk = Clock()
  rst = Reset()
  call = Output(CallBundle)

  @generator
  def construct(ports):
    ports.call = esi.FuncService.get(name=AppID("loopback_coerced"),
                                     bundle_type=CallBundle)


# CHECK-LABEL:  hw.module @LoopbackCall(in %clk : !seq.clock, in %rst : i1) attributes {output_file = #hw.output_file<"LoopbackCall.sv", includeReplicatedOps>} {
# CHECK-NEXT:     [[R2:%.+]] = esi.buffer %clk, %rst, %chanOutput {stages = 1 : i64} : !esi.channel<i16> -> !esi.channel<i16, FIFO>
# CHECK-NEXT:     [[R0:%.+]] = esi.service.req <@_FuncService::@call>(#esi.appid<"loopback">) : !esi.bundle<[!esi.channel<i24> to "arg", !esi.channel<i16, FIFO> from "result"]>
# CHECK-NEXT:     %arg = esi.bundle.unpack [[R2]] from [[R0]] : !esi.bundle<[!esi.channel<i24> to "arg", !esi.channel<i16, FIFO> from "result"]>
# CHECK-NEXT:     %rawOutput, %valid = esi.unwrap.vr %arg, %ready : i24
# CHECK-NEXT:     [[R1:%.+]] = comb.extract %rawOutput from 0 : (i24) -> i16
# CHECK-NEXT:     %chanOutput, %ready = esi.wrap.vr [[R1]], %valid : i16
# CHECK-NEXT:     hw.output
# CHECK-NEXT:   }
# CHECK-NEXT:   esi.service.std.func @_FuncService
@unittestmodule(print=True)
class LoopbackCall(Module):
  clk = Clock()
  rst = Input(BIT)

  metadata = Metadata(
      name="LoopbackCall",
      version="0.1",
  )

  @generator
  def construct(self):
    loopback_src = Wire(Channel(I16))
    loopback = loopback_src.buffer(self.clk, self.rst, 1, ChannelSignaling.FIFO)
    args = esi.FuncService.get_call_chans(name=AppID("loopback"),
                                          arg_type=Bits(24),
                                          result=loopback)

    ready = Wire(BIT)
    wide_data, valid = args.unwrap(ready)
    data = wide_data[0:16]
    data_chan, data_ready = loopback_src.type.wrap(data, valid)
    ready.assign(data_ready)
    loopback_src.assign(data_chan)


class Producer(Module):
  clk = Clock()
  int_out = OutputChannel(I32)

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


ExStruct = StructType({
    'a': Bits(4),
    'b': UInt(32),
})

Bundle1 = Bundle([
    BundledChannel("req", ChannelDirection.TO, Channel(I32)),
    BundledChannel("resp", ChannelDirection.FROM, Channel(BIT)),
])
# CHECK: Bundle<[('req', ChannelDirection.TO, Channel<Bits<32>, ValidReady>), ('resp', ChannelDirection.FROM, Channel<Bits<1>, ValidReady>)]>
print(Bundle1)
# CHECK: Channel<Bits<1>, ValidReady>
print(Bundle1.resp)


# CHECK-LABEL:  hw.module @SendBundleTest(in %clk : !seq.clock, in %rst : i1, in %s1_in : !esi.channel<i32>, out b_send : !esi.bundle<[!esi.channel<i32> to "req", !esi.channel<i1> from "resp"]>, out i1_out : !esi.channel<i1>) attributes {output_file = #hw.output_file<"SendBundleTest.sv", includeReplicatedOps>} {
# CHECK-NEXT:     [[B0:%.+]] = esi.buffer %clk, %rst, %s1_in {stages = 4 : i64} : !esi.channel<i32> -> !esi.channel<i32>
# CHECK-NEXT:     %bundle, %resp = esi.bundle.pack [[B0]] : !esi.bundle<[!esi.channel<i32> to "req", !esi.channel<i1> from "resp"]>
# CHECK-NEXT:     hw.output %bundle, %resp : !esi.bundle<[!esi.channel<i32> to "req", !esi.channel<i1> from "resp"]>, !esi.channel<i1>
@unittestmodule()
class SendBundleTest(Module):
  clk = Clock()
  rst = Reset()
  b_send = SendBundle(Bundle1)
  s1_in = InputChannel(I32)
  i1_out = OutputChannel(BIT)

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
  s1_out = OutputChannel(I32)
  i1_in = InputChannel(BIT)

  @generator
  def build(self):
    to_channels = self.b_recv.unpack(resp=self.i1_in)
    self.s1_out = to_channels['req']


# CHECK-LABEL:  hw.module @ListTest(in %lst_in : !esi.window<"default_window", !hw.struct<data: !esi.list<i8>>, [<"", [<"data">]>]>, out lst_out : !esi.window<"default_window", !hw.struct<data: !esi.list<i8>>, [<"", [<"data">]>]>)
# CHECK-NEXT:     hw.output %lst_in : !esi.window<"default_window", !hw.struct<data: !esi.list<i8>>, [<"", [<"data">]>]>
@unittestmodule()
class ListTest(Module):
  list_window = Window.default_of(List(Bits(8)))
  lst_in = Input(list_window)
  lst_out = Output(list_window)

  @generator
  def build(self):
    self.lst_out = self.lst_in


# CHECK-LABEL:  hw.module @ChannelTransform(in %clk : !seq.clock, in %s1_in : !esi.channel<i32>, out s2_out : !esi.channel<i8>)
# CHECK-NEXT:     %valid, %ready, %data = esi.snoop.vr %s1_in : !esi.channel<i32>
# CHECK-NEXT:     %transaction, [[SNOOP_DATA:%.+]] = esi.snoop.xact %s1_in : !esi.channel<i32>
# CHECK-NEXT:     [[CLK_I1:%.+]] = seq.from_clock %clk
# CHECK-NEXT:     sv.alwaysff(posedge [[CLK_I1]]) {
# CHECK-NEXT:       sv.if %transaction {
# CHECK-NEXT:         sv.info.procedural "Pre-transform: %p"([[SNOOP_DATA]]) : i32
# CHECK-NEXT:       }
# CHECK-NEXT:     }
# CHECK-NEXT:     %rawOutput, [[VALID2:%.+]] = esi.unwrap.vr %s1_in, [[READY2:%.+]] : i32
# CHECK-NEXT:     [[R0:%.+]] = comb.extract %rawOutput from 0 : (i32) -> i8
# CHECK-NEXT:     %chanOutput, [[READY2]] = esi.wrap.vr [[R0]], [[VALID2]] : i8
# CHECK-NEXT:     hw.output %chanOutput : !esi.channel<i8>
@unittestmodule()
class ChannelTransform(Module):
  clk = Clock()
  s1_in = InputChannel(Bits(32))
  s2_out = OutputChannel(Bits(8))

  @generator
  def build(self):
    from pycde.testing import print_info
    valid, ready, data = self.s1_in.snoop()
    xact, snooped_data = self.s1_in.snoop_xact()
    xact.when_true(lambda: print_info("Pre-transform: %p", snooped_data))
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


def Writer(type):

  class Writer(Module):
    clk = Clock()
    rst = Reset()
    cmd = Input(type)

  return Writer


# CHECK:  hw.module @Ram1(in %clk : !seq.clock, in %rst : i1)
# CHECK:    esi.service.instance #esi.appid<"ram"> svc @ram impl as "sv"(%clk, %rst) : (!seq.clock, i1) -> ()
# CHECK:    [[WR:%.+]] = esi.service.req <@ram::@write>(#esi.appid<"ram_writer"[0]>) : !esi.bundle<[!esi.channel<!hw.struct<address: ui3, data: i32>> from "req", !esi.channel<i0> to "ack"]>
# CHECK:    %rawOutput, %valid = esi.unwrap.vr %req, %ready : !hw.struct<address: ui3, data: ui32>
# CHECK:    [[CASTED:%.+]] = hw.bitcast %rawOutput : (!hw.struct<address: ui3, data: ui32>) -> !hw.struct<address: ui3, data: i32>
# CHECK:    %chanOutput, %ready = esi.wrap.vr [[CASTED]], %valid : !hw.struct<address: ui3, data: i32>
# CHECK:    %ack = esi.bundle.unpack %chanOutput from [[WR]] : !esi.bundle<[!esi.channel<!hw.struct<address: ui3, data: i32>> from "req", !esi.channel<i0> to "ack"]>
# CHECK:    %bundle, %req = esi.bundle.pack %ack : !esi.bundle<[!esi.channel<!hw.struct<address: ui3, data: ui32>> from "req", !esi.channel<i0> to "ack"]>
# CHECK:    hw.instance "Writer" sym @Writer @Writer(clk: %clk: !seq.clock, rst: %rst: i1, cmd: %bundle: !esi.bundle<[!esi.channel<!hw.struct<address: ui3, data: ui32>> from "req", !esi.channel<i0> to "ack"]>) -> ()


@unittestmodule()
class Ram1(Module):
  clk = Clock()
  rst = Reset()

  @generator
  def build(ports):
    ramsvc = esi.DeclareRandomAccessMemory(Bits(32), 8, "ram")
    ramsvc.implement_as("sv", ports.clk, ports.rst)

    mem_write = ramsvc.get_write(UInt(32))
    Writer(mem_write.type)(clk=ports.clk, rst=ports.rst, cmd=mem_write)


# CHECK-LABEL: hw.module @UTurn(in %clk : !seq.clock, in %rst : i1, out out1 : !esi.bundle<[!esi.channel<i32> to "req", !esi.channel<i1> from "resp"]>, out out2 : !esi.bundle<[!esi.channel<i32> from "req", !esi.channel<i1> to "resp"]>)
# CHECK-NEXT:      %bundle, %resp = esi.bundle.pack %req : !esi.bundle<[!esi.channel<i32> to "req", !esi.channel<i1> from "resp"]>
# CHECK-NEXT:      %bundle_0, %req = esi.bundle.pack %resp : !esi.bundle<[!esi.channel<i32> from "req", !esi.channel<i1> to "resp"]>
# CHECK-NEXT:      hw.output %bundle, %bundle_0 : !esi.bundle<[!esi.channel<i32> to "req", !esi.channel<i1> from "resp"]>, !esi.bundle<[!esi.channel<i32> from "req", !esi.channel<i1> to "resp"]>
@unittestmodule()
class UTurn(Module):
  clk = Clock()
  rst = Reset()

  out1 = Output(Bundle1)
  out2 = Output(Bundle1.inverted())

  @generator
  def build(ports):
    ports.out1, ports.out2 = Bundle1.create_uturn()


# Define test bundles with both TO and FROM channels
TestBundleInput = Bundle([
    BundledChannel("req", ChannelDirection.TO, Channel(Bits(8))),
    BundledChannel("resp", ChannelDirection.FROM, Channel(Bits(24)))
])
TestBundleOutput = Bundle([
    BundledChannel("req", ChannelDirection.TO, Channel(Bits(16))),
    BundledChannel("resp", ChannelDirection.FROM, Channel(Bits(48)))
])


def transform_double_width(data):
  """Transform function that doubles the width by padding with zeros."""
  from pycde.dialects import hw
  from pycde.signals import BitsSignal
  # Convert to bits to enable operations
  data_bits = data.as_bits()
  zero_pad = Bits(8)(0)
  return BitsSignal.concat([zero_pad, data_bits])


def transform_truncate_half(data):
  """Transform function that truncates to half width."""
  # Convert to bits to enable slicing
  data_bits = data.as_bits()
  half_width = data_bits.type.width // 2
  return data_bits[0:half_width]


# CHECK-LABEL:   hw.module @TestBundleTransformBasic(in %clk : !seq.clock, in %rst : i1, in %bundle_in : !esi.bundle<[!esi.channel<i8> to "req", !esi.channel<i24> from "resp"]>, out bundle_out : !esi.bundle<[!esi.channel<i16> to "req", !esi.channel<i48> from "resp"]>) attributes {output_file = #hw.output_file<"TestBundleTransformBasic.sv", includeReplicatedOps>} {
# CHECK-NEXT:      %req = esi.bundle.unpack %chanOutput_2 from %bundle_in : !esi.bundle<[!esi.channel<i8> to "req", !esi.channel<i24> from "resp"]>
# CHECK-NEXT:      %rawOutput, %valid = esi.unwrap.vr %req, %ready : i8
# CHECK-NEXT:      %c0_i8 = hw.constant 0 : i8
# CHECK-NEXT:      [[R0:%.+]] = comb.concat %c0_i8, %rawOutput : i8, i8
# CHECK-NEXT:      %chanOutput, %ready = esi.wrap.vr [[R0]], %valid : i16
# CHECK-NEXT:      %bundle, %resp = esi.bundle.pack %chanOutput : !esi.bundle<[!esi.channel<i16> to "req", !esi.channel<i48> from "resp"]>
# CHECK-NEXT:      %rawOutput_0, %valid_1 = esi.unwrap.vr %resp, %ready_3 : i48
# CHECK-NEXT:      [[R1:%.+]] = comb.extract %rawOutput_0 from 0 : (i48) -> i24
# CHECK-NEXT:      %chanOutput_2, %ready_3 = esi.wrap.vr [[R1]], %valid_1 : i24
# CHECK-NEXT:      hw.output %bundle : !esi.bundle<[!esi.channel<i16> to "req", !esi.channel<i48> from "resp"]>
@unittestmodule()
class TestBundleTransformBasic(Module):
  """Test basic bundle transform functionality with bidirectional bundle."""

  clk = Clock()
  rst = Input(Bits(1))

  # Bundle input/output for testing
  bundle_in = Input(TestBundleInput)
  bundle_out = Output(TestBundleOutput)

  @generator
  def build(self):
    # Transform the bundle by doubling the width of the req channel
    # and truncating the resp channel to half width
    transformed_bundle = self.bundle_in.transform(
        req=transform_double_width, resp=(Bits(48), transform_truncate_half))

    self.bundle_out = transformed_bundle
