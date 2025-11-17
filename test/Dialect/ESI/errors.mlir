// RUN: circt-opt %s --verify-esi-connections --esi-connect-services -split-input-file -verify-diagnostics

sv.interface @IData {
  sv.interface.signal @data : i32
  sv.interface.signal @valid : i1
  sv.interface.signal @stall : i1
  sv.interface.modport @Sink (output @data, output @valid, input @ready)
}

hw.module @test() {
  %ifaceOut = sv.interface.instance : !sv.interface<@IData>
  %ifaceOutSink = sv.modport.get %ifaceOut @Sink: !sv.interface<@IData> -> !sv.modport<@IData::@Sink>
  // expected-error @+1 {{Interface is not a valid ESI interface.}}
  %idataChanOut = esi.wrap.iface %ifaceOutSink: !sv.modport<@IData::@Sink> -> !esi.channel<i32>
}

// -----

sv.interface @IData {
  sv.interface.signal @data : i2
  sv.interface.signal @valid : i1
  sv.interface.signal @ready : i1
  sv.interface.modport @Sink (output @data, output @valid, input @ready)
}

hw.module @test() {
  %ifaceOut = sv.interface.instance : !sv.interface<@IData>
  %ifaceOutSink = sv.modport.get %ifaceOut @Sink: !sv.interface<@IData> -> !sv.modport<@IData::@Sink>
  // expected-error @+1 {{Operation specifies '!esi.channel<i32>' but type inside doesn't match interface data type 'i2'.}}
  %idataChanOut = esi.wrap.iface %ifaceOutSink: !sv.modport<@IData::@Sink> -> !esi.channel<i32>
}

// -----

sv.interface @IData {
  sv.interface.signal @data : i2
  sv.interface.signal @valid : i1
  sv.interface.signal @ready : i1
  sv.interface.modport @Sink (output @data, output @valid, input @ready)
}

hw.module @test(in %m : !sv.modport<@IData::@Noexist>) {
  // expected-error @+1 {{Could not find modport @IData::@Noexist in symbol table.}}
  %idataChanOut = esi.wrap.iface %m: !sv.modport<@IData::@Noexist> -> !esi.channel<i32>
}

// -----

hw.module @testFifoTypes(in %clk: !seq.clock, in %rst: i1, in %a: !esi.channel<i32, FIFO>, in %b: !esi.channel<i16, FIFO(2)>) {
  // expected-error @+1 {{input and output types must match}}
  %fifo = esi.fifo in %a clk %clk rst %rst depth 12 : !esi.channel<i32, FIFO> -> !esi.channel<i16, FIFO(2)>
  hw.output %fifo : !esi.channel<i16, FIFO(2)>
}

// -----

esi.service.decl @HostComms {
  esi.service.port @Send : !esi.bundle<[!esi.channel<i16> from "send"]>
}

hw.module @Loopback (in %clk: i1) {
  // expected-error @+2 {{Request channel type does not match service port bundle channel type}}
  // expected-note @+1 {{Service port 'send' type: '!esi.channel<i16>'}}
  %dataIn = esi.service.req <@HostComms::@Send> (#esi.appid<"loopback_fromhw">) : !esi.bundle<[!esi.channel<i8> from "send"]>
}

// -----

esi.service.decl @HostComms {
  esi.service.port @Send : !esi.bundle<[!esi.channel<i16> from "send"]>
}

hw.module @Loopback (in %clk: i1) {
  // expected-error @+1 {{Request port bundle channel count does not match service port bundle channel count}}
  %dataIn = esi.service.req<@HostComms::@Send> (#esi.appid<"loopback_fromhw">) : !esi.bundle<[!esi.channel<i8> to "send", !esi.channel<i3> to "foo"]>
}

// -----

esi.service.decl @HostComms {
}

hw.module @Loopback (in %clk: i1) {
  // expected-error @+1 {{'esi.service.req' op Could not locate port "Recv"}}
  %dataIn = esi.service.req <@HostComms::@Recv> (#esi.appid<"loopback_tohw">) : !esi.bundle<[!esi.channel<i1> from "foo"]>
}

// -----

hw.module @Loopback (in %clk: i1) {
  // expected-error @+1 {{'esi.service.req' op Could not find service declaration @HostComms}}
  %dataIn = esi.service.req <@HostComms::@Recv> (#esi.appid<"loopback_tohw">) : !esi.bundle<[!esi.channel<i1> from "foo"]>
}

// -----

!reqResp = !esi.bundle<[!esi.channel<i16> from "req", !esi.channel<i8> to "resp"]>
esi.service.decl @HostComms {
  esi.service.port @ReqResp : !reqResp
}

hw.module @Top(in %clk: i1, in %rst: i1) {
  // expected-error @+2 {{'esi.service.impl_req' op did not recognize option name "badOpt"}}
  // expected-error @+1 {{'esi.service.impl_req' op failed to generate server}}
  esi.service.instance #esi.appid<"cosim"> svc @HostComms impl as  "cosim" opts {badOpt = "wrong!"} (%clk, %rst) : (i1, i1) -> ()
}

// -----

!TypeA = !hw.struct<bar: i6>
// expected-error @+1 {{invalid field name: "header5"}}
!TypeAwin1 = !esi.window<
  "TypeAwin1", !TypeA, [
    <"FrameA", [
      <"header5">
    ]>
  ]>

hw.module.extern @TypeAModuleDst(in %windowed: !TypeAwin1)

// -----

!TypeA = !hw.struct<a1: !hw.array<4xi3>, a2: !hw.array<5xi2>>
// expected-error @+1 {{cannot have two array or list fields with num items (in "a2")}}
!TypeAwin1 = !esi.window<
  "TypeAwin1", !TypeA, [
    <"FrameA", [
      <"a1", 1>,
      <"a2", 1>
    ]>
  ]>



// -----

!TypeA = !hw.struct<bar: i6>
// expected-error @+1 {{specification of num items only allowed on array or list fields (in "bar")}}
!TypeAwin1 = !esi.window<
  "TypeAwin1", !TypeA, [
    <"FrameA", [
      <"bar", 4>
    ]>
  ]>

hw.module.extern @TypeAModuleDst(in %windowed: !TypeAwin1)

// -----

!TypeA = !hw.struct<bar: !hw.array<5xi2>>
// expected-error @+1 {{num items is larger than array size}}
!TypeAwin1 = !esi.window<
  "TypeAwin1", !TypeA, [
    <"FrameA", [
      <"bar", 8>
    ]>
  ]>

hw.module.extern @TypeAModuleDst(in %windowed: !TypeAwin1)

// -----

hw.module.extern @Source(out a: !esi.channel<i1>)
hw.module.extern @Sink(in %a: !esi.channel<i1>)

hw.module @Top() {
  // expected-error @+1 {{channels must have at most one consumer}}
  %a = hw.instance "src" @Source() -> (a: !esi.channel<i1>)
  // expected-note @+1 {{channel used here}}
  hw.instance "sink1" @Sink(a: %a: !esi.channel<i1>) -> ()
  // expected-note @+1 {{channel used here}}
  hw.instance "sink2" @Sink(a: %a: !esi.channel<i1>) -> ()
}

// -----

!bundleType = !esi.bundle<[!esi.channel<i32> to addr]>

hw.module.extern @Source(out a: !bundleType)
hw.module.extern @Sink(in %a: !bundleType)

hw.module @Top() {
  // expected-error @+1 {{bundles must have exactly one use}}
  %a = hw.instance "src" @Source() -> (a: !bundleType)
  // expected-note @+1 {{bundle used here}}
  hw.instance "sink1" @Sink(a: %a: !bundleType) -> ()
  // expected-note @+1 {{bundle used here}}
  hw.instance "sink2" @Sink(a: %a: !bundleType) -> ()
}

// -----

!bundleType = !esi.bundle<[!esi.channel<i32> to addr]>

hw.module.extern @Source(out a: !bundleType)
hw.module.extern @Sink(in %a: !bundleType)

hw.module @Top() {
  // expected-error @+1 {{bundles must have exactly one use}}
  %a = hw.instance "src" @Source() -> (a: !bundleType)
}

// -----

hw.module @wrap_multi_unwrap(in %a_data: i8, in %a_valid: i1, out a_ready: i1) {
  // expected-error @+1 {{'esi.wrap.vr' op channels must have at most one consumer}}
  %a_chan, %a_ready = esi.wrap.vr %a_data, %a_valid : i8
  %true = hw.constant true
  // expected-note @+1 {{channel used here}}
  %ap_data, %ap_valid = esi.unwrap.vr %a_chan, %true : i8
  // expected-note @+1 {{channel used here}}
  %ab_data, %ab_valid = esi.unwrap.vr %a_chan, %true : i8
  hw.output %a_ready : i1
}
