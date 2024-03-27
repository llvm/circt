// RUN: circt-opt %s --esi-connect-services -split-input-file -verify-diagnostics

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

esi.service.decl @HostComms {
  esi.service.port @Send : !esi.bundle<[!esi.channel<i16> from "send"]>
}

hw.module @Loopback (in %clk: i1) {
  // expected-error @+1 {{Request channel type does not match service port bundle channel type}}
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
  // expected-error @+1 {{'esi.service.instance' op failed to generate server}}
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

!TypeA = !hw.struct<bar: i6>
// expected-error @+1 {{cannot specify num items on non-array field "bar"}}
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

!TypeA = !hw.struct<foo : i3, bar: !hw.array<5xi2>>
// expected-error @+1 {{array with size specified must be in their own frame (in "bar")}}
!TypeAwin1 = !esi.window<
  "TypeAwin1", !TypeA, [
    <"FrameA", [
      <"foo">,
      <"bar", 5>
    ]>
  ]>

hw.module.extern @TypeAModuleDst(in %windowed: !TypeAwin1)
