// RUN: circt-opt %s --verify-esi-connections --esi-connect-services -split-input-file -verify-diagnostics

hw.module @testFifoTypes(in %clk: !seq.clock, in %rst: i1, in %a: !esi.channel<i32, FIFO>, in %b: !esi.channel<i16, FIFO(2)>) {
  // expected-error @+1 {{input and output types must match}}
  %fifo = esi.fifo in %a clk %clk rst %rst depth 12 : !esi.channel<i32, FIFO> -> !esi.channel<i16, FIFO(2)>
  hw.output %fifo : !esi.channel<i16, FIFO(2)>
}

// -----

esi.service.decl @HostComms {
  esi.service.port @Send : !esi.bundle<[!esi.channel<i16, ValidReady> from "send"]>
}

hw.module @Loopback (in %clk: i1) {
  // expected-error @+2 {{Request channel type does not match service port bundle channel type}}
  // expected-note @+1 {{Service port 'send' type: '!esi.channel<i16, ValidReady>'}}
  %dataIn = esi.service.req <@HostComms::@Send> (#esi.appid<"loopback_fromhw">) : !esi.bundle<[!esi.channel<i8, ValidReady> from "send"]>
}

// -----

esi.service.decl @HostComms {
  esi.service.port @Send : !esi.bundle<[!esi.channel<i16, ValidReady> from "send"]>
}

hw.module @Loopback (in %clk: i1) {
  // expected-error @+1 {{Request port bundle channel count does not match service port bundle channel count}}
  %dataIn = esi.service.req<@HostComms::@Send> (#esi.appid<"loopback_fromhw">) : !esi.bundle<[!esi.channel<i8, ValidReady> to "send", !esi.channel<i3, ValidReady> to "foo"]>
}

// -----

esi.service.decl @HostComms {
}

hw.module @Loopback (in %clk: i1) {
  // expected-error @+1 {{'esi.service.req' op Could not locate port "Recv"}}
  %dataIn = esi.service.req <@HostComms::@Recv> (#esi.appid<"loopback_tohw">) : !esi.bundle<[!esi.channel<i1, ValidReady> from "foo"]>
}

// -----

hw.module @Loopback (in %clk: i1) {
  // expected-error @+1 {{'esi.service.req' op Could not find service declaration @HostComms}}
  %dataIn = esi.service.req <@HostComms::@Recv> (#esi.appid<"loopback_tohw">) : !esi.bundle<[!esi.channel<i1, ValidReady> from "foo"]>
}

// -----

!reqResp = !esi.bundle<[!esi.channel<i16, ValidReady> from "req", !esi.channel<i8, ValidReady> to "resp"]>
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

hw.module.extern @Source(out a: !esi.channel<i1, ValidReady>)
hw.module.extern @Sink(in %a: !esi.channel<i1, ValidReady>)

hw.module @Top() {
  // expected-error @+1 {{channels must have at most one consumer}}
  %a = hw.instance "src" @Source() -> (a: !esi.channel<i1, ValidReady>)
  // expected-note @+1 {{channel used here}}
  hw.instance "sink1" @Sink(a: %a: !esi.channel<i1, ValidReady>) -> ()
  // expected-note @+1 {{channel used here}}
  hw.instance "sink2" @Sink(a: %a: !esi.channel<i1, ValidReady>) -> ()
}

// -----

!bundleType = !esi.bundle<[!esi.channel<i32, ValidReady> to addr]>

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

!bundleType = !esi.bundle<[!esi.channel<i32, ValidReady> to addr]>

hw.module.extern @Source(out a: !bundleType)
hw.module.extern @Sink(in %a: !bundleType)

hw.module @Top() {
  // expected-error @+1 {{bundles must have exactly one use}}
  %a = hw.instance "src" @Source() -> (a: !bundleType)
}

// -----

hw.module @wrap_multi_unwrap(in %a_data: i8, in %a_valid: i1, out a_ready: i1) {
  // expected-error @+1 {{'esi.wrap.vr' op channels must have at most one consumer}}
  %a_chan, %a_ready = esi.wrap.vr %a_data, %a_valid : !esi.channel<i8, ValidReady>
  %true = hw.constant true
  // expected-note @+1 {{channel used here}}
  %ap_data, %ap_valid = esi.unwrap.vr %a_chan, %true : !esi.channel<i8, ValidReady>
  // expected-note @+1 {{channel used here}}
  %ab_data, %ab_valid = esi.unwrap.vr %a_chan, %true : !esi.channel<i8, ValidReady>
  hw.output %a_ready : i1
}

// -----

hw.module @wrap_vo_multi_unwrap(in %a_data: i8, in %a_valid: i1) {
  // expected-error @+1 {{'esi.wrap.vo' op channels must have at most one consumer}}
  %a_chan = esi.wrap.vo %a_data, %a_valid : !esi.channel<i8, ValidOnly>
  // expected-note @+1 {{channel used here}}
  %ap_data, %ap_valid = esi.unwrap.vo %a_chan : !esi.channel<i8, ValidOnly>
  // expected-note @+1 {{channel used here}}
  %ab_data, %ab_valid = esi.unwrap.vo %a_chan : !esi.channel<i8, ValidOnly>
}

// -----

hw.module @wrap_vr_type_mismatch(in %data: i16, in %valid: i1) {
  // expected-error @+1 {{'esi.wrap.vr' op input and output types must match}}
  %chan, %ready = "esi.wrap.vr"(%data, %valid) : (i16, i1) -> (!esi.channel<i8, ValidReady>, i1)
}

// -----

hw.module @wrap_vo_type_mismatch(in %data: i16, in %valid: i1) {
  // expected-error @+1 {{'esi.wrap.vo' op input and output types must match}}
  %chan = "esi.wrap.vo"(%data, %valid) : (i16, i1) -> !esi.channel<i8, ValidOnly>
}

// -----

hw.module @unwrap_vr_type_mismatch(in %chan: !esi.channel<i8, ValidReady>, in %ready: i1) {
  // expected-error @+2 {{'esi.unwrap.vr' op inferred type(s) 'i8', 'i1' are incompatible with return type(s) of operation 'i16', 'i1'}}
  // expected-error @+1 {{'esi.unwrap.vr' op failed to infer returned types}}
  %data, %valid = "esi.unwrap.vr"(%chan, %ready) : (!esi.channel<i8, ValidReady>, i1) -> (i16, i1)
}

// -----

hw.module @wrap_fifo_type_mismatch(in %data: i16, in %empty: i1) {
  // expected-error @+1 {{'esi.wrap.fifo' op input and output types must match}}
  %chan, %rden = "esi.wrap.fifo"(%data, %empty) : (i16, i1) -> (!esi.channel<i8, FIFO>, i1)
}

// -----

// HostMem 'read_list' requires the request's 'length' field to be an unsigned
// integer (of any width).
esi.service.std.hostmem @hostmem
!respWin = !esi.window<"HostMemReadResp", !hw.struct<tag: ui8, data: !esi.list<i64>>, [<"", [<"tag">, <"data", 4>]>]>
!badReadListReq = !esi.bundle<[!esi.channel<!hw.struct<address: ui64, tag: ui8, length: i32>, ValidReady> from "req", !esi.channel<!respWin, ValidReady> to "resp"]>
hw.module @BadReadListLength(in %req : !esi.channel<!hw.struct<address: ui64, tag: ui8, length: i32>, ValidReady>, out resp : !esi.channel<!respWin, ValidReady>) {
  // expected-error @+1 {{'read_list' request 'length' must be an unsigned integer, got 'i32'}}
  %bundle = esi.service.req <@hostmem::@read_list> (#esi.appid<"badReadList">) : !badReadListReq
  %resp = esi.bundle.unpack %req from %bundle : !badReadListReq
  hw.output %resp : !esi.channel<!respWin, ValidReady>
}

// -----

// The signaling is required.
// expected-error @+1 {{expected ','}}
hw.module.extern @MissingSignaling(in %a: !esi.channel<i8>)

// -----

hw.module @stageFIFO(in %clk: !seq.clock, in %rst: i1, in %a: !esi.channel<i8, FIFO>) {
  // expected-error @+1 {{only supports valid-ready signaling}}
  %0 = esi.stage %clk, %rst, %a : !esi.channel<i8, FIFO>
}

// -----

hw.module @bufferDelayMismatch(in %clk: !seq.clock, in %rst: i1, in %a: !esi.channel<i8, FIFO(1)>) {
  // expected-error @+1 {{input and output data delays must match}}
  %0 = esi.buffer %clk, %rst, %a : !esi.channel<i8, FIFO(1)> -> !esi.channel<i8, ValidReady>
}

// -----

hw.module @fifoDepthInputDelay(in %clk: !seq.clock, in %rst: i1, in %a: !esi.channel<i8, ValidReady(2)>) {
  // expected-error @+1 {{depth must be greater than the input data delay}}
  %0 = esi.fifo in %a clk %clk rst %rst depth 2 : !esi.channel<i8, ValidReady(2)> -> !esi.channel<i8, ValidReady>
}
