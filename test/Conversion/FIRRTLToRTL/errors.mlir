// RUN: circt-opt -lower-firrtl-to-rtl -split-input-file -verify-diagnostics %s

firrtl.circuit "Div" {

  // module MemAggregate :
  //    input clock1 : Clock
  //    input clock2 : Clock
  //
  //    mem _M : @[Decoupled.scala 209:24]
  //          data-type => { id : UInt<4>, other: SInt<8> }
  //          depth => 20
  //          read-latency => 0
  //          write-latency => 1
  //          reader => read
  //          writer => write
  //          read-under-write => undefined
  // COM: This is a memory with aggregates which is currently not
  // supported.
  firrtl.module @Div(in %clock1: !firrtl.clock, in %clock2: !firrtl.clock) {
  // expected-error @+2 {{'firrtl.mem' should have simple type and known width}}
  // expected-error @+1 {{'firrtl.mem' op should have already been lowered from a ground type to an aggregate type using the LowerTypes pass}}
    %_M_read, %_M_write = firrtl.mem Undefined {depth = 20 : i64, name = "_M", portNames = ["read", "write"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.flip<bundle<addr: uint<5>, en: uint<1>, clk: clock, data: flip<bundle<id: uint<4>, other: sint<8>>>>>, !firrtl.flip<bundle<addr: uint<5>, en: uint<1>, clk: clock, data: bundle<id: uint<4>, other: sint<8>>, mask: bundle<id: uint<1>, other: uint<1>>>>
  }

  // module MemOne :
  //   mem _M : @[Decoupled.scala 209:24]
  //         data-type => { id : UInt<4>, other: SInt<8> }
  //         depth => 1
  //         read-latency => 0
  //         write-latency => 1
  //         reader => read
  //         writer => write
  //         read-under-write => undefined
  // COM: This is an aggregate memory which is not supported.
  firrtl.module @MemOne() {
  // expected-error @+2 {{'firrtl.mem' should have simple type and known width}}
  // expected-error @+1 {{'firrtl.mem' op should have already been lowered from a ground type to an aggregate type using the LowerTypes pass}}
    %_M_read, %_M_write = firrtl.mem Undefined {depth = 1 : i64, name = "_M", portNames=["read", "write"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: flip<bundle<id: uint<4>, other: sint<8>>>>>, !firrtl.flip<bundle<addr: uint<1>, en: uint<1>, clk: clock, data: bundle<id: uint<4>, other: sint<8>>, mask: bundle<id: uint<1>, other: uint<1>>>>
  }

  // COM: Unknown widths are unsupported
  // expected-error @+1 {{cannot lower this port type to RTL}}
  firrtl.module @UnknownWidth(in %a: !firrtl.uint) {}

  // https://github.com/llvm/circt/issues/778
  firrtl.module @zero_width_mem(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %r0en: !firrtl.uint<1>) {
    %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c0_ui25 = firrtl.constant 0 : !firrtl.uint<25>
  // expected-error @+1 {{'firrtl.mem' should have simple type and known width}}
    %tmp41_r0, %tmp41_w0 = firrtl.mem Undefined {depth = 10 : i64, name = "tmp41", portNames = ["r0", "w0"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: flip<uint<0>>>>, !firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<0>, mask: uint<1>>>
    %0 = firrtl.subfield %tmp41_r0("clk") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: flip<uint<0>>>>) -> !firrtl.clock
    firrtl.connect %0, %clock : !firrtl.clock, !firrtl.clock
    %1 = firrtl.subfield %tmp41_r0("en") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: flip<uint<0>>>>) -> !firrtl.uint<1>
    firrtl.connect %1, %r0en : !firrtl.uint<1>, !firrtl.uint<1>
    %2 = firrtl.subfield %tmp41_r0("addr") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: flip<uint<0>>>>) -> !firrtl.uint<4>
    firrtl.connect %2, %c0_ui4 : !firrtl.uint<4>, !firrtl.uint<4>
    %3 = firrtl.subfield %tmp41_w0("clk") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<0>, mask: uint<1>>>) -> !firrtl.clock
    firrtl.connect %3, %clock : !firrtl.clock, !firrtl.clock
    %4 = firrtl.subfield %tmp41_w0("en") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<0>, mask: uint<1>>>) -> !firrtl.uint<1>
    firrtl.connect %4, %r0en : !firrtl.uint<1>, !firrtl.uint<1>
    %5 = firrtl.subfield %tmp41_w0("addr") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<0>, mask: uint<1>>>) -> !firrtl.uint<4>
    firrtl.connect %5, %c0_ui4 : !firrtl.uint<4>, !firrtl.uint<4>
    %6 = firrtl.subfield %tmp41_w0("mask") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<0>, mask: uint<1>>>) -> !firrtl.uint<1>
    firrtl.connect %6, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %7 = firrtl.subfield %tmp41_w0("data") : (!firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<0>, mask: uint<1>>>) -> !firrtl.uint<0>
    firrtl.partialconnect %7, %c0_ui25 : !firrtl.uint<0>, !firrtl.uint<25>
  }
}
