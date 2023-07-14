// RUN: circt-opt %s --verify-diagnostics -pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-types{preserve-aggregate=none}))'

// Don't silently drop symbols from memories.

// https://github.com/llvm/circt/issues/5592
firrtl.circuit "Issue5592" {
  firrtl.module @Issue5592(in %clock: !firrtl.clock, in %rAddr: !firrtl.uint<4>, in %rEn: !firrtl.uint<1>, out %rData: !firrtl.vector<uint<8>, 4>, in %wMask: !firrtl.vector<uint<1>, 4>, in %wData: !firrtl.vector<uint<8>, 4>) attributes {convention = #firrtl<convention scalarized>} {
    // expected-error @below {{has a symbol, but no symbols may exist on aggregates passed through LowerTypes}}
    %memory_r, %memory_w = firrtl.mem sym @X interesting_name Undefined {depth = 16 : i64, name = "memory", portNames = ["r", "w"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: vector<uint<8>, 4>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: vector<uint<8>, 4>, mask: vector<uint<1>, 4>>
    %0 = firrtl.subfield %memory_w[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: vector<uint<8>, 4>, mask: vector<uint<1>, 4>>
    %1 = firrtl.subfield %memory_w[mask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: vector<uint<8>, 4>, mask: vector<uint<1>, 4>>
    %2 = firrtl.subfield %memory_w[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: vector<uint<8>, 4>, mask: vector<uint<1>, 4>>
    %3 = firrtl.subfield %memory_w[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: vector<uint<8>, 4>, mask: vector<uint<1>, 4>>
    %4 = firrtl.subfield %memory_w[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: vector<uint<8>, 4>, mask: vector<uint<1>, 4>>
    %5 = firrtl.subfield %memory_r[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: vector<uint<8>, 4>>
    %6 = firrtl.subfield %memory_r[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: vector<uint<8>, 4>>
    %7 = firrtl.subfield %memory_r[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: vector<uint<8>, 4>>
    %8 = firrtl.subfield %memory_r[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: vector<uint<8>, 4>>
    firrtl.strictconnect %8, %clock : !firrtl.clock
    firrtl.strictconnect %7, %rEn : !firrtl.uint<1>
    firrtl.strictconnect %6, %rAddr : !firrtl.uint<4>
    firrtl.strictconnect %rData, %5 : !firrtl.vector<uint<8>, 4>
    firrtl.strictconnect %4, %clock : !firrtl.clock
    firrtl.strictconnect %3, %rEn : !firrtl.uint<1>
    firrtl.strictconnect %2, %rAddr : !firrtl.uint<4>
    firrtl.strictconnect %1, %wMask : !firrtl.vector<uint<1>, 4>
    firrtl.strictconnect %0, %wData : !firrtl.vector<uint<8>, 4>
  }
  sv.verbatim "Testing {{0}}" {symbols = [#hw.innerNameRef<@Issue5592::@X>]}
}
