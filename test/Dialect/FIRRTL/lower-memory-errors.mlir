// RUN: circt-opt -firrtl-lower-memory -verify-diagnostics -split-input-file %s

// LowerMemory requires that memories have been flattened and only supports
// UInt data types.
firrtl.circuit "SInt" {
firrtl.module @SInt() {
  // expected-error@below {{memories should be flattened before running LowerMemory}}
  %MRead_read = firrtl.mem Undefined {depth = 12 : i64, name = "MRead", portNames = ["read"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<42>>
}
}

// -----

// LowerMemory only supports seqmems.  All other memories should have been
// lowered by now. The following memory is a combmem with readLatency=1.
firrtl.circuit "CombMem" {
firrtl.module @CombMem() {
  // expected-error@below {{only seqmems are supported by LowerMemory}}
  %MRead_read = firrtl.mem Undefined {depth = 12 : i64, name = "MRead", portNames = ["read"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
}
}