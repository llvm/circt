// RUN: circt-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: @FirReg
hw.module @FirReg(%clk: i1) -> (out : i32) {
  // CHECK: %c0_i32 = hw.constant 0 : i32
  // CHECK: hw.output %c0_i32 : i32
  %reg = seq.firreg %reg clock %clk : i32
  hw.output %reg : i32
}

// CHECK-LABEL: @FirRegReset
hw.module @FirRegReset(%clk: i1, %r : i1, %v : i32) -> (out : i32) {
  // CHECK: hw.output %v : i32
  %reg = seq.firreg %reg clock %clk reset sync %r, %v : i32
  hw.output %reg : i32
}

// This should not optimize anything until we have constant aggregate attribute
// support.
// CHECK-LABEL: @FirRegAggregate
hw.module @FirRegAggregate(%clk: i1) -> (out : !hw.struct<foo: i32>) {
  // CHECK: %reg = seq.firreg %reg clock %clk : !hw.struct<foo: i32>
  // CHECK: hw.output %reg : !hw.struct<foo: i32>
  %reg = seq.firreg %reg clock %clk : !hw.struct<foo: i32>
  hw.output %reg : !hw.struct<foo: i32>
}