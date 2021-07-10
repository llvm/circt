// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s -verify-diagnostics --lower-seq-to-sv | circt-opt -verify-diagnostics | FileCheck %s --check-prefix=SV
hw.module @top(%clk: i1, %rst: i1, %i: i32, %s: !hw.struct<foo: i32>) {
  %rv = hw.constant 0 : i32

  %r0 = seq.compreg %i, %clk, %rst, %rv : i32
  seq.compreg %i, %clk : i32
  // CHECK: %{{.+}} = seq.compreg %i, %clk, %rst, %c0_i32  : i32
  // CHECK: %{{.+}} = seq.compreg %i, %clk : i32
  // SV: [[REG0:%.+]] = sv.reg  : !hw.inout<i32>
  // SV: [[REG5:%.+]] = sv.read_inout [[REG0]] : !hw.inout<i32>
  // SV: sv.alwaysff(posedge %clk)  {
  // SV:   sv.passign [[REG0]], %i : i32
  // SV: }(syncreset : posedge %rst)  {
  // SV:   sv.passign [[REG0]], %c0_i32 : i32
  // SV: }
  // SV: [[REG1:%.+]] = sv.reg  : !hw.inout<i32>
  // SV: sv.alwaysff(posedge %clk)  {
  // SV:   sv.passign [[REG1]], %i : i32
  // SV: }

  %sv = hw.struct_create (%r0) : !hw.struct<foo: i32>

  seq.compreg %s, %clk, %rst, %sv : !hw.struct<foo: i32>
  seq.compreg %s, %clk : !hw.struct<foo: i32>
  // CHECK: %{{.+}} = seq.compreg %s, %clk, %rst, %{{.+}} : !hw.struct<foo: i32>
  // CHECK: %{{.+}} = seq.compreg %s, %clk : !hw.struct<foo: i32>

  // SV: [[REGST:%.+]] = hw.struct_create ([[REG5]]) : !hw.struct<foo: i32>
  // SV: [[REG3:%.+]] = sv.reg  : !hw.inout<struct<foo: i32>>
  // SV: sv.alwaysff(posedge %clk)  {
  // SV:   sv.passign [[REG3]], %s : !hw.struct<foo: i32>
  // SV: }(syncreset : posedge %rst)  {
  // SV:   sv.passign [[REG3]], [[REGST]] : !hw.struct<foo: i32>
  // SV: }
  // SV: [[REG4:%.+]] = sv.reg  : !hw.inout<struct<foo: i32>>
  // SV: sv.alwaysff(posedge %clk)  {
  // SV:   sv.passign [[REG4]], %s : !hw.struct<foo: i32>
  // SV: }
}

hw.module @testRegWithSyncClear(%clk: i1, %d: !hw.struct<foo:i32>, %reset: i1, %resetValue: !hw.struct<foo:i32>) -> (%o : !hw.struct<foo:i32>) {
  %gnd = hw.constant 0 : i1
  %vdd = hw.constant 1 : i1
  %r0  = seq.reg %d, posedge %clk, enable %vdd, syncreset posedge %reset, %resetValue  : !hw.struct<foo:i32>
  hw.output %r0 : !hw.struct<foo:i32>
}
