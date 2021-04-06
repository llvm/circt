// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s -verify-diagnostics --lower-seq-to-sv | circt-opt -verify-diagnostics | FileCheck %s --check-prefix=SV
rtl.module @top(%clk: i1, %rst: i1, %i: i32, %s: !rtl.struct<foo: i32>) {
  %rv = rtl.constant 0 : i32

  %r0 = seq.compreg %i, %clk, %rst, %rv : i32
  seq.compreg %i, %clk : i32
  // CHECK: %{{.+}} = seq.compreg %i, %clk, %rst, %c0_i32  : i32
  // CHECK: %{{.+}} = seq.compreg %i, %clk : i32
  // SV: [[REG0:%.+]] = sv.reg  : !rtl.inout<i32>
  // SV: [[REG5:%.+]] = sv.read_inout [[REG0]] : !rtl.inout<i32>
  // SV: sv.alwaysff(posedge %clk)  {
  // SV:   sv.passign [[REG0]], %i : i32
  // SV: }(syncreset : posedge %rst)  {
  // SV:   sv.passign [[REG0]], %c0_i32 : i32
  // SV: }
  // SV: [[REG1:%.+]] = sv.reg  : !rtl.inout<i32>
  // SV: sv.alwaysff(posedge %clk)  {
  // SV:   sv.passign [[REG1]], %i : i32
  // SV: }

  %sv = rtl.struct_create (%r0) : !rtl.struct<foo: i32>

  seq.compreg %s, %clk, %rst, %sv : !rtl.struct<foo: i32>
  seq.compreg %s, %clk : !rtl.struct<foo: i32>
  // CHECK: %{{.+}} = seq.compreg %s, %clk, %rst, %{{.+}} : !rtl.struct<foo: i32>
  // CHECK: %{{.+}} = seq.compreg %s, %clk : !rtl.struct<foo: i32>

  // SV: [[REGST:%.+]] = rtl.struct_create ([[REG5]]) : !rtl.struct<foo: i32>
  // SV: [[REG3:%.+]] = sv.reg  : !rtl.inout<struct<foo: i32>>
  // SV: sv.alwaysff(posedge %clk)  {
  // SV:   sv.passign [[REG3]], %s : !rtl.struct<foo: i32>
  // SV: }(syncreset : posedge %rst)  {
  // SV:   sv.passign [[REG3]], [[REGST]] : !rtl.struct<foo: i32>
  // SV: }
  // SV: [[REG4:%.+]] = sv.reg  : !rtl.inout<struct<foo: i32>>
  // SV: sv.alwaysff(posedge %clk)  {
  // SV:   sv.passign [[REG4]], %s : !rtl.struct<foo: i32>
  // SV: }
}
