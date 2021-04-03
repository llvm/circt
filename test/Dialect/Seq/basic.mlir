// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s -verify-diagnostics --lower-seq-to-sv | circt-opt -verify-diagnostics | FileCheck %s
rtl.module @top(%clk: i1, %rst: i1, %i: i32, %s: !rtl.struct<foo: i23>) {
  %rv = rtl.constant 0 : i32

  seq.compreg %i, %clk, %rst, %rv : i32
  seq.compreg %i, %clk : i32
  // CHECK: %{{.+}} = seq.compreg %i, %clk, %rst, %c0_i32  : i32
  // CHECK: %{{.+}} = seq.compreg %i, %clk : i32

  %c0_i23 = rtl.constant 0 : i23
  %sv = rtl.struct_create (%c0_i23) : !rtl.struct<foo: i23>

  seq.compreg %s, %clk, %rst, %sv : !rtl.struct<foo: i23>
  seq.compreg %s, %clk : !rtl.struct<foo: i23>
  // CHECK: %{{.+}} = seq.compreg %s, %clk, %rst, %{{.+}} : !rtl.struct<foo: i23>
  // CHECK: %{{.+}} = seq.compreg %s, %clk : !rtl.struct<foo: i23>
}
