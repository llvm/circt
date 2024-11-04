// RUN: circt-opt %s | circt-opt | FileCheck %s

hw.module.extern @flopoco_fmult(in %clk : i1 {calyx.clk}, in %left : i32 {calyx.data}, in %right : i32 {calyx.data}, in %ce : i1, out result : i32 {calyx.stable, calyx.data}) attributes {filename = "fmult.sv", verilogName = "fmult"}

// CHECK-LABEL: @lib0
// CHECK-NEXT: oplib.operator @fmult
calyxlib.library @lib0 {
  calyxlib.operator @fmult [
    latency<4>,
    incDelay<0.5>,
    outDelay<0.5>
  ] {
    calyxlib.target @target0(%l: f32, %r: f32) -> f32 { // only need to match bitwidth not type
      // %o = oplib.operation "mulf" in "arith"(%l, %r : f32, f32) : f32
      // oplib.output %o : f32
      calyxlib.output
    }
    // oplib.calyx_match<@target0> produce {
    //   %clk, %left, %right, %ce, %result = calyx.primitive @fmult of @flopoco_fmult : i1, i32, i32, i1, i32
    //   oplib.yield clk(%clk), ce(%ce), ins(%left, %right), outs(%result) : i1, i1, i32, i32, i32
    // }
  }
}
