// RUN: circt-opt %s | circt-opt | FileCheck %s

hw.module.extern @ext_fmult(in %clk : i1 {calyx.clk}, in %left : i32 {calyx.data}, in %right : i32 {calyx.data}, in %ce : i1, out result : i32 {calyx.stable, calyx.data}) attributes {filename = "fmult.sv", verilogName = "fmult"}

// CHECK-LABEL: @lib0

// CHECK-LABEL: oplib.operator @fmult latency<4>, incDelay<5.000000e-01>, outDelay<5.000000e-01>
// CHECK-NEXT: oplib.target @target0(%[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32) -> f32
// CHECK-NEXT: %[[VAL_2:.*]] = oplib.operation "mulf" in "arith"(%[[VAL_0]], %[[VAL_1]] : f32, f32) : f32
// CHECK-NEXT: oplib.output %[[VAL_2]] : f32
// CHECK:      oplib.calyx_match(@target0 : (f32, f32) -> f32) produce
// CHECK-NEXT: %[[VAL_3:.*]], %[[VAL_4:.*]], %[[VAL_5:.*]], %[[VAL_6:.*]], %[[VAL_7:.*]] = calyx.primitive @fmult of @ext_fmult : i1, i32, i32, i1, i32
// CHECK-NEXT: oplib.yield clk(%[[VAL_3]] : i1), ce(%[[VAL_6]] : i1), ins(%[[VAL_4]], %[[VAL_5]] : i32, i32), outs(%[[VAL_7]] : i32)

// CHECK-LABEL: oplib.operator @addi latency<0>, incDelay<2.000000e-01>, outDelay<2.000000e-01>
// CHECK-NEXT: oplib.target @target0(%[[VAL_8:.*]]: i32, %[[VAL_9:.*]]: i32) -> i32
// CHECK-NEXT: %[[VAL_10:.*]] = oplib.operation "addi" in "arith"(%[[VAL_8]], %[[VAL_9]] : i32, i32) : i32
// CHECK-NEXT: oplib.output %[[VAL_10]] : i32
// CHECK:      oplib.calyx_match(@target0 : (i32, i32) -> i32) produce
// CHECK-NEXT: %[[VAL_11:.*]], %[[VAL_12:.*]], %[[VAL_13:.*]] = calyx.std_add @add : i32, i32, i32
// CHECK-NEXT: oplib.yield ins(%[[VAL_11]], %[[VAL_12]] : i32, i32), outs(%[[VAL_13]] : i32)

// CHECK-LABEL: oplib.operator @trunci latency<0>

oplib.library @lib0 {
  oplib.operator @fmult latency<4>, incDelay<0.5>, outDelay<0.5> {
    oplib.target @target0(%l: f32, %r: f32) -> f32 {
      %o = oplib.operation "mulf" in "arith"(%l, %r : f32, f32) : f32
      oplib.output %o : f32
    }
    oplib.calyx_match(@target0 : (f32, f32) -> f32) produce {
      %clk, %left, %right, %ce, %result = calyx.primitive @fmult of @ext_fmult : i1, i32, i32, i1, i32
      oplib.yield clk(%clk : i1), ce(%ce : i1), ins(%left, %right : i32, i32), outs(%result : i32)
    }
  }
  oplib.operator @addi latency<0>, incDelay<0.2>, outDelay<0.2> {
    oplib.target @target0(%l: i32, %r: i32) -> i32 {
      %o = oplib.operation "addi" in "arith"(%l, %r : i32, i32) : i32
      oplib.output %o : i32
    }
    oplib.calyx_match(@target0 : (i32, i32) -> i32) produce {
      %left, %right, %out = calyx.std_add @add : i32, i32, i32
      oplib.yield ins(%left, %right : i32, i32), outs(%out : i32)
    }
  }
  oplib.operator @trunci latency<0> {
    oplib.target @target0(%in: i5) -> i4 {
      %o = oplib.operation "trunci" in "arith"(%in : i5) : i4
      oplib.output %o : i4
    }
    oplib.calyx_match(@target0 : (i5) -> i4) produce {
      %slice.in, %slice.out = calyx.std_slice @slice : i5, i4
      oplib.yield ins(%slice.in : i5), outs(%slice.out : i4)
    }
  }
  oplib.operator @cmpi latency<0> {
    oplib.target @target0(%l: i32, %r: i32) -> i1 {
      %o = oplib.operation "cmpi" in "arith" with {predicate = 0 : i64}(%l, %r : i32, i32) : i1
      oplib.output %o : i1
    }
    oplib.calyx_match(@target0 : (i32, i32) -> i1) produce {
      %eq.left, %eq.right, %eq.out = calyx.std_eq @eq : i32, i32, i1
      oplib.yield ins(%eq.left, %eq.right : i32, i32), outs(%eq.out : i1)
    }
  }
}
