// RUN: circt-opt --split-input-file -lower-calyx-to-hw --canonicalize %s | FileCheck %s

// CHECK-LABEL:   hw.module @main(
// CHECK-SAME:                    %[[IN0:.*]]: i4, %[[CLK:.*]]: i1, %[[RESET:.*]]: i1, %[[GO:.*]]: i1) -> (out0: i8, done: i1) {
// CHECK:           %[[C0:.*]] = hw.constant 0 : i4
// CHECK:           %[[C1:.*]] = hw.constant true
// CHECK:           %[[PADDED:.*]] = comb.concat %[[C0]], %[[IN0]] : i4, i4
// CHECK:           hw.output %[[PADDED]], %[[C1]] : i8, i1
// CHECK:         }
calyx.program "main" {
  calyx.component @main(%in0: i4, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i8, %done: i1 {done}) {
    %true = hw.constant true
    %std_pad.in, %std_pad.out = calyx.std_pad @std_pad : i4, i8
    calyx.wires {
      calyx.assign %std_pad.in = %in0 : i4
      calyx.assign %out0 = %std_pad.out : i8
      calyx.assign %done = %true : i1
    }
    calyx.control {}
  }
}

// -----

// CHECK-LABEL:   hw.module @main(
// CHECK-SAME:                    %[[IN0:.*]]: i4, %[[CLK:.*]]: i1, %[[RESET:.*]]: i1, %[[GO:.*]]: i1) -> (out0: i8, done: i1) {
// CHECK:           %[[C1:.*]] = hw.constant true
// CHECK:           %[[SIGN:.*]] = comb.extract %[[IN0]] from 3 : (i4) -> i1
// CHECK:           %[[SIGNVEC:.*]] = comb.replicate %[[SIGN]] : (i1) -> i4
// CHECK:           %[[EXTENDED:.*]] = comb.concat %[[SIGNVEC]], %[[IN0]] : i4, i4
// CHECK:           hw.output %[[EXTENDED]], %[[C1]] : i8, i1
// CHECK:         }
calyx.program "main" {
  calyx.component @main(%in0: i4, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i8, %done: i1 {done}) {
    %true = hw.constant true
    %std_extsi.in, %std_extsi.out = calyx.std_extsi @std_extsi : i4, i8
    calyx.wires {
      calyx.assign %std_extsi.in = %in0 : i4
      calyx.assign %out0 = %std_extsi.out : i8
      calyx.assign %done = %true : i1
    }
    calyx.control {}
  }
}
