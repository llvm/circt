// RUN: circt-opt %s --arc-lower-vectorizations=mode=full -split-input-file | FileCheck %s

hw.module @vectorize_body_already_lowered(%in0: i1, %in1: i1, %in2: i1, %in3: i1) -> (out0: i1, out1: i1, out2: i1, out3: i1) {
  %0:2 = arc.vectorize (%in0, %in1), (%in2, %in2) : (i1, i1, i1, i1) -> (i1, i1) {
  ^bb0(%arg0: i2, %arg1: i2):
    %1 = arith.andi %arg0, %arg1 : i2
    arc.vectorize.return %1 : i2
  }

  %1:2 = arc.vectorize (%in0, %in1), (%in2, %in3) : (i1, i1, i1, i1) -> (i1, i1) {
  ^bb0(%arg0: vector<2xi1>, %arg1: vector<2xi1>):
    %1 = arith.andi %arg0, %arg1 : vector<2xi1>
    arc.vectorize.return %1 : vector<2xi1>
  }

  hw.output %0#0, %0#1, %1#0, %1#1 : i1, i1, i1, i1
}

// CHECK-LABEL: hw.module @vectorize_body_already_lowered
//  CHECK-SAME: ([[IN0:%.+]]: i1, [[IN1:%.+]]: i1, [[IN2:%.+]]: i1, [[IN3:%.+]]: i1) ->
//       CHECK: [[V0:%.+]] = comb.concat [[IN0]], [[IN1]] :
//       CHECK: [[V1:%.+]] = comb.concat [[IN2]], [[IN2]] :
//       CHECK: [[V2:%.+]] = arith.andi [[V0]], [[V1]]
//       CHECK: [[V3:%.+]] = comb.extract [[V2]] from 0
//       CHECK: [[V4:%.+]] = comb.extract [[V2]] from 1
//       CHECK: [[CST:%.+]] = arith.constant dense<false>
//       CHECK: [[V5:%.+]] = vector.insert [[IN0]], [[CST]] [0]
//       CHECK: [[V6:%.+]] = vector.insert [[IN1]], [[V5]] [1]
//       CHECK: [[CST:%.+]] = arith.constant dense<false>
//       CHECK: [[V7:%.+]] = vector.insert [[IN2]], [[CST]] [0]
//       CHECK: [[V8:%.+]] = vector.insert [[IN3]], [[V7]] [1]
//       CHECK: [[V9:%.+]] = arith.andi [[V6]], [[V8]]
//       CHECK: [[V10:%.+]] = vector.extract [[V9]][0]
//       CHECK: [[V11:%.+]] = vector.extract [[V9]][1]
//       CHECK: hw.output [[V3]], [[V4]], [[V10]], [[V11]]
