// RUN: circt-opt %s --arc-lower-vectorizations=mode=inline-body --split-input-file --verify-diagnostics | FileCheck %s

hw.module @inline_body(in %in0: i1, in %in1: i1, in %in2: i1, in %in3: i1, out out0: i1, out out1: i1, out out2: i1, out out3: i1) {
  %0 = comb.concat %in0, %in1 : i1, i1
  %1 = comb.concat %in2, %in2 : i1, i1

  %2 = arc.vectorize (%0), (%1) : (i2, i2) -> i2 {
  ^bb0(%arg0: i2, %arg1: i2):
    %12 = arith.andi %arg0, %arg1 : i2
    arc.vectorize.return %12 : i2
  }

  %3 = comb.extract %2 from 0 : (i2) -> i1
  %4 = comb.extract %2 from 1 : (i2) -> i1

  %cst = arith.constant dense<false> : vector<2xi1>
  %5 = vector.insert %in0, %cst [0] : i1 into vector<2xi1>
  %6 = vector.insert %in1, %5 [1] : i1 into vector<2xi1>
  %cst_0 = arith.constant dense<false> : vector<2xi1>
  %7 = vector.insert %in2, %cst_0 [0] : i1 into vector<2xi1>
  %8 = vector.insert %in3, %7 [1] : i1 into vector<2xi1>

  %9 = arc.vectorize (%6), (%8) : (vector<2xi1>, vector<2xi1>) -> vector<2xi1> {
  ^bb0(%arg0: vector<2xi1>, %arg1: vector<2xi1>):
    %12 = arith.andi %arg0, %arg1 : vector<2xi1>
    arc.vectorize.return %12 : vector<2xi1>
  }

  %10 = vector.extract %9[0] : i1 from vector<2xi1>
  %11 = vector.extract %9[1] : i1 from vector<2xi1>
  hw.output %3, %4, %10, %11 : i1, i1, i1, i1
}

// CHECK-LABEL: hw.module @inline_body
//       CHECK: [[V0:%.+]] = comb.concat %in0, %in1 :
//       CHECK: [[V1:%.+]] = comb.concat %in2, %in2 :
//       CHECK: [[V2:%.+]] = arith.andi [[V0]], [[V1]]
//       CHECK: [[V3:%.+]] = comb.extract [[V2]] from 0
//       CHECK: [[V4:%.+]] = comb.extract [[V2]] from 1
//       CHECK: [[CST:%.+]] = arith.constant dense<false>
//       CHECK: [[V5:%.+]] = vector.insert %in0, [[CST]] [0]
//       CHECK: [[V6:%.+]] = vector.insert %in1, [[V5]] [1]
//       CHECK: [[CST:%.+]] = arith.constant dense<false>
//       CHECK: [[V7:%.+]] = vector.insert %in2, [[CST]] [0]
//       CHECK: [[V8:%.+]] = vector.insert %in3, [[V7]] [1]
//       CHECK: [[V9:%.+]] = arith.andi [[V6]], [[V8]]
//       CHECK: [[V10:%.+]] = vector.extract [[V9]][0]
//       CHECK: [[V11:%.+]] = vector.extract [[V9]][1]
//       CHECK: hw.output [[V3]], [[V4]], [[V10]], [[V11]]

// -----

hw.module @vectorize(in %in0: i1, in %in1: i1, in %in2: i1, in %in3: i1, out out0: i1, out out1: i1) {
  // expected-error @below {{can only inline body if boundary and body are already vectorized}}
  %0:2 = arc.vectorize (%in0, %in1), (%in2, %in3) : (i1, i1, i1, i1) -> (i1, i1) {
  ^bb0(%arg0: i1, %arg1: i1):
    %1 = comb.and %arg0, %arg1 : i1
    arc.vectorize.return %1 : i1
  }
  hw.output %0#0, %0#1 : i1, i1
}
