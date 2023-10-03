// RUN: circt-opt %s --arc-lower-vectorizations=mode=boundary -split-input-file | FileCheck %s

hw.module @vectorize(in %in0: i1, in %in1: i1, in %in2: i1, in %in3: i1, in %in4: i64, in %in5: i64, in %in6: i32, in %in7: i32, out out0: i1, out out1: i1, out out2: i64, out out3: i64) {
  %0:2 = arc.vectorize (%in0, %in1), (%in2, %in3) : (i1, i1, i1, i1) -> (i1, i1) {
  ^bb0(%arg0: i1, %arg1: i1):
    %1 = comb.and %arg0, %arg1 : i1
    arc.vectorize.return %1 : i1
  }
  %1:2 = arc.vectorize (%in4, %in5), (%in6, %in6) : (i64, i64, i32, i32) -> (i64, i64) {
  ^bb0(%arg0: i64, %arg1: i32):
    %c0_i32 = hw.constant 0 : i32
    %2 = comb.concat %c0_i32, %arg1 : i32, i32
    %3 = comb.and %arg0, %2 : i64
    arc.vectorize.return %3 : i64
  }
  hw.output %0#0, %0#1, %1#0, %1#1 : i1, i1, i64, i64
}

// CHECK-LABEL: hw.module @vectorize
//  CHECK-SAME: (in [[IN0:%.+]] : i1, in [[IN1:%.+]] : i1, in [[IN2:%.+]] : i1, in [[IN3:%.+]] : i1, in [[IN4:%.+]] : i64, in [[IN5:%.+]] : i64, in [[IN6:%.+]] : i32, in [[IN7:%.+]] : i32,
//       CHECK: [[V0:%.+]] = comb.concat [[IN0]], [[IN1]] :
//       CHECK: [[V1:%.+]] = comb.concat [[IN2]], [[IN3]] :
//       CHECK: [[V2:%.+]] = arc.vectorize ([[V0]]), ([[V1]])
//       CHECK: ^bb0({{.*}}: i1, {{.*}}: i1):
//       CHECK: arc.vectorize.return {{.*}} : i1
//       CHECK: [[V3:%.+]] = comb.extract [[V2]] from 0
//       CHECK: [[V4:%.+]] = comb.extract [[V2]] from 1
//       CHECK: [[CST:%.+]] = arith.constant dense<0>
//       CHECK: [[V5:%.+]] = vector.insert [[IN4]], [[CST]] [0]
//       CHECK: [[V6:%.+]] = vector.insert [[IN5]], [[V5]] [1]
//       CHECK: [[V7:%.+]] = vector.broadcast [[IN6]]
//       CHECK: [[V8:%.+]] = arc.vectorize ([[V6]]), ([[V7]])
//       CHECK: ^bb0({{.*}}: i64, {{.*}}: i32):
//       CHECK: arc.vectorize.return {{.*}} : i64
//       CHECK: [[V9:%.+]] = vector.extract [[V8]][0]
//       CHECK: [[V10:%.+]] = vector.extract [[V8]][1]
//       CHECK: hw.output [[V3]], [[V4]], [[V9]], [[V10]]

// -----

hw.module @vectorize_body_already_lowered(in %in0: i1, in %in1: i1, in %in2: i1, in %in3: i1, out out0: i1, out out1: i1, out out2: i1, out out3: i1) {
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
//       CHECK: [[V0:%.+]] = comb.concat %in0, %in1 :
//       CHECK: [[V1:%.+]] = comb.concat %in2, %in2 :
//       CHECK: [[V2:%.+]] = arc.vectorize ([[V0]]), ([[V1]])
//       CHECK: ^bb0({{.*}}: i2, {{.*}}: i2):
//       CHECK: arc.vectorize.return {{.*}} : i2
//       CHECK: [[V3:%.+]] = comb.extract [[V2]] from 0
//       CHECK: [[V4:%.+]] = comb.extract [[V2]] from 1
//       CHECK: [[CST:%.+]] = arith.constant dense<false>
//       CHECK: [[V5:%.+]] = vector.insert %in0, [[CST]] [0]
//       CHECK: [[V6:%.+]] = vector.insert %in1, [[V5]] [1]
//       CHECK: [[CST:%.+]] = arith.constant dense<false>
//       CHECK: [[V7:%.+]] = vector.insert %in2, [[CST]] [0]
//       CHECK: [[V8:%.+]] = vector.insert %in3, [[V7]] [1]
//       CHECK: [[V9:%.+]] = arc.vectorize ([[V6]]), ([[V8]])
//       CHECK: ^bb0({{.*}}: vector<2xi1>, {{.*}}: vector<2xi1>):
//       CHECK: arc.vectorize.return {{.*}} : vector<2xi1>
//       CHECK: [[V10:%.+]] = vector.extract [[V9]][0]
//       CHECK: [[V11:%.+]] = vector.extract [[V9]][1]
//       CHECK: hw.output [[V3]], [[V4]], [[V10]], [[V11]]

// -----

hw.module @boundary_already_vectorized(in %in0: i1, in %in1: i1, in %in2: i1, out out0: i1, out out1: i1) {
  %cst = arith.constant dense<0> : vector<2xi1>
  %0 = vector.insert %in0, %cst[0] : i1 into vector<2xi1>
  %1 = vector.insert %in1, %0[1] : i1 into vector<2xi1>
  %2 = vector.broadcast %in2 : i1 to vector<2xi1>
  %3 = arc.vectorize (%1), (%2) :
    (vector<2xi1>, vector<2xi1>) -> (vector<2xi1>) {
  ^bb0(%arg0: i1, %arg1: i1):
    %4 = arith.andi %arg0, %arg1 : i1
    arc.vectorize.return %4 : i1
  }
  %4 = vector.extract %2[0] : i1 from vector<2xi1>
  %5 = vector.extract %2[1] : i1 from vector<2xi1>
  hw.output %4, %5 : i1, i1
}

// CHECK-LABEL: hw.module @boundary_already_vectorized
//       CHECK:  [[CST:%.+]] = arith.constant dense<false>
//       CHECK:  [[V0:%.+]] = vector.insert %in0, [[CST]] [0]
//       CHECK:  [[V1:%.+]] = vector.insert %in1, [[V0]] [1]
//       CHECK:  [[V2:%.+]] = vector.broadcast %in2
//       CHECK:  [[V3:%.+]] = arc.vectorize ([[V1]]), ([[V2]]) :
//       CHECK:  [[V4:%.+]] = vector.extract [[V2]][0]
//       CHECK:  [[V5:%.+]] = vector.extract [[V2]][1]
//       CHECK:  hw.output [[V4]], [[V5]]
