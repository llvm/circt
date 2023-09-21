// RUN: circt-opt --pass-pipeline='builtin.module(ibis.class(ibis.method(ibis-reblock)))' %s | FileCheck %s

// CHECK-LABEL:   ibis.class @Reblock {
// CHECK:           %[[VAL_0:.*]] = ibis.this @Reblock
// CHECK:           ibis.method @foo(%[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32) -> i32 {
// CHECK:             %[[VAL_3:.*]]:2 = ibis.sblock () -> (i32, i32) attributes {maxThreads = 2 : i64} {
// CHECK:               %[[VAL_4:.*]] = arith.addi %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:               %[[VAL_5:.*]] = arith.addi %[[VAL_4]], %[[VAL_2]] : i32
// CHECK:               %[[VAL_6:.*]] = arith.addi %[[VAL_5]], %[[VAL_2]] : i32
// CHECK:               ibis.sblock.return %[[VAL_5]], %[[VAL_6]] : i32, i32
// CHECK:             }
// CHECK:             cf.br ^bb1(%[[VAL_7:.*]]#0 : i32)
// CHECK:           ^bb1(%[[VAL_8:.*]]: i32):
// CHECK:             ibis.return %[[VAL_7]]#1 : i32
// CHECK:           }
// CHECK:         }

ibis.class @Reblock {
  %this = ibis.this @Reblock

  ibis.method @foo(%arg0 : i32, %arg1 : i32) -> i32 {
      ibis.sblock.inline.begin {maxThreads = 2}
      %inner = arith.addi %arg0, %arg1 : i32
      %0 = arith.addi %inner, %arg1 : i32
      %1 = arith.addi %0, %arg1 : i32
      ibis.sblock.inline.end
      // %0 is used within the same MLIR block but outside the scope.
      cf.br ^bb1(%0 : i32)
    ^bb1(%barg : i32):
      // %1 is used in a different MLIR block (dominated by the sblock parent block).
      ibis.return %1 : i32
  }
}
