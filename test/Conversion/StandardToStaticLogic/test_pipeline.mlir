// RUN: circt-opt -create-pipeline %s | FileCheck %s

// CHECK:       module {
// CHECK-LABEL:   func @simple_loop() {
// CHECK:           br ^bb1
// CHECK:         ^bb1: // pred: ^bb0
// CHECK:           %[[VAL_0:.*]]:2 = "staticlogic.pipeline"() ({
// CHECK:             %c1 = arith.constant 1 : index
// CHECK:             %c42 = arith.constant 42 : index
// CHECK:             "staticlogic.return"(%c1, %c42) : (index, index) -> ()
// CHECK:           }) : () -> (index, index)
// CHECK:           br ^bb2(%[[VAL_0:.*]]#0 : index)
// CHECK:         ^bb2(%[[VAL_1:.*]]: index): // 2 preds: ^bb1, ^bb3
// CHECK:           %[[VAL_2:.*]] = "staticlogic.pipeline"(%[[VAL_1:.*]], %[[VAL_0:.*]]#1) ({
// CHECK:           ^bb0(%[[ARG_0:.*]]: index, %[[ARG_1:.*]]: index):
// CHECK:             %[[TMP_0:.*]] = arith.cmpi slt, %[[ARG_0:.*]], %[[ARG_1:.*]] : index
// CHECK:             "staticlogic.return"(%[[TMP_0:.*]]) : (i1) -> ()
// CHECK:           }) : (index, index) -> i1
// CHECK:           cond_br %[[VAL_2:.*]], ^bb3, ^bb4
// CHECK:         ^bb3: // pred: ^bb2
// CHECK:           %[[VAL_3:.*]] = "staticlogic.pipeline"(%[[VAL_1:.*]]) ({
// CHECK:           ^bb0(%[[ARG_0:.*]]: index):
// CHECK:             %c1 = arith.constant 1 : index
// CHECK:             %[[TMP_1:.*]] = arith.addi %[[ARG_0:.*]], %c1 : index
// CHECK:             "staticlogic.return"(%[[TMP_1:.*]]) : (index) -> ()
// CHECK:           }) : (index) -> index
// CHECK:           br ^bb2(%[[VAL_3:.*]] : index)
// CHECK:         ^bb4: // pred: ^bb2
// CHECK:           return
// CHECK:         }
// CHECK:       }

func @simple_loop() {
^bb0:
  br ^bb1
^bb1:	// pred: ^bb0
  %c1 = arith.constant 1 : index
  %c42 = arith.constant 42 : index
  br ^bb2(%c1 : index)
^bb2(%0: index):	// 2 preds: ^bb1, ^bb3
  %1 = arith.cmpi slt, %0, %c42 : index
  cond_br %1, ^bb3, ^bb4
^bb3:	// pred: ^bb2
  %c1_0 = arith.constant 1 : index
  %2 = arith.addi %0, %c1_0 : index
  br ^bb2(%2 : index)
^bb4:	// pred: ^bb2
  return
}
