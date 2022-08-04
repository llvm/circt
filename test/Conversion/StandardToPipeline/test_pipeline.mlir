// RUN: circt-opt -create-pipeline %s | FileCheck %s

// CHECK:       module {
// CHECK-LABEL:   func @simple_loop() {
// CHECK:           cf.br ^bb1
// CHECK:         ^bb1: // pred: ^bb0
// CHECK:           %[[VAL_0:.*]]:2 = "pipeline.pipeline"() ({
// CHECK:             %c1 = arith.constant 1 : index
// CHECK:             %c42 = arith.constant 42 : index
// CHECK:             "pipeline.return"(%c1, %c42) : (index, index) -> ()
// CHECK:           }) : () -> (index, index)
// CHECK:           cf.br ^bb2(%[[VAL_0:.*]]#0 : index)
// CHECK:         ^bb2(%[[VAL_1:.*]]: index): // 2 preds: ^bb1, ^bb3
// CHECK:           %[[VAL_2:.*]] = "pipeline.pipeline"(%[[VAL_1:.*]], %[[VAL_0:.*]]#1) ({
// CHECK:           ^bb0(%[[ARG_0:.*]]: index, %[[ARG_1:.*]]: index):
// CHECK:             %[[TMP_0:.*]] = arith.cmpi slt, %[[ARG_0:.*]], %[[ARG_1:.*]] : index
// CHECK:             "pipeline.return"(%[[TMP_0:.*]]) : (i1) -> ()
// CHECK:           }) : (index, index) -> i1
// CHECK:           cf.cond_br %[[VAL_2:.*]], ^bb3, ^bb4
// CHECK:         ^bb3: // pred: ^bb2
// CHECK:           %[[VAL_3:.*]] = "pipeline.pipeline"(%[[VAL_1:.*]]) ({
// CHECK:           ^bb0(%[[ARG_0:.*]]: index):
// CHECK:             %c1 = arith.constant 1 : index
// CHECK:             %[[TMP_1:.*]] = arith.addi %[[ARG_0:.*]], %c1 : index
// CHECK:             "pipeline.return"(%[[TMP_1:.*]]) : (index) -> ()
// CHECK:           }) : (index) -> index
// CHECK:           cf.br ^bb2(%[[VAL_3:.*]] : index)
// CHECK:         ^bb4: // pred: ^bb2
// CHECK:           return
// CHECK:         }
// CHECK:       }

func.func @simple_loop() {
^bb0:
  cf.br ^bb1
^bb1:	// pred: ^bb0
  %c1 = arith.constant 1 : index
  %c42 = arith.constant 42 : index
  cf.br ^bb2(%c1 : index)
^bb2(%0: index):	// 2 preds: ^bb1, ^bb3
  %1 = arith.cmpi slt, %0, %c42 : index
  cf.cond_br %1, ^bb3, ^bb4
^bb3:	// pred: ^bb2
  %c1_0 = arith.constant 1 : index
  %2 = arith.addi %0, %c1_0 : index
  cf.br ^bb2(%2 : index)
^bb4:	// pred: ^bb2
  return
}
