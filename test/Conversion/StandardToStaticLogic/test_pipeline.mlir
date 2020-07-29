// RUN: circt-opt -create-pipeline %s | FileCheck %s

// CHECK: staticlogic.pipeline @pipeline0(...) -> (index, index) {
// CHECK:   %c1 = constant 1 : index
// CHECK:   %c42 = constant 42 : index
// CHECK:   "staticlogic.return"(%c1, %c42) : (index, index) -> ()
// CHECK: }
// CHECK: staticlogic.pipeline @pipeline1(%arg0: index, %arg1: index, ...) -> (i1, index, index) {
// CHECK:   %0 = cmpi "slt", %arg0, %arg1 : index
// CHECK:   "staticlogic.return"(%0, %arg0, %arg1) : (i1, index, index) -> ()
// CHECK: }
// CHECK: staticlogic.pipeline @pipeline2(%arg0: index, %arg1: index, ...) -> (index, index) {
// CHECK:   %c1 = constant 1 : index
// CHECK:   %0 = addi %arg0, %c1 : index
// CHECK:   "staticlogic.return"(%0, %arg1) : (index, index) -> ()
// CHECK: }

func @simple_loop() {
^bb0:
  br ^bb1
// CHECK:       ^bb1:  // pred: ^bb0
// CHECK-NEXT:    %0:2 = "staticlogic.instance"() {module = @pipeline0} : () -> (index, index)
// CHECK-NEXT:    br ^bb2(%0#0, %0#1 : index, index)
^bb1:	// pred: ^bb0
  %c1 = constant 1 : index
  %c42 = constant 42 : index
  br ^bb2(%c1 : index)

// CHECK:       ^bb2(%1: index, %2: index):  // 2 preds: ^bb1, ^bb3
// CHECK-NEXT:    %3:3 = "staticlogic.instance"(%1, %2) {module = @pipeline1} : (index, index) -> (i1, index, index)
// CHECK-NEXT:    cond_br %3#0, ^bb3(%3#1, %3#2 : index, index), ^bb4
^bb2(%0: index):	// 2 preds: ^bb1, ^bb3
  %1 = cmpi "slt", %0, %c42 : index
  cond_br %1, ^bb3, ^bb4

// CHECK:       ^bb3(%4: index, %5: index):  // pred: ^bb2
// CHECK-NEXT:    %6:2 = "staticlogic.instance"(%4, %5) {module = @pipeline2} : (index, index) -> (index, index)
// CHECK-NEXT:    br ^bb2(%6#0, %6#1 : index, index)
^bb3:	// pred: ^bb2
  %c1_0 = constant 1 : index
  %2 = addi %0, %c1_0 : index
  br ^bb2(%2 : index)
^bb4:	// pred: ^bb2
  return
}