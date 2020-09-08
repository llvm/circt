// RUN: circt-opt %s -affine-to-handshake | FileCheck %s

func @affine_for () -> () {
  affine.for %i = 0 to 10 {
  }
  return
}

// CHECK: handshake.func @affine_for(%[[ARG0:.*]]: none, ...) -> none {
// CHECK-NEXT:   %0 = "handshake.merge"(%[[ARG0]]) : (none) -> none
// CHECK-NEXT:   "handshake.sink"(%0) : (none) -> ()
// CHECK-NEXT:   %1:4 = "handshake.fork"(%arg0) {control = true} : (none) -> (none, none, none, none)
// CHECK-NEXT:   %2 = "handshake.constant"(%1#2) {value = 0 : index} : (none) -> index
// CHECK-NEXT:   %3 = "handshake.constant"(%1#1) {value = 10 : index} : (none) -> index
// CHECK-NEXT:   %4 = "handshake.constant"(%1#0) {value = 1 : index} : (none) -> index
// CHECK-NEXT:   %5 = "handshake.branch"(%1#3) {control = true} : (none) -> none
// CHECK-NEXT:   %6 = "handshake.branch"(%2) {control = false} : (index) -> index
// CHECK-NEXT:   %7 = "handshake.branch"(%3) {control = false} : (index) -> index
// CHECK-NEXT:   %8 = "handshake.branch"(%4) {control = false} : (index) -> index
// CHECK-NEXT:   %9 = "handshake.mux"(%13#2, %7, %25) : (index, index, index) -> index
// CHECK-NEXT:   %10:2 = "handshake.fork"(%9) {control = false} : (index) -> (index, index)
// CHECK-NEXT:   %11 = "handshake.mux"(%13#1, %8, %24) : (index, index, index) -> index
// CHECK-NEXT:   %12:2 = "handshake.control_merge"(%5, %26) {control = true} : (none, none) -> (none, index)
// CHECK-NEXT:   %13:3 = "handshake.fork"(%12#1) {control = false} : (index) -> (index, index, index)
// CHECK-NEXT:   %14 = "handshake.mux"(%13#0, %6, %27) : (index, index, index) -> index
// CHECK-NEXT:   %15:2 = "handshake.fork"(%14) {control = false} : (index) -> (index, index)
// CHECK-NEXT:   %16 = cmpi "slt", %15#1, %10#1 : index
// CHECK-NEXT:   %17:4 = "handshake.fork"(%16) {control = false} : (i1) -> (i1, i1, i1, i1)
// CHECK-NEXT:   %trueResult, %falseResult = "handshake.conditional_branch"(%17#3, %10#0) {control = false} : (i1, index) -> (index, index)
// CHECK-NEXT:   "handshake.sink"(%falseResult) : (index) -> ()
// CHECK-NEXT:   %trueResult_0, %falseResult_1 = "handshake.conditional_branch"(%17#2, %11) {control = false} : (i1, index) -> (index, index)
// CHECK-NEXT:   "handshake.sink"(%falseResult_1) : (index) -> ()
// CHECK-NEXT:   %trueResult_2, %falseResult_3 = "handshake.conditional_branch"(%17#1, %12#0) {control = true} : (i1, none) -> (none, none)
// CHECK-NEXT:   %trueResult_4, %falseResult_5 = "handshake.conditional_branch"(%17#0, %15#0) {control = false} : (i1, index) -> (index, index)
// CHECK-NEXT:   "handshake.sink"(%falseResult_5) : (index) -> ()
// CHECK-NEXT:   %18 = "handshake.merge"(%trueResult_4) : (index) -> index
// CHECK-NEXT:   %19 = "handshake.merge"(%trueResult_0) : (index) -> index
// CHECK-NEXT:   %20:2 = "handshake.fork"(%19) {control = false} : (index) -> (index, index)
// CHECK-NEXT:   %21 = "handshake.merge"(%trueResult) : (index) -> index
// CHECK-NEXT:   %22:2 = "handshake.control_merge"(%trueResult_2) {control = true} : (none) -> (none, index)
// CHECK-NEXT:   "handshake.sink"(%22#1) : (index) -> ()
// CHECK-NEXT:   %23 = addi %18, %20#1 : index
// CHECK-NEXT:   %24 = "handshake.branch"(%20#0) {control = false} : (index) -> index
// CHECK-NEXT:   %25 = "handshake.branch"(%21) {control = false} : (index) -> index
// CHECK-NEXT:   %26 = "handshake.branch"(%22#0) {control = true} : (none) -> none
// CHECK-NEXT:   %27 = "handshake.branch"(%23) {control = false} : (index) -> index
// CHECK-NEXT:   %28:2 = "handshake.control_merge"(%falseResult_3) {control = true} : (none) -> (none, index)
// CHECK-NEXT:   "handshake.sink"(%28#1) : (index) -> ()
// CHECK-NEXT:   handshake.return %28#0 : none
// CHECK-NEXT: }
