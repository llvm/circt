// RUN: circt-opt %s -affine-to-handshake | FileCheck %s

func @affine_for () -> () {
  %A = alloc() : memref<10xf32>
  affine.for %i = 0 to 10 {
    %0 = affine.load %A[%i] : memref<10xf32>
    affine.store %0, %A[%i] : memref<10xf32>
  }
  return
}

// CHECK:       handshake.func @affine_for(%arg0: none, ...) -> none {
// CHECK-NEXT:    %0:3 = "handshake.memory"(%30#0, %30#1, %addressResults) {id = 0 : i32, ld_count = 1 : i32, lsq = false, st_count = 1 : i32, type = memref<10xf32>} : (f32, index, index) -> (f32, none, none)
// CHECK-NEXT:    %1:2 = "handshake.fork"(%0#2) {control = false} : (none) -> (none, none)
// CHECK-NEXT:    %2 = "handshake.merge"(%arg0) : (none) -> none
// CHECK-NEXT:    "handshake.sink"(%2) : (none) -> ()
// CHECK-NEXT:    %3:4 = "handshake.fork"(%arg0) {control = true} : (none) -> (none, none, none, none)
// CHECK-NEXT:    %4 = "handshake.constant"(%3#2) {value = 0 : index} : (none) -> index
// CHECK-NEXT:    %5 = "handshake.constant"(%3#1) {value = 10 : index} : (none) -> index
// CHECK-NEXT:    %6 = "handshake.constant"(%3#0) {value = 1 : index} : (none) -> index
// CHECK-NEXT:    %7 = "handshake.branch"(%3#3) {control = true} : (none) -> none
// CHECK-NEXT:    %8 = "handshake.branch"(%4) {control = false} : (index) -> index
// CHECK-NEXT:    %9 = "handshake.branch"(%5) {control = false} : (index) -> index
// CHECK-NEXT:    %10 = "handshake.branch"(%6) {control = false} : (index) -> index
// CHECK-NEXT:    %11 = "handshake.mux"(%15#2, %9, %33) : (index, index, index) -> index
// CHECK-NEXT:    %12:2 = "handshake.fork"(%11) {control = false} : (index) -> (index, index)
// CHECK-NEXT:    %13 = "handshake.mux"(%15#1, %10, %32) : (index, index, index) -> index
// CHECK-NEXT:    %14:2 = "handshake.control_merge"(%7, %34) {control = true} : (none, none) -> (none, index)
// CHECK-NEXT:    %15:3 = "handshake.fork"(%14#1) {control = false} : (index) -> (index, index, index)
// CHECK-NEXT:    %16 = "handshake.mux"(%15#0, %8, %35) : (index, index, index) -> index
// CHECK-NEXT:    %17:2 = "handshake.fork"(%16) {control = false} : (index) -> (index, index)
// CHECK-NEXT:    %18 = cmpi "slt", %17#1, %12#1 : index
// CHECK-NEXT:    %19:4 = "handshake.fork"(%18) {control = false} : (i1) -> (i1, i1, i1, i1)
// CHECK-NEXT:    %trueResult, %falseResult = "handshake.conditional_branch"(%19#3, %12#0) {control = false} : (i1, index) -> (index, index)
// CHECK-NEXT:    "handshake.sink"(%falseResult) : (index) -> ()
// CHECK-NEXT:    %trueResult_0, %falseResult_1 = "handshake.conditional_branch"(%19#2, %13) {control = false} : (i1, index) -> (index, index)
// CHECK-NEXT:    "handshake.sink"(%falseResult_1) : (index) -> ()
// CHECK-NEXT:    %trueResult_2, %falseResult_3 = "handshake.conditional_branch"(%19#1, %14#0) {control = true} : (i1, none) -> (none, none)
// CHECK-NEXT:    %trueResult_4, %falseResult_5 = "handshake.conditional_branch"(%19#0, %17#0) {control = false} : (i1, index) -> (index, index)
// CHECK-NEXT:    "handshake.sink"(%falseResult_5) : (index) -> ()
// CHECK-NEXT:    %20 = "handshake.merge"(%trueResult_4) : (index) -> index
// CHECK-NEXT:    %21:3 = "handshake.fork"(%20) {control = false} : (index) -> (index, index, index)
// CHECK-NEXT:    %22 = "handshake.merge"(%trueResult_0) : (index) -> index
// CHECK-NEXT:    %23:2 = "handshake.fork"(%22) {control = false} : (index) -> (index, index)
// CHECK-NEXT:    %24 = "handshake.merge"(%trueResult) : (index) -> index
// CHECK-NEXT:    %25:2 = "handshake.control_merge"(%trueResult_2) {control = true} : (none) -> (none, index)
// CHECK-NEXT:    %26:3 = "handshake.fork"(%25#0) {control = true} : (none) -> (none, none, none)
// CHECK-NEXT:    %27 = "handshake.join"(%26#2, %1#1, %0#1) {control = true} : (none, none, none) -> none
// CHECK-NEXT:    "handshake.sink"(%25#1) : (index) -> ()
// CHECK-NEXT:    %28, %addressResults = "handshake.load"(%21#1, %0#0, %26#1) : (index, f32, none) -> (f32, index)
// CHECK-NEXT:    %29 = "handshake.join"(%26#0, %1#0) {control = true} : (none, none) -> none
// CHECK-NEXT:    %30:2 = "handshake.store"(%28, %21#0, %29) : (f32, index, none) -> (f32, index)
// CHECK-NEXT:    %31 = addi %21#2, %23#1 : index
// CHECK-NEXT:    %32 = "handshake.branch"(%23#0) {control = false} : (index) -> index
// CHECK-NEXT:    %33 = "handshake.branch"(%24) {control = false} : (index) -> index
// CHECK-NEXT:    %34 = "handshake.branch"(%27) {control = true} : (none) -> none
// CHECK-NEXT:    %35 = "handshake.branch"(%31) {control = false} : (index) -> index
// CHECK-NEXT:    %36:2 = "handshake.control_merge"(%falseResult_3) {control = true} : (none) -> (none, index)
// CHECK-NEXT:    "handshake.sink"(%36#1) : (index) -> ()
// CHECK-NEXT:    handshake.return %36#0 : none
// CHECK-NEXT:  }
