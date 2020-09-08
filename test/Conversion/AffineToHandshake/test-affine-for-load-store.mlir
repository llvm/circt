// RUN: circt-opt %s -affine-to-handshake | FileCheck %s

func @affine_for () -> () {
  %A = alloc() : memref<10xf32>
  affine.for %i = 0 to 10 {
    %0 = affine.load %A[%i] : memref<10xf32>
    affine.store %0, %A[%i] : memref<10xf32>
  }
  return
}

// CHECK:  handshake.func @affine_for(%arg0: none, ...) -> none {
// CHECK-NEXT:    %0:3 = "handshake.memory"(%34#0, %34#1, %addressResults) {id = 0 : i32, ld_count = 1 : i32, lsq = false, st_count = 1 : i32, type = memref<10xf32>} : (f32, index, index) -> (f32, none, none)
// CHECK-NEXT:    %1:2 = "handshake.fork"(%0#2) {control = false} : (none) -> (none, none)
// CHECK-NEXT:    %2 = "handshake.merge"(%arg0) : (none) -> none
// CHECK-NEXT:    "handshake.sink"(%2) : (none) -> ()
// CHECK-NEXT:    %3:4 = "handshake.fork"(%arg0) {control = true} : (none) -> (none, none, none, none)
// CHECK-NEXT:    %4 = alloc() : memref<10xf32>
// CHECK-NEXT:    %5 = "handshake.constant"(%3#2) {value = 0 : index} : (none) -> index
// CHECK-NEXT:    %6 = "handshake.constant"(%3#1) {value = 10 : index} : (none) -> index
// CHECK-NEXT:    %7 = "handshake.constant"(%3#0) {value = 1 : index} : (none) -> index
// CHECK-NEXT:    %8 = "handshake.branch"(%3#3) {control = true} : (none) -> none
// CHECK-NEXT:    %9 = "handshake.branch"(%4) {control = false} : (memref<10xf32>) -> memref<10xf32>
// CHECK-NEXT:    %10 = "handshake.branch"(%5) {control = false} : (index) -> index
// CHECK-NEXT:    %11 = "handshake.branch"(%6) {control = false} : (index) -> index
// CHECK-NEXT:    %12 = "handshake.branch"(%7) {control = false} : (index) -> index
// CHECK-NEXT:    %13 = "handshake.mux"(%18#3, %11, %38) : (index, index, index) -> index
// CHECK-NEXT:    %14:2 = "handshake.fork"(%13) {control = false} : (index) -> (index, index)
// CHECK-NEXT:    %15 = "handshake.mux"(%18#2, %9, %36) : (index, memref<10xf32>, memref<10xf32>) -> memref<10xf32>
// CHECK-NEXT:    %16 = "handshake.mux"(%18#1, %12, %37) : (index, index, index) -> index
// CHECK-NEXT:    %17:2 = "handshake.control_merge"(%8, %39) {control = true} : (none, none) -> (none, index)
// CHECK-NEXT:    %18:4 = "handshake.fork"(%17#1) {control = false} : (index) -> (index, index, index, index)
// CHECK-NEXT:    %19 = "handshake.mux"(%18#0, %10, %40) : (index, index, index) -> index
// CHECK-NEXT:    %20:2 = "handshake.fork"(%19) {control = false} : (index) -> (index, index)
// CHECK-NEXT:    %21 = cmpi "slt", %20#1, %14#1 : index
// CHECK-NEXT:    %22:5 = "handshake.fork"(%21) {control = false} : (i1) -> (i1, i1, i1, i1, i1)
// CHECK-NEXT:    %trueResult, %falseResult = "handshake.conditional_branch"(%22#4, %14#0) {control = false} : (i1, index) -> (index, index)
// CHECK-NEXT:    "handshake.sink"(%falseResult) : (index) -> ()
// CHECK-NEXT:    %trueResult_0, %falseResult_1 = "handshake.conditional_branch"(%22#3, %15) {control = false} : (i1, memref<10xf32>) -> (memref<10xf32>, memref<10xf32>)
// CHECK-NEXT:    "handshake.sink"(%falseResult_1) : (memref<10xf32>) -> ()
// CHECK-NEXT:    %trueResult_2, %falseResult_3 = "handshake.conditional_branch"(%22#2, %16) {control = false} : (i1, index) -> (index, index)
// CHECK-NEXT:    "handshake.sink"(%falseResult_3) : (index) -> ()
// CHECK-NEXT:    %trueResult_4, %falseResult_5 = "handshake.conditional_branch"(%22#1, %17#0) {control = true} : (i1, none) -> (none, none)
// CHECK-NEXT:    %trueResult_6, %falseResult_7 = "handshake.conditional_branch"(%22#0, %20#0) {control = false} : (i1, index) -> (index, index)
// CHECK-NEXT:    "handshake.sink"(%falseResult_7) : (index) -> ()
// CHECK-NEXT:    %23 = "handshake.merge"(%trueResult_0) : (memref<10xf32>) -> memref<10xf32>
// CHECK-NEXT:    %24 = "handshake.merge"(%trueResult_6) : (index) -> index
// CHECK-NEXT:    %25:3 = "handshake.fork"(%24) {control = false} : (index) -> (index, index, index)
// CHECK-NEXT:    %26 = "handshake.merge"(%trueResult_2) : (index) -> index
// CHECK-NEXT:    %27:2 = "handshake.fork"(%26) {control = false} : (index) -> (index, index)
// CHECK-NEXT:    %28 = "handshake.merge"(%trueResult) : (index) -> index
// CHECK-NEXT:    %29:2 = "handshake.control_merge"(%trueResult_4) {control = true} : (none) -> (none, index)
// CHECK-NEXT:    %30:3 = "handshake.fork"(%29#0) {control = true} : (none) -> (none, none, none)
// CHECK-NEXT:    %31 = "handshake.join"(%30#2, %1#1, %0#1) {control = true} : (none, none, none) -> none
// CHECK-NEXT:    "handshake.sink"(%29#1) : (index) -> ()
// CHECK-NEXT:    %32, %addressResults = "handshake.load"(%25#1, %0#0, %30#1) : (index, f32, none) -> (f32, index)
// CHECK-NEXT:    %33 = "handshake.join"(%30#0, %1#0) {control = true} : (none, none) -> none
// CHECK-NEXT:    %34:2 = "handshake.store"(%32, %25#0, %33) : (f32, index, none) -> (f32, index)
// CHECK-NEXT:    %35 = addi %25#2, %27#1 : index
// CHECK-NEXT:    %36 = "handshake.branch"(%23) {control = false} : (memref<10xf32>) -> memref<10xf32>
// CHECK-NEXT:    %37 = "handshake.branch"(%27#0) {control = false} : (index) -> index
// CHECK-NEXT:    %38 = "handshake.branch"(%28) {control = false} : (index) -> index
// CHECK-NEXT:    %39 = "handshake.branch"(%31) {control = true} : (none) -> none
// CHECK-NEXT:    %40 = "handshake.branch"(%35) {control = false} : (index) -> index
// CHECK-NEXT:    %41:2 = "handshake.control_merge"(%falseResult_5) {control = true} : (none) -> (none, index)
// CHECK-NEXT:    "handshake.sink"(%41#1) : (index) -> ()
// CHECK-NEXT:    handshake.return %41#0 : none
// CHECK-NEXT:  }
