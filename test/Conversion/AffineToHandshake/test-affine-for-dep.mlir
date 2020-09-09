// RUN: circt-opt %s -affine-to-handshake | FileCheck %s

func @affine_for_dep () -> () {
  %c0 = constant 0 : index
  %A = alloc() : memref<10xf32>
  %B = alloc() : memref<10xf32>
  affine.for %i = 1 to 10 {
    %0 = affine.load %A[%i] : memref<10xf32>
    %1 = affine.load %B[%i] : memref<10xf32>
    %2 = mulf %0, %1 : f32
    %3 = affine.load %A[%c0] : memref<10xf32>
    %4 = addf %2, %3 : f32 
    affine.store %4, %A[%i] : memref<10xf32>
  }
  return
}

// CHECK:       handshake.func @affine_for_dep(%arg0: none, ...) -> none {
// CHECK-NEXT:    %0:2 = "handshake.memory"(%addressResults_6) {id = 1 : i32, ld_count = 1 : i32, lsq = false, st_count = 0 : i32, type = memref<10xf32>} : (index) -> (f32, none)
// CHECK-NEXT:    %1:5 = "handshake.memory"(%37#0, %37#1, %addressResults, %addressResults_7) {id = 0 : i32, ld_count = 2 : i32, lsq = false, st_count = 1 : i32, type = memref<10xf32>} : (f32, index, index, index) -> (f32, f32, none, none, none)
// CHECK-NEXT:    %2:2 = "handshake.fork"(%1#4) {control = false} : (none) -> (none, none)
// CHECK-NEXT:    %3:2 = "handshake.fork"(%1#3) {control = false} : (none) -> (none, none)
// CHECK-NEXT:    %4 = "handshake.merge"(%arg0) : (none) -> none
// CHECK-NEXT:    "handshake.sink"(%4) : (none) -> ()
// CHECK-NEXT:    %5:5 = "handshake.fork"(%arg0) {control = true} : (none) -> (none, none, none, none, none)
// CHECK-NEXT:    %6 = "handshake.constant"(%5#3) {value = 0 : index} : (none) -> index
// CHECK-NEXT:    %7 = "handshake.constant"(%5#2) {value = 1 : index} : (none) -> index
// CHECK-NEXT:    %8 = "handshake.constant"(%5#1) {value = 10 : index} : (none) -> index
// CHECK-NEXT:    %9 = "handshake.constant"(%5#0) {value = 1 : index} : (none) -> index
// CHECK-NEXT:    %10 = "handshake.branch"(%5#4) {control = true} : (none) -> none
// CHECK-NEXT:    %11 = "handshake.branch"(%7) {control = false} : (index) -> index
// CHECK-NEXT:    %12 = "handshake.branch"(%8) {control = false} : (index) -> index
// CHECK-NEXT:    %13 = "handshake.branch"(%9) {control = false} : (index) -> index
// CHECK-NEXT:    %14 = "handshake.mux"(%18#2, %12, %40) : (index, index, index) -> index
// CHECK-NEXT:    %15:2 = "handshake.fork"(%14) {control = false} : (index) -> (index, index)
// CHECK-NEXT:    %16 = "handshake.mux"(%18#1, %13, %39) : (index, index, index) -> index
// CHECK-NEXT:    %17:2 = "handshake.control_merge"(%10, %41) {control = true} : (none, none) -> (none, index)
// CHECK-NEXT:    %18:3 = "handshake.fork"(%17#1) {control = false} : (index) -> (index, index, index)
// CHECK-NEXT:    %19 = "handshake.mux"(%18#0, %11, %42) : (index, index, index) -> index
// CHECK-NEXT:    %20:2 = "handshake.fork"(%19) {control = false} : (index) -> (index, index)
// CHECK-NEXT:    %21 = cmpi "slt", %20#1, %15#1 : index
// CHECK-NEXT:    %22:4 = "handshake.fork"(%21) {control = false} : (i1) -> (i1, i1, i1, i1)
// CHECK-NEXT:    %trueResult, %falseResult = "handshake.conditional_branch"(%22#3, %15#0) {control = false} : (i1, index) -> (index, index)
// CHECK-NEXT:    "handshake.sink"(%falseResult) : (index) -> ()
// CHECK-NEXT:    %trueResult_0, %falseResult_1 = "handshake.conditional_branch"(%22#2, %16) {control = false} : (i1, index) -> (index, index)
// CHECK-NEXT:    "handshake.sink"(%falseResult_1) : (index) -> ()
// CHECK-NEXT:    %trueResult_2, %falseResult_3 = "handshake.conditional_branch"(%22#1, %17#0) {control = true} : (i1, none) -> (none, none)
// CHECK-NEXT:    %trueResult_4, %falseResult_5 = "handshake.conditional_branch"(%22#0, %20#0) {control = false} : (i1, index) -> (index, index)
// CHECK-NEXT:    "handshake.sink"(%falseResult_5) : (index) -> ()
// CHECK-NEXT:    %23 = "handshake.merge"(%trueResult_4) : (index) -> index
// CHECK-NEXT:    %24:4 = "handshake.fork"(%23) {control = false} : (index) -> (index, index, index, index)
// CHECK-NEXT:    %25 = "handshake.merge"(%trueResult_0) : (index) -> index
// CHECK-NEXT:    %26:2 = "handshake.fork"(%25) {control = false} : (index) -> (index, index)
// CHECK-NEXT:    %27 = "handshake.merge"(%trueResult) : (index) -> index
// CHECK-NEXT:    %28:2 = "handshake.control_merge"(%trueResult_2) {control = true} : (none) -> (none, index)
// CHECK-NEXT:    %29:5 = "handshake.fork"(%28#0) {control = true} : (none) -> (none, none, none, none, none)
// CHECK-NEXT:    %30 = "handshake.join"(%29#1, %3#0, %2#0, %1#2, %0#1) {control = true} : (none, none, none, none, none) -> none
// CHECK-NEXT:    "handshake.sink"(%28#1) : (index) -> ()
// CHECK-NEXT:    %31, %addressResults = "handshake.load"(%24#2, %1#0, %29#4) : (index, f32, none) -> (f32, index)
// CHECK-NEXT:    %32, %addressResults_6 = "handshake.load"(%24#1, %0#0, %29#0) : (index, f32, none) -> (f32, index)
// CHECK-NEXT:    %33 = mulf %31, %32 : f32
// CHECK-NEXT:    %34, %addressResults_7 = "handshake.load"(%6, %1#1, %29#3) : (index, f32, none) -> (f32, index)
// CHECK-NEXT:    %35 = addf %33, %34 : f32
// CHECK-NEXT:    %36 = "handshake.join"(%29#2, %3#1, %2#1) {control = true} : (none, none, none) -> none
// CHECK-NEXT:    %37:2 = "handshake.store"(%35, %24#0, %36) : (f32, index, none) -> (f32, index)
// CHECK-NEXT:    %38 = addi %24#3, %26#1 : index
// CHECK-NEXT:    %39 = "handshake.branch"(%26#0) {control = false} : (index) -> index
// CHECK-NEXT:    %40 = "handshake.branch"(%27) {control = false} : (index) -> index
// CHECK-NEXT:    %41 = "handshake.branch"(%30) {control = true} : (none) -> none
// CHECK-NEXT:    %42 = "handshake.branch"(%38) {control = false} : (index) -> index
// CHECK-NEXT:    %43:2 = "handshake.control_merge"(%falseResult_3) {control = true} : (none) -> (none, index)
// CHECK-NEXT:    "handshake.sink"(%43#1) : (index) -> ()
// CHECK-NEXT:    handshake.return %43#0 : none
// CHECK-NEXT:  }
