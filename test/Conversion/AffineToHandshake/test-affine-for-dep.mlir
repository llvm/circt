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

// CHECK: handshake.func @affine_for_dep(%arg0: none, ...) -> none {
// CHECK-NEXT:   %0:2 = "handshake.memory"(%addressResults_12) {id = 0 : i32, ld_count = 1 : i32, lsq = false, st_count = 0 : i32, type = memref<10xf32>} : (index) -> (f32, none)
// CHECK-NEXT:   %1:5 = "handshake.memory"(%49#0, %49#1, %addressResults, %addressResults_13) {id = 0 : i32, ld_count = 2 : i32, lsq = false, st_count = 1 : i32, type = memref<10xf32>} : (f32, index, index, index) -> (f32, f32, none, none, none)
// CHECK-NEXT:   %2:2 = "handshake.fork"(%1#4) {control = false} : (none) -> (none, none)
// CHECK-NEXT:   %3:2 = "handshake.fork"(%1#3) {control = false} : (none) -> (none, none)
// CHECK-NEXT:   %4 = "handshake.merge"(%arg0) : (none) -> none
// CHECK-NEXT:   "handshake.sink"(%4) : (none) -> ()
// CHECK-NEXT:   %5:5 = "handshake.fork"(%arg0) {control = true} : (none) -> (none, none, none, none, none)
// CHECK-NEXT:   %6 = "handshake.constant"(%5#3) {value = 0 : index} : (none) -> index
// CHECK-NEXT:   %7 = alloc() : memref<10xf32>
// CHECK-NEXT:   %8 = alloc() : memref<10xf32>
// CHECK-NEXT:   %9 = "handshake.constant"(%5#2) {value = 1 : index} : (none) -> index
// CHECK-NEXT:   %10 = "handshake.constant"(%5#1) {value = 10 : index} : (none) -> index
// CHECK-NEXT:   %11 = "handshake.constant"(%5#0) {value = 1 : index} : (none) -> index
// CHECK-NEXT:   %12 = "handshake.branch"(%5#4) {control = true} : (none) -> none
// CHECK-NEXT:   %13 = "handshake.branch"(%6) {control = false} : (index) -> index
// CHECK-NEXT:   %14 = "handshake.branch"(%7) {control = false} : (memref<10xf32>) -> memref<10xf32>
// CHECK-NEXT:   %15 = "handshake.branch"(%8) {control = false} : (memref<10xf32>) -> memref<10xf32>
// CHECK-NEXT:   %16 = "handshake.branch"(%9) {control = false} : (index) -> index
// CHECK-NEXT:   %17 = "handshake.branch"(%10) {control = false} : (index) -> index
// CHECK-NEXT:   %18 = "handshake.branch"(%11) {control = false} : (index) -> index
// CHECK-NEXT:   %19 = "handshake.mux"(%26#5, %17, %55) : (index, index, index) -> index
// CHECK-NEXT:   %20:2 = "handshake.fork"(%19) {control = false} : (index) -> (index, index)
// CHECK-NEXT:   %21 = "handshake.mux"(%26#4, %14, %51) : (index, memref<10xf32>, memref<10xf32>) -> memref<10xf32>
// CHECK-NEXT:   %22 = "handshake.mux"(%26#3, %15, %52) : (index, memref<10xf32>, memref<10xf32>) -> memref<10xf32>
// CHECK-NEXT:   %23 = "handshake.mux"(%26#2, %13, %53) : (index, index, index) -> index
// CHECK-NEXT:   %24 = "handshake.mux"(%26#1, %18, %54) : (index, index, index) -> index
// CHECK-NEXT:   %25:2 = "handshake.control_merge"(%12, %56) {control = true} : (none, none) -> (none, index)
// CHECK-NEXT:   %26:6 = "handshake.fork"(%25#1) {control = false} : (index) -> (index, index, index, index, index, index)
// CHECK-NEXT:   %27 = "handshake.mux"(%26#0, %16, %57) : (index, index, index) -> index
// CHECK-NEXT:   %28:2 = "handshake.fork"(%27) {control = false} : (index) -> (index, index)
// CHECK-NEXT:   %29 = cmpi "slt", %28#1, %20#1 : index
// CHECK-NEXT:   %30:7 = "handshake.fork"(%29) {control = false} : (i1) -> (i1, i1, i1, i1, i1, i1, i1)
// CHECK-NEXT:   %trueResult, %falseResult = "handshake.conditional_branch"(%30#6, %20#0) {control = false} : (i1, index) -> (index, index)
// CHECK-NEXT:   "handshake.sink"(%falseResult) : (index) -> ()
// CHECK-NEXT:   %trueResult_0, %falseResult_1 = "handshake.conditional_branch"(%30#5, %21) {control = false} : (i1, memref<10xf32>) -> (memref<10xf32>, memref<10xf32>)
// CHECK-NEXT:   "handshake.sink"(%falseResult_1) : (memref<10xf32>) -> ()
// CHECK-NEXT:   %trueResult_2, %falseResult_3 = "handshake.conditional_branch"(%30#4, %22) {control = false} : (i1, memref<10xf32>) -> (memref<10xf32>, memref<10xf32>)
// CHECK-NEXT:   "handshake.sink"(%falseResult_3) : (memref<10xf32>) -> ()
// CHECK-NEXT:   %trueResult_4, %falseResult_5 = "handshake.conditional_branch"(%30#3, %23) {control = false} : (i1, index) -> (index, index)
// CHECK-NEXT:   "handshake.sink"(%falseResult_5) : (index) -> ()
// CHECK-NEXT:   %trueResult_6, %falseResult_7 = "handshake.conditional_branch"(%30#2, %24) {control = false} : (i1, index) -> (index, index)
// CHECK-NEXT:   "handshake.sink"(%falseResult_7) : (index) -> ()
// CHECK-NEXT:   %trueResult_8, %falseResult_9 = "handshake.conditional_branch"(%30#1, %25#0) {control = true} : (i1, none) -> (none, none)
// CHECK-NEXT:   %trueResult_10, %falseResult_11 = "handshake.conditional_branch"(%30#0, %28#0) {control = false} : (i1, index) -> (index, index)
// CHECK-NEXT:   "handshake.sink"(%falseResult_11) : (index) -> ()
// CHECK-NEXT:   %31 = "handshake.merge"(%trueResult_0) : (memref<10xf32>) -> memref<10xf32>
// CHECK-NEXT:   %32 = "handshake.merge"(%trueResult_10) : (index) -> index
// CHECK-NEXT:   %33:4 = "handshake.fork"(%32) {control = false} : (index) -> (index, index, index, index)
// CHECK-NEXT:   %34 = "handshake.merge"(%trueResult_2) : (memref<10xf32>) -> memref<10xf32>
// CHECK-NEXT:   %35 = "handshake.merge"(%trueResult_4) : (index) -> index
// CHECK-NEXT:   %36:2 = "handshake.fork"(%35) {control = false} : (index) -> (index, index)
// CHECK-NEXT:   %37 = "handshake.merge"(%trueResult_6) : (index) -> index
// CHECK-NEXT:   %38:2 = "handshake.fork"(%37) {control = false} : (index) -> (index, index)
// CHECK-NEXT:   %39 = "handshake.merge"(%trueResult) : (index) -> index
// CHECK-NEXT:   %40:2 = "handshake.control_merge"(%trueResult_8) {control = true} : (none) -> (none, index)
// CHECK-NEXT:   %41:5 = "handshake.fork"(%40#0) {control = true} : (none) -> (none, none, none, none, none)
// CHECK-NEXT:   %42 = "handshake.join"(%41#1, %3#0, %2#0, %1#2, %0#1) {control = true} : (none, none, none, none, none) -> none
// CHECK-NEXT:   "handshake.sink"(%40#1) : (index) -> ()
// CHECK-NEXT:   %43, %addressResults = "handshake.load"(%33#2, %1#0, %41#4) : (index, f32, none) -> (f32, index)
// CHECK-NEXT:   %44, %addressResults_12 = "handshake.load"(%33#1, %0#0, %41#0) : (index, f32, none) -> (f32, index)
// CHECK-NEXT:   %45 = mulf %43, %44 : f32
// CHECK-NEXT:   %46, %addressResults_13 = "handshake.load"(%36#0, %1#1, %41#3) : (index, f32, none) -> (f32, index)
// CHECK-NEXT:   %47 = addf %45, %46 : f32
// CHECK-NEXT:   %48 = "handshake.join"(%41#2, %3#1, %2#1) {control = true} : (none, none, none) -> none
// CHECK-NEXT:   %49:2 = "handshake.store"(%47, %33#0, %48) : (f32, index, none) -> (f32, index)
// CHECK-NEXT:   %50 = addi %33#3, %38#1 : index
// CHECK-NEXT:   %51 = "handshake.branch"(%31) {control = false} : (memref<10xf32>) -> memref<10xf32>
// CHECK-NEXT:   %52 = "handshake.branch"(%34) {control = false} : (memref<10xf32>) -> memref<10xf32>
// CHECK-NEXT:   %53 = "handshake.branch"(%36#1) {control = false} : (index) -> index
// CHECK-NEXT:   %54 = "handshake.branch"(%38#0) {control = false} : (index) -> index
// CHECK-NEXT:   %55 = "handshake.branch"(%39) {control = false} : (index) -> index
// CHECK-NEXT:   %56 = "handshake.branch"(%42) {control = true} : (none) -> none
// CHECK-NEXT:   %57 = "handshake.branch"(%50) {control = false} : (index) -> index
// CHECK-NEXT:   %58:2 = "handshake.control_merge"(%falseResult_9) {control = true} : (none) -> (none, index)
// CHECK-NEXT:   "handshake.sink"(%58#1) : (index) -> ()
// CHECK-NEXT:   handshake.return %58#0 : none
// CHECK-NEXT: }
