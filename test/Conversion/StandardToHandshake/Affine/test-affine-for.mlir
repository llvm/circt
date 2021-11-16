// RUN: circt-opt %s -lower-std-to-handshake -split-input-file | FileCheck %s

// -----

// Simple affine.for with an empty loop body.

func @empty_body () -> () {
  affine.for %i = 0 to 10 { }
  return
}

// CHECK: handshake.func @empty_body(%arg0: none, ...) -> none attributes {argNames = ["inCtrl"], resNames = ["outCtrl"]} {
// CHECK:   %0:4 = "handshake.fork"(%arg0) {control = true} : (none) -> (none, none, none, none)
// CHECK:   %1 = "handshake.constant"(%0#2) {value = 0 : index} : (none) -> index
// CHECK:   %2 = "handshake.constant"(%0#1) {value = 10 : index} : (none) -> index
// CHECK:   %3 = "handshake.constant"(%0#0) {value = 1 : index} : (none) -> index
// CHECK:   %4 = "handshake.branch"(%0#3) {control = true} : (none) -> none
// CHECK:   %5 = "handshake.branch"(%1) {control = false} : (index) -> index
// CHECK:   %6 = "handshake.branch"(%2) {control = false} : (index) -> index
// CHECK:   %7 = "handshake.branch"(%3) {control = false} : (index) -> index
// CHECK:   %8 = "handshake.mux"(%12#2, %6, %32) : (index, index, index) -> index
// CHECK:   %9:2 = "handshake.fork"(%8) {control = false} : (index) -> (index, index)
// CHECK:   %10 = "handshake.mux"(%12#1, %7, %31) : (index, index, index) -> index
// CHECK:   %11:2 = "handshake.control_merge"(%4, %33) {control = true} : (none, none) -> (none, index)
// CHECK:   %12:3 = "handshake.fork"(%11#1) {control = false} : (index) -> (index, index, index)
// CHECK:   %13 = "handshake.mux"(%12#0, %5, %34) : (index, index, index) -> index
// CHECK:   %14:2 = "handshake.fork"(%13) {control = false} : (index) -> (index, index)
// CHECK:   %15 = arith.cmpi slt, %14#1, %9#1 : index
// CHECK:   %16:4 = "handshake.fork"(%15) {control = false} : (i1) -> (i1, i1, i1, i1)
// CHECK:   %trueResult, %falseResult = "handshake.conditional_branch"(%16#3, %9#0) {control = false} : (i1, index) -> (index, index)
// CHECK:   "handshake.sink"(%falseResult) : (index) -> ()
// CHECK:   %trueResult_0, %falseResult_1 = "handshake.conditional_branch"(%16#2, %10) {control = false} : (i1, index) -> (index, index)
// CHECK:   "handshake.sink"(%falseResult_1) : (index) -> ()
// CHECK:   %trueResult_2, %falseResult_3 = "handshake.conditional_branch"(%16#1, %11#0) {control = true} : (i1, none) -> (none, none)
// CHECK:   %trueResult_4, %falseResult_5 = "handshake.conditional_branch"(%16#0, %14#0) {control = false} : (i1, index) -> (index, index)
// CHECK:   "handshake.sink"(%falseResult_5) : (index) -> ()
// CHECK:   %17 = "handshake.merge"(%trueResult_4) : (index) -> index
// CHECK:   %18 = "handshake.merge"(%trueResult_0) : (index) -> index
// CHECK:   %19 = "handshake.merge"(%trueResult) : (index) -> index
// CHECK:   %20:2 = "handshake.control_merge"(%trueResult_2) {control = true} : (none) -> (none, index)
// CHECK:   "handshake.sink"(%20#1) : (index) -> ()
// CHECK:   %21 = "handshake.branch"(%17) {control = false} : (index) -> index
// CHECK:   %22 = "handshake.branch"(%18) {control = false} : (index) -> index
// CHECK:   %23 = "handshake.branch"(%19) {control = false} : (index) -> index
// CHECK:   %24 = "handshake.branch"(%20#0) {control = true} : (none) -> none
// CHECK:   %25 = "handshake.merge"(%21) : (index) -> index
// CHECK:   %26 = "handshake.merge"(%22) : (index) -> index
// CHECK:   %27:2 = "handshake.fork"(%26) {control = false} : (index) -> (index, index)
// CHECK:   %28 = "handshake.merge"(%23) : (index) -> index
// CHECK:   %29:2 = "handshake.control_merge"(%24) {control = true} : (none) -> (none, index)
// CHECK:   "handshake.sink"(%29#1) : (index) -> ()
// CHECK:   %30 = arith.addi %25, %27#1 : index
// CHECK:   %31 = "handshake.branch"(%27#0) {control = false} : (index) -> index
// CHECK:   %32 = "handshake.branch"(%28) {control = false} : (index) -> index
// CHECK:   %33 = "handshake.branch"(%29#0) {control = true} : (none) -> none
// CHECK:   %34 = "handshake.branch"(%30) {control = false} : (index) -> index
// CHECK:   %35:2 = "handshake.control_merge"(%falseResult_3) {control = true} : (none) -> (none, index)
// CHECK:   "handshake.sink"(%35#1) : (index) -> ()
// CHECK:   return %35#0 : none
// CHECK: }

// -----

// Simple load store pair in the loop body.

func @load_store () -> () {
  %A = memref.alloc() : memref<10xf32>
  affine.for %i = 0 to 10 {
    %0 = affine.load %A[%i] : memref<10xf32>
    affine.store %0, %A[%i] : memref<10xf32>
  }
  return
}

// CHECK: handshake.func @load_store(%arg0: none, ...) -> none attributes {argNames = ["inCtrl"], resNames = ["outCtrl"]} {
// CHECK:   %0:3 = "handshake.memory"(%28#0, %28#1, %addressResults) {id = 0 : i32, ld_count = 1 : i32, lsq = false, st_count = 1 : i32, type = memref<10xf32>} : (f32, index, index) -> (f32, none, none)
// CHECK:   %1:2 = "handshake.fork"(%0#2) {control = true} : (none) -> (none, none)
// CHECK:   %2:4 = "handshake.fork"(%arg0) {control = true} : (none) -> (none, none, none, none)
// CHECK:   %3 = "handshake.constant"(%2#2) {value = 0 : index} : (none) -> index
// CHECK:   %4 = "handshake.constant"(%2#1) {value = 10 : index} : (none) -> index
// CHECK:   %5 = "handshake.constant"(%2#0) {value = 1 : index} : (none) -> index
// CHECK:   %6 = "handshake.branch"(%2#3) {control = true} : (none) -> none
// CHECK:   %7 = "handshake.branch"(%3) {control = false} : (index) -> index
// CHECK:   %8 = "handshake.branch"(%4) {control = false} : (index) -> index
// CHECK:   %9 = "handshake.branch"(%5) {control = false} : (index) -> index
// CHECK:   %10 = "handshake.mux"(%14#2, %8, %40) : (index, index, index) -> index
// CHECK:   %11:2 = "handshake.fork"(%10) {control = false} : (index) -> (index, index)
// CHECK:   %12 = "handshake.mux"(%14#1, %9, %39) : (index, index, index) -> index
// CHECK:   %13:2 = "handshake.control_merge"(%6, %41) {control = true} : (none, none) -> (none, index)
// CHECK:   %14:3 = "handshake.fork"(%13#1) {control = false} : (index) -> (index, index, index)
// CHECK:   %15 = "handshake.mux"(%14#0, %7, %42) : (index, index, index) -> index
// CHECK:   %16:2 = "handshake.fork"(%15) {control = false} : (index) -> (index, index)
// CHECK:   %17 = arith.cmpi slt, %16#1, %11#1 : index
// CHECK:   %18:4 = "handshake.fork"(%17) {control = false} : (i1) -> (i1, i1, i1, i1)
// CHECK:   %trueResult, %falseResult = "handshake.conditional_branch"(%18#3, %11#0) {control = false} : (i1, index) -> (index, index)
// CHECK:   "handshake.sink"(%falseResult) : (index) -> ()
// CHECK:   %trueResult_0, %falseResult_1 = "handshake.conditional_branch"(%18#2, %12) {control = false} : (i1, index) -> (index, index)
// CHECK:   "handshake.sink"(%falseResult_1) : (index) -> ()
// CHECK:   %trueResult_2, %falseResult_3 = "handshake.conditional_branch"(%18#1, %13#0) {control = true} : (i1, none) -> (none, none)
// CHECK:   %trueResult_4, %falseResult_5 = "handshake.conditional_branch"(%18#0, %16#0) {control = false} : (i1, index) -> (index, index)
// CHECK:   "handshake.sink"(%falseResult_5) : (index) -> ()
// CHECK:   %19 = "handshake.merge"(%trueResult_4) : (index) -> index
// CHECK:   %20:3 = "handshake.fork"(%19) {control = false} : (index) -> (index, index, index)
// CHECK:   %21 = "handshake.merge"(%trueResult_0) : (index) -> index
// CHECK:   %22 = "handshake.merge"(%trueResult) : (index) -> index
// CHECK:   %23:2 = "handshake.control_merge"(%trueResult_2) {control = true} : (none) -> (none, index)
// CHECK:   %24:3 = "handshake.fork"(%23#0) {control = true} : (none) -> (none, none, none)
// CHECK:   %25 = "handshake.join"(%24#2, %1#1, %0#1) {control = true} : (none, none, none) -> none
// CHECK:   "handshake.sink"(%23#1) : (index) -> ()
// CHECK:   %26, %addressResults = "handshake.load"(%20#2, %0#0, %24#1) : (index, f32, none) -> (f32, index)
// CHECK:   %27 = "handshake.join"(%24#0, %1#0) {control = true} : (none, none) -> none
// CHECK:   %28:2 = "handshake.store"(%26, %20#1, %27) : (f32, index, none) -> (f32, index)
// CHECK:   %29 = "handshake.branch"(%20#0) {control = false} : (index) -> index
// CHECK:   %30 = "handshake.branch"(%21) {control = false} : (index) -> index
// CHECK:   %31 = "handshake.branch"(%22) {control = false} : (index) -> index
// CHECK:   %32 = "handshake.branch"(%25) {control = true} : (none) -> none
// CHECK:   %33 = "handshake.merge"(%29) : (index) -> index
// CHECK:   %34 = "handshake.merge"(%30) : (index) -> index
// CHECK:   %35:2 = "handshake.fork"(%34) {control = false} : (index) -> (index, index)
// CHECK:   %36 = "handshake.merge"(%31) : (index) -> index
// CHECK:   %37:2 = "handshake.control_merge"(%32) {control = true} : (none) -> (none, index)
// CHECK:   "handshake.sink"(%37#1) : (index) -> ()
// CHECK:   %38 = arith.addi %33, %35#1 : index
// CHECK:   %39 = "handshake.branch"(%35#0) {control = false} : (index) -> index
// CHECK:   %40 = "handshake.branch"(%36) {control = false} : (index) -> index
// CHECK:   %41 = "handshake.branch"(%37#0) {control = true} : (none) -> none
// CHECK:   %42 = "handshake.branch"(%38) {control = false} : (index) -> index
// CHECK:   %43:2 = "handshake.control_merge"(%falseResult_3) {control = true} : (none) -> (none, index)
// CHECK:   "handshake.sink"(%43#1) : (index) -> ()
// CHECK:   return %43#0 : none
// CHECK: }

// TODO: affine expr in loop bounds
// TODO: nested loops
// TODO: yield carries values
