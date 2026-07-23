// RUN: circt-opt %s --convert-hw-to-llvm | FileCheck %s

// Allocas backing array temporaries must land in the frame's entry block.
// Emitted at a use site inside a loop they push a fresh slot on every
// iteration (stack slots are only reclaimed on function return), which
// overflows the default stack in run-to-completion process loops (ivtest
// br_gh661a/b: a dynamic-index array access in a 256k-iteration loop).
// The value stores stay at the use site (here the input is loop-invariant,
// so the spill cache also hoists the store).

// CHECK-LABEL: @getInLoop
func.func @getInLoop(%arr: !hw.array<16xi32>, %idx: i4, %c: i1) -> i32 {
  // CHECK: llvm.alloca {{.*}} x !llvm.array<16 x i32>
  // CHECK-NEXT: llvm.store
  // CHECK: cf.br
  cf.br ^bb1
// CHECK: ^bb1:
^bb1:
  // CHECK-NOT: llvm.alloca
  // CHECK: llvm.getelementptr
  // CHECK: llvm.load
  %v = hw.array_get %arr[%idx] : !hw.array<16xi32>, i4
  cf.cond_br %c, ^bb1, ^bb2
^bb2:
  return %v : i32
}

// CHECK-LABEL: @injectInLoop
func.func @injectInLoop(%arr: !hw.array<16xi32>, %idx: i4, %e: i32, %c: i1) -> !hw.array<16xi32> {
  // CHECK: llvm.alloca {{.*}} x !llvm.array<16 x i32>
  // CHECK-NEXT: cf.br
  cf.br ^bb1
// CHECK: ^bb1:
^bb1:
  // CHECK-NOT: llvm.alloca
  // CHECK: llvm.store
  // CHECK: llvm.getelementptr
  // CHECK: llvm.store
  // CHECK: llvm.load
  %r = hw.array_inject %arr[%idx], %e : !hw.array<16xi32>, i4
  cf.cond_br %c, ^bb1, ^bb2
^bb2:
  return %r : !hw.array<16xi32>
}
