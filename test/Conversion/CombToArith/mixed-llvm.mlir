// RUN: circt-opt %s --convert-comb-to-arith | FileCheck %s

module {
  func.func @mixed(%arg0: i1) -> i64 {
    %c1 = hw.constant true
    %x = comb.xor %arg0, %c1 : i1
    %z = llvm.zext %x : i1 to i64
    return %z : i64
  }
}

// CHECK-LABEL: func.func @mixed
// CHECK: arith.constant true
// CHECK: arith.xori
// CHECK: llvm.zext
// CHECK-NOT: comb.xor
// CHECK-NOT: hw.constant
