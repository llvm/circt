// RUN: mlir-opt --convert-std-to-llvm %s | mlir-cpu-runner --entry-point-result=i32 | FileCheck %s
// RUN: circt-opt -create-dataflow %s | handshake-runner | FileCheck %s
// RUN: handshake-runner %s | FileCheck %s
// CHECK: 42
module {
  func @main() -> i32 {
    %c1 = constant 1 : i32
    %c42 = constant 42 : i32
    %c1_0 = constant 1 : i32
    br ^bb1(%c1 : i32)
  ^bb1(%0: i32):	// 2 preds: ^bb0, ^bb2
    %1 = cmpi slt, %0, %c42 : i32
    cond_br %1, ^bb2, ^bb3
  ^bb2:	// pred: ^bb1
    %2 = addi %0, %c1_0 : i32
    br ^bb1(%2 : i32)
  ^bb3:	// pred: ^bb1
    return %0 : i32
  }
}
