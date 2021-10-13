// RUN: mlir-opt --convert-std-to-llvm %s | mlir-cpu-runner --entry-point-result=i64 | FileCheck %s
// RUN: circt-opt -lower-std-to-handshake %s | handshake-runner | FileCheck %s
// RUN: handshake-runner %s | FileCheck %s
// CHECK: 42
module {
  func @main() -> index {
    %c1 = constant 1 : index
    %c42 = constant 42 : index
    %c1_0 = constant 1 : index
    br ^bb1(%c1 : index)
  ^bb1(%0: index):	// 2 preds: ^bb0, ^bb2
    %1 = cmpi slt, %0, %c42 : index
    cond_br %1, ^bb2, ^bb3
  ^bb2:	// pred: ^bb1
//    call @body(%0) : (index) -> ()
    %2 = addi %0, %c1_0 : index
    br ^bb1(%2 : index)
  ^bb3:	// pred: ^bb1
    return %0 : index
  }
}
