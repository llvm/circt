// RUN: handshake-runner %s | FileCheck %s
// RUN: circt-opt -lower-std-to-handshake %s | handshake-runner | FileCheck %s
// CHECK: 0

module {
  func @main() -> index {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c4 = constant 4 : index
    %0 = memref.alloc() : memref<256xi32>
    br ^bb1(%c0 : index)
  ^bb1(%1: index):	// 2 preds: ^bb0, ^bb11
    %2 = cmpi slt, %1, %c4 : index
    cond_br %2, ^bb2, ^bb12
  ^bb2:	// pred: ^bb1
    br ^bb3(%c0 : index)
  ^bb3(%3: index):	// 2 preds: ^bb2, ^bb10
    %4 = cmpi slt, %3, %c4 : index
    cond_br %4, ^bb4, ^bb11
  ^bb4:	// pred: ^bb3
    br ^bb5(%c0 : index)
  ^bb5(%5: index):	// 2 preds: ^bb4, ^bb9
    %6 = cmpi slt, %5, %c4 : index
    cond_br %6, ^bb6, ^bb10
  ^bb6:	// pred: ^bb5
    %7 = muli %3, %c4 : index
    %8 = muli %1, %c4 : index
    %9 = addi %7, %5 : index
    %10 = addi %7, %1 : index
    %11 = addi %8, %5 : index
    %12 = memref.load %0[%9] : memref<256xi32>
    %13 = memref.load %0[%10] : memref<256xi32>
    %14 = memref.load %0[%11] : memref<256xi32>
    %15 = addi %13, %14 : i32
    %16 = cmpi ult, %12, %15 : i32
    cond_br %16, ^bb7, ^bb8
  ^bb7:	// pred: ^bb6
    memref.store %12, %0[%9] : memref<256xi32>
    br ^bb9
  ^bb8:	// pred: ^bb6
    memref.store %15, %0[%9] : memref<256xi32>
    br ^bb9
  ^bb9:	// 2 preds: ^bb7, ^bb8
    %17 = addi %5, %c1 : index
    br ^bb5(%17 : index)
  ^bb10:	// pred: ^bb5
    %18 = addi %3, %c1 : index
    br ^bb3(%18 : index)
  ^bb11:	// pred: ^bb3
    %19 = addi %1, %c1 : index
    br ^bb1(%19 : index)
  ^bb12:	// pred: ^bb1
    return %c0 : index
  }
}
