// RUN: handshake-runner %s | FileCheck %s
// RUN: circt-opt -lower-std-to-handshake %s | handshake-runner | FileCheck %s
// CHECK: 0

module {
  func @main() -> index {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c4 = constant 4 : index
    %c0_i32 = constant 0 : i32
    %0 = memref.alloc() : memref<256xi32>
    %1 = memref.alloc() : memref<256xi32>
    %2 = memref.alloc() : memref<256xi32>
    %3 = memref.alloc() : memref<1xi32>
    br ^bb1(%c0 : index)
  ^bb1(%4: index):	// 2 preds: ^bb0, ^bb8
    %5 = cmpi slt, %4, %c4 : index
    cond_br %5, ^bb2, ^bb9
  ^bb2:	// pred: ^bb1
    br ^bb3(%c0 : index)
  ^bb3(%6: index):	// 2 preds: ^bb2, ^bb7
    %7 = cmpi slt, %6, %c4 : index
    cond_br %7, ^bb4, ^bb8
  ^bb4:	// pred: ^bb3
    memref.store %c0_i32, %3[%c0] : memref<1xi32>
    %8 = muli %4, %c4 : index
    br ^bb5(%c0 : index)
  ^bb5(%9: index):	// 2 preds: ^bb4, ^bb6
    %10 = cmpi slt, %9, %c4 : index
    cond_br %10, ^bb6, ^bb7
  ^bb6:	// pred: ^bb5
    %11 = muli %9, %c4 : index
    %12 = addi %8, %9 : index
    %13 = addi %11, %6 : index
    %14 = memref.load %0[%12] : memref<256xi32>
    %15 = memref.load %1[%13] : memref<256xi32>
    %16 = memref.load %3[%c0] : memref<1xi32>
    %17 = muli %14, %15 : i32
    %18 = addi %16, %17 : i32
    memref.store %18, %3[%c0] : memref<1xi32>
    %19 = addi %9, %c1 : index
    br ^bb5(%19 : index)
  ^bb7:	// pred: ^bb5
    %20 = addi %8, %6 : index
    %21 = memref.load %3[%c0] : memref<1xi32>
    memref.store %21, %2[%20] : memref<256xi32>
    %22 = addi %6, %c1 : index
    br ^bb3(%22 : index)
  ^bb8:	// pred: ^bb3
    %23 = addi %4, %c1 : index
    br ^bb1(%23 : index)
  ^bb9:	// pred: ^bb1
    return %c0 : index
  }
}
