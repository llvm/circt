// RUN: handshake-runner %s | FileCheck %s
// RUN: circt-opt -create-dataflow %s | handshake-runner | FileCheck %s
// CHECK: 0

module {
  func @main() -> index {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c10 = constant 10 : index
    %0 = alloc() : memref<100xi32>
    %1 = alloc() : memref<100xi32>
    %2 = alloc() : memref<100xi32>
    %3 = alloc() : memref<100xi32>
    %4 = alloc() : memref<100xi32>
    %5 = alloc() : memref<100xi32>
    %6 = alloc() : memref<100xi32>
    %7 = alloc() : memref<100xi32>
    %8 = alloc() : memref<100xi32>
    %9 = alloc() : memref<100xi32>
    %10 = alloc() : memref<100xi1>
    br ^bb1(%c0 : index)
  ^bb1(%11: index):	// 2 preds: ^bb0, ^bb5
    %12 = cmpi slt, %11, %c10 : index
    cond_br %12, ^bb2, ^bb6
  ^bb2:	// pred: ^bb1
    %13 = load %0[%11] : memref<100xi32>
    %14 = load %2[%11] : memref<100xi32>
    %15 = load %0[%11] : memref<100xi32>
    %16 = load %2[%11] : memref<100xi32>
    %17 = load %10[%11] : memref<100xi1>
    cond_br %17, ^bb3, ^bb4
  ^bb3:	// pred: ^bb2
    %18 = load %6[%11] : memref<100xi32>
    %19 = load %9[%11] : memref<100xi32>
    %20 = muli %13, %18 : i32
    %21 = muli %15, %18 : i32
    %22 = addi %20, %15 : i32
    %23 = subi %13, %21 : i32
    %24 = muli %22, %19 : i32
    %25 = muli %23, %19 : i32
    store %24, %5[%11] : memref<100xi32>
    store %25, %4[%11] : memref<100xi32>
    br ^bb5
  ^bb4:	// pred: ^bb2
    %26 = load %7[%11] : memref<100xi32>
    %27 = load %8[%11] : memref<100xi32>
    %28 = muli %13, %26 : i32
    %29 = muli %15, %26 : i32
    %30 = addi %29, %13 : i32
    %31 = subi %28, %15 : i32
    %32 = muli %30, %27 : i32
    %33 = muli %31, %27 : i32
    store %32, %5[%11] : memref<100xi32>
    store %33, %4[%11] : memref<100xi32>
    br ^bb5
  ^bb5:	// 2 preds: ^bb3, ^bb4
    %34 = addi %11, %c1 : index
    br ^bb1(%34 : index)
  ^bb6:	// pred: ^bb1
    return %c0 : index
  }
}
