// RUN: handshake-runner %s 2,3,4,5 | FileCheck %s
// BROKEN: circt-opt -create-dataflow %s | handshake-runner - 2,3,4,5 | FileCheck %s
// CHECK: 5 5,3,4,5

module {
  func @main(%0: memref<4xi32>) -> i32{
    %c0 = constant 0 : index
    %c5 = constant 5 : i32
    memref.store %c5, %0[%c0] : memref<4xi32>
    return %c5 : i32
    }

}
