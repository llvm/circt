// RUN: handshake-runner %s "1,2" "3,4" | FileCheck %s
// RUN: circt-opt -lower-cf-to-handshake -handshake-materialize-forks-sinks %s | handshake-runner - "1,2" "3,4" | FileCheck %s
// CHECK: 3 1,2 3,4

// Each memref argument must be initialized from its own positional argument.
// The returned value is %b[0], so binding %b to the first argument (as the
// runner used to do) would print 1 instead of 3, and the second memref would
// come back as "1,2".

module {
  func.func @main(%a: memref<2xi32>, %b: memref<2xi32>) -> i32 {
    %c0 = arith.constant 0 : index
    %0 = memref.load %b[%c0] : memref<2xi32>
    return %0 : i32
  }
}
