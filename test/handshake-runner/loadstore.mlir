// RUN: handshake-runner %s 2 | FileCheck %s
// RUN: circt-opt -lower-std-to-handshake %s | handshake-runner - 2 | FileCheck %s
// CHECK: 1

module {
  func @main(%arg0: index) -> (i8) {
    %0 = memref.alloc() : memref<10xi8>
    %c1 = constant 1 : i8
	 memref.store %c1, %0[%arg0] : memref<10xi8>
	 %1 = memref.load %0[%arg0] : memref<10xi8>
	 return %1 : i8
  }
}
