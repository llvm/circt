// RUN: circt-lec %s -c1=modA -c2=modB --emit-mlir | FileCheck %s --check-prefix=CHECK-MLIR
// RUN: circt-lec %s -c1=modA -c2=modB --emit-llvm | FileCheck %s --check-prefix=CHECK-LLVM

// CHECK-MLIR: func.func @modA()
// CHECK-LLVM: define void @modA()

hw.module @modA(in %in: i32, out out: i32) {
  hw.output %in : i32
}

hw.module @modB(in %in: i32, out out: i32) {
  hw.output %in : i32
}
