// RUN: circt-verilog %s | FileCheck %s
// RUN: cat %s | circt-verilog --format mlir | FileCheck %s
// REQUIRES: slang

// CHECK: hw.module @Foo() {
// CHECK: }
moore.module @Foo() {
}
