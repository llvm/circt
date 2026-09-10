// RUN: ! (arcilator %s --run --jit-entry=foo 2> %t) && FileCheck --input-file=%t %s
// REQUIRES: arcilator-jit

// CHECK: entry point 'foo' was found but on an operation of type 'llvm.mlir.global' while an LLVM function was expected

llvm.mlir.global @foo(0 : i32) : i32
