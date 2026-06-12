// RUN: arcilator --run %s --jit-entry=entry | FileCheck --match-full-lines %s
// REQUIRES: arcilator-jit

// Exercise the SystemVerilog execution runtime bound into the JIT: a model that
// calls `circt_sv_string_len` and emits the result.

llvm.mlir.global internal constant @hello("hello\00")

func.func private @circt_sv_string_len(!llvm.ptr) -> i32

func.func @entry() {
  %p = llvm.mlir.addressof @hello : !llvm.ptr
  %len = func.call @circt_sv_string_len(%p) : (!llvm.ptr) -> i32
  // CHECK: len = 00000005
  arc.sim.emit "len", %len : i32
  return
}
