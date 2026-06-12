// RUN: arcilator --run %s --jit-entry=entry | FileCheck --match-full-lines %s
// REQUIRES: arcilator-jit

// Exercise the SystemVerilog string runtime bound into the JIT.

llvm.mlir.global internal constant @hello("hello\00")
llvm.mlir.global internal constant @world("world\00")

func.func private @circt_sv_string_len(!llvm.ptr) -> i32
func.func private @circt_sv_strcmp(!llvm.ptr, !llvm.ptr) -> i32
func.func private @circt_sv_string_getc(!llvm.ptr, i32) -> i8
func.func private @circt_sv_string_substr(!llvm.ptr, i32, i32) -> !llvm.ptr

func.func @entry() {
  %hello = llvm.mlir.addressof @hello : !llvm.ptr
  %world = llvm.mlir.addressof @world : !llvm.ptr

  // CHECK: len = 00000005
  %len = func.call @circt_sv_string_len(%hello) : (!llvm.ptr) -> i32
  arc.sim.emit "len", %len : i32

  // strcmp("hello", "hello") == 0
  // CHECK: eq = 00000000
  %eq = func.call @circt_sv_strcmp(%hello, %hello) : (!llvm.ptr, !llvm.ptr) -> i32
  arc.sim.emit "eq", %eq : i32

  // "hello"[1] == 'e' == 0x65
  %c1 = arith.constant 1 : i32
  // CHECK: getc = 65
  %ch = func.call @circt_sv_string_getc(%hello, %c1) : (!llvm.ptr, i32) -> i8
  arc.sim.emit "getc", %ch : i8

  // substr("hello", 1, 3) == "ell"; check via its length.
  %c3 = arith.constant 3 : i32
  %sub = func.call @circt_sv_string_substr(%hello, %c1, %c3)
      : (!llvm.ptr, i32, i32) -> !llvm.ptr
  // CHECK: sublen = 00000003
  %sublen = func.call @circt_sv_string_len(%sub) : (!llvm.ptr) -> i32
  arc.sim.emit "sublen", %sublen : i32

  return
}
