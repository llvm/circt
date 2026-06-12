// RUN: arcilator --run %s --jit-entry=entry | FileCheck %s
// REQUIRES: arcilator-jit

// Exercise the SystemVerilog integer-formatting runtime bound into the JIT.
// Each call writes to stdout with no separator, so the checks below match the
// concatenation of every print in order.

llvm.mlir.global internal constant @b42(42 : i8) : i8
llvm.mlir.global internal constant @bff(-1 : i8) : i8
llvm.mlir.global internal constant @b10(10 : i8) : i8
llvm.mlir.global internal constant @fval(2 : i8) : i8
llvm.mlir.global internal constant @funk(1 : i8) : i8

func.func private @circt_sv_print_int(!llvm.ptr, i32, i32, i32, i32)
func.func private @circt_sv_print_fvint(!llvm.ptr, !llvm.ptr, i32, i32, i32, i32)

func.func @entry() {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  %c10 = arith.constant 10 : i32
  %c16 = arith.constant 16 : i32

  %b42 = llvm.mlir.addressof @b42 : !llvm.ptr
  %bff = llvm.mlir.addressof @bff : !llvm.ptr
  %b10 = llvm.mlir.addressof @b10 : !llvm.ptr
  %fval = llvm.mlir.addressof @fval : !llvm.ptr
  %funk = llvm.mlir.addressof @funk : !llvm.ptr

  // 42 as i8, base 10 -> "42"
  func.call @circt_sv_print_int(%b42, %c8, %c10, %c0, %c0)
      : (!llvm.ptr, i32, i32, i32, i32) -> ()
  // 0xFF as i8, base 16 lowercase -> "ff"
  func.call @circt_sv_print_int(%bff, %c8, %c16, %c0, %c0)
      : (!llvm.ptr, i32, i32, i32, i32) -> ()
  // 0xFF as i8, base 16 uppercase (flag bit0) -> "FF"
  func.call @circt_sv_print_int(%bff, %c8, %c16, %c0, %c1)
      : (!llvm.ptr, i32, i32, i32, i32) -> ()
  // low 4 bits of 0x0A, base 2 -> "1010"
  func.call @circt_sv_print_int(%b10, %c4, %c2, %c0, %c0)
      : (!llvm.ptr, i32, i32, i32, i32) -> ()
  // 0xFF as signed i8, base 10 (flag bit3) -> "-1"
  func.call @circt_sv_print_int(%bff, %c8, %c10, %c0, %c8)
      : (!llvm.ptr, i32, i32, i32, i32) -> ()
  // i2 value 0b10 with unknown mask 0b01, base 2 -> "1x"
  func.call @circt_sv_print_fvint(%fval, %funk, %c2, %c2, %c0, %c0)
      : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()

  // CHECK: 42ffFF1010-11x
  return
}
