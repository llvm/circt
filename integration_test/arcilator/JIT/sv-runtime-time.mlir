// RUN: arcilator --run %s --jit-entry=entry | FileCheck %s
// REQUIRES: arcilator-jit

// Exercise the SystemVerilog $timeformat runtime bound into the JIT: configure
// nanosecond units with two fractional digits and a " ns" suffix, then render a
// femtosecond time value (1500000 fs == 1.5 ns).

llvm.mlir.global internal constant @suffix(" ns\00")

func.func private @circt_sv_set_timeformat(i32, i32, !llvm.ptr, i32)
func.func private @circt_sv_print_time(i64, i32)

func.func @entry() {
  %unit = arith.constant -9 : i32
  %prec = arith.constant 2 : i32
  %width = arith.constant 0 : i32
  %suffix = llvm.mlir.addressof @suffix : !llvm.ptr

  // set_timeformat(unit = -9 (ns), precision = 2, suffix = " ns", width = 0)
  func.call @circt_sv_set_timeformat(%unit, %prec, %suffix, %width)
      : (i32, i32, !llvm.ptr, i32) -> ()

  %time = arith.constant 1500000 : i64
  %override = arith.constant -1 : i32
  // CHECK: 1.50 ns
  func.call @circt_sv_print_time(%time, %override) : (i64, i32) -> ()

  return
}
