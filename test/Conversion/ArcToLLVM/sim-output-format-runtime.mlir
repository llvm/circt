// RUN: circt-opt %s --lower-arc-to-llvm | FileCheck %s

// CHECK-DAG: llvm.func @arcRuntimeIR_format(!llvm.ptr, ...)
// CHECK-DAG: llvm.func @arcRuntimeIR_formatToString(!llvm.ptr, ...) -> !llvm.ptr
// CHECK-DAG: llvm.func @arcRuntimeIR_setTimeFormat(i32, i32, !llvm.ptr, i32)

func.func @format_ints() {
  %v8 = arith.constant 42 : i8
  %v2 = arith.constant 2 : i2
  %u2 = arith.constant 1 : i2
  %bin = sim.fmt.bin %v8 : i8
  %int = sim.fmt.int 10 0 0 %v8 : i8
  %fv = sim.fmt.fvint 2 0 0 %v2, %u2 : i2
  %msg = sim.fmt.concat (%bin, %int, %fv)
  // CHECK: llvm.call @arcRuntimeIR_format
  sim.proc.print %msg
  return
}

func.func @timeformat_and_time() {
  // CHECK: llvm.call @arcRuntimeIR_setTimeFormat
  sim.proc.timeformat -9, 2, " ns", 0
  %time = arith.constant 1500000 : i64
  %fmt = sim.fmt.time %time, width 0 : i64
  // CHECK: llvm.call @arcRuntimeIR_format
  sim.proc.print %fmt
  return
}

func.func @format_to_string(%input: !sim.dstring) -> !sim.dstring {
  %str_fmt = sim.fmt.string %input specifierWidth 5 : !sim.dstring
  %value = arith.constant 42 : i8
  %int_fmt = sim.fmt.int 10 0 0 %value : i8
  %msg = sim.fmt.concat (%str_fmt, %int_fmt)
  %str = sim.fmt.to_string %msg
  return %str : !sim.dstring
  // CHECK: llvm.call @arcRuntimeIR_formatToString
}
