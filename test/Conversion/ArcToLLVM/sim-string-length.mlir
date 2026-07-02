// RUN: circt-opt %s --lower-arc-to-llvm | FileCheck %s

// CHECK-DAG: llvm.func @circt_sim_string_len(!llvm.ptr) -> i32
// CHECK-DAG: llvm.mlir.global internal constant @_sim_str_0("hello\00")

// CHECK-LABEL: llvm.func @string_length_arg(
// CHECK-SAME:    %[[STR:.+]]: !llvm.ptr
func.func @string_length_arg(%str: !sim.dstring) -> i64 {
  %len = sim.string.length %str
  return %len : i64
  // CHECK: %[[LEN32:.+]] = llvm.call @circt_sim_string_len(%[[STR]]) : (!llvm.ptr) -> i32
  // CHECK: %[[LEN64:.+]] = llvm.zext %[[LEN32]] : i32 to i64
  // CHECK: llvm.return %[[LEN64]] : i64
}

// CHECK-LABEL: llvm.func @string_length_literal()
func.func @string_length_literal() -> i64 {
  %str = sim.string.literal "hello"
  %len = sim.string.length %str
  return %len : i64
  // CHECK: %[[STR:.+]] = llvm.mlir.addressof @_sim_str_0 : !llvm.ptr
  // CHECK: %[[LEN32:.+]] = llvm.call @circt_sim_string_len(%[[STR]]) : (!llvm.ptr) -> i32
  // CHECK: %[[LEN64:.+]] = llvm.zext %[[LEN32]] : i32 to i64
  // CHECK: llvm.return %[[LEN64]] : i64
}
