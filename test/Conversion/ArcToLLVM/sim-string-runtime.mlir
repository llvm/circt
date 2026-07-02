// RUN: circt-opt %s --lower-arc-to-llvm | FileCheck %s

// CHECK-DAG: llvm.func @circt_sim_string_cmp(!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @circt_sim_string_concat(!llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @circt_sim_string_int_to_string(!llvm.ptr, i32) -> !llvm.ptr
// CHECK-DAG: llvm.func @circt_sim_string_get_char(!llvm.ptr, i32) -> i8
// CHECK-DAG: llvm.func @circt_sim_string_substr(!llvm.ptr, i32, i32) -> !llvm.ptr

// CHECK-LABEL: llvm.func @string_runtime(
// CHECK-SAME:    %[[LHS:[^:]+]]: !llvm.ptr
// CHECK-SAME:    %[[RHS:[^:]+]]: !llvm.ptr
// CHECK-SAME:    %[[IDX:[^:]+]]: i32
// CHECK-SAME:    %[[END:[^:]+]]: i32
// CHECK-SAME:    %[[VALUE:[^:]+]]: i40
func.func @string_runtime(%lhs: !sim.dstring, %rhs: !sim.dstring, %idx: i32, %end: i32, %value: i40) {
  %cmp = sim.string.cmp %lhs, %rhs
  %cat = sim.string.concat (%lhs, %rhs)
  %legacy_ch = sim.string.get %lhs[%idx]
  %ch = sim.string.get_char %lhs[%idx]
  %sub = sim.string.substr %lhs[%idx : %end]
  %str = sim.string.int_to_string %value : i40
  return
  // CHECK: llvm.call @circt_sim_string_cmp(%[[LHS]], %[[RHS]]) : (!llvm.ptr, !llvm.ptr) -> i32
  // CHECK: llvm.call @circt_sim_string_concat(%[[LHS]], %[[RHS]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  // CHECK: llvm.call @circt_sim_string_get_char(%[[LHS]], %[[IDX]]) : (!llvm.ptr, i32) -> i8
  // CHECK: llvm.call @circt_sim_string_get_char(%[[LHS]], %[[IDX]]) : (!llvm.ptr, i32) -> i8
  // CHECK: llvm.call @circt_sim_string_substr(%[[LHS]], %[[IDX]], %[[END]]) : (!llvm.ptr, i32, i32) -> !llvm.ptr
  // CHECK: %[[BITS:.+]] = llvm.mlir.constant(40 : i32) : i32
  // CHECK: llvm.call @circt_sim_string_int_to_string({{.*}}, %[[BITS]]) : (!llvm.ptr, i32) -> !llvm.ptr
  // CHECK: llvm.return
}

// CHECK-LABEL: llvm.func @string_concat_zero()
func.func @string_concat_zero() -> !sim.dstring {
  %cat = sim.string.concat ()
  return %cat : !sim.dstring
  // CHECK: %[[EMPTY:.+]] = llvm.mlir.addressof @_sim_str_0 : !llvm.ptr
  // CHECK: llvm.return %[[EMPTY]] : !llvm.ptr
}

// CHECK-LABEL: llvm.func @string_concat_single(
// CHECK-SAME:    %[[LHS:[^:]+]]: !llvm.ptr
func.func @string_concat_single(%lhs: !sim.dstring) -> !sim.dstring {
  %cat = sim.string.concat (%lhs)
  return %cat : !sim.dstring
  // CHECK: llvm.return %[[LHS]] : !llvm.ptr
}

// CHECK-LABEL: llvm.func @string_int_to_string_i128(
// CHECK-SAME:    %[[VALUE:[^:]+]]: i128
func.func @string_int_to_string_i128(%value: i128) -> !sim.dstring {
  %str = sim.string.int_to_string %value : i128
  return %str : !sim.dstring
  // CHECK: llvm.store
  // CHECK: llvm.store
  // CHECK: %[[BITS:.+]] = llvm.mlir.constant(128 : i32) : i32
  // CHECK: llvm.call @circt_sim_string_int_to_string({{.*}}, %[[BITS]]) : (!llvm.ptr, i32) -> !llvm.ptr
  // CHECK: llvm.return
}
