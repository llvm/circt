// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

moore.class.classdecl @HandleCompareClass {
}

// CHECK-LABEL: func.func @ChandleCompare
// CHECK-SAME: ([[LHS:%.+]]: !llvm.ptr, [[RHS:%.+]]: !llvm.ptr)
func.func @ChandleCompare(%lhs: !moore.chandle, %rhs: !moore.chandle) -> (!moore.i1, !moore.i1, !moore.i1, !moore.i1) {
  // CHECK: [[EQ:%.+]] = llvm.icmp "eq" [[LHS]], [[RHS]] : !llvm.ptr
  // CHECK: [[NE:%.+]] = llvm.icmp "ne" [[LHS]], [[RHS]] : !llvm.ptr
  // CHECK: [[CASE_EQ:%.+]] = llvm.icmp "eq" [[LHS]], [[RHS]] : !llvm.ptr
  // CHECK: [[CASE_NE:%.+]] = llvm.icmp "ne" [[LHS]], [[RHS]] : !llvm.ptr
  // CHECK: return [[EQ]], [[NE]], [[CASE_EQ]], [[CASE_NE]] : i1, i1, i1, i1
  %eq = moore.handle_eq %lhs, %rhs : !moore.chandle : !moore.chandle -> !moore.i1
  %ne = moore.handle_ne %lhs, %rhs : !moore.chandle : !moore.chandle -> !moore.i1
  %case_eq = moore.handle_case_eq %lhs, %rhs : !moore.chandle, !moore.chandle
  %case_ne = moore.handle_case_ne %lhs, %rhs : !moore.chandle, !moore.chandle
  return %eq, %ne, %case_eq, %case_ne : !moore.i1, !moore.i1, !moore.i1, !moore.i1
}

// CHECK-LABEL: func.func @ClassCompare
// CHECK-SAME: ([[LHS:%.+]]: !llvm.ptr, [[RHS:%.+]]: !llvm.ptr)
func.func @ClassCompare(%lhs: !moore.class<@HandleCompareClass>, %rhs: !moore.class<@HandleCompareClass>) -> (!moore.i1, !moore.i1, !moore.i1, !moore.i1) {
  // CHECK: [[CLASS_EQ:%.+]] = llvm.icmp "eq" [[LHS]], [[RHS]] : !llvm.ptr
  // CHECK: [[CLASS_NE:%.+]] = llvm.icmp "ne" [[LHS]], [[RHS]] : !llvm.ptr
  // CHECK: [[CLASS_CASE_EQ:%.+]] = llvm.icmp "eq" [[LHS]], [[RHS]] : !llvm.ptr
  // CHECK: [[CLASS_CASE_NE:%.+]] = llvm.icmp "ne" [[LHS]], [[RHS]] : !llvm.ptr
  // CHECK: return [[CLASS_EQ]], [[CLASS_NE]], [[CLASS_CASE_EQ]], [[CLASS_CASE_NE]] : i1, i1, i1, i1
  %eq = moore.handle_eq %lhs, %rhs : !moore.class<@HandleCompareClass> : !moore.class<@HandleCompareClass> -> !moore.i1
  %ne = moore.handle_ne %lhs, %rhs : !moore.class<@HandleCompareClass> : !moore.class<@HandleCompareClass> -> !moore.i1
  %case_eq = moore.handle_case_eq %lhs, %rhs : !moore.class<@HandleCompareClass>, !moore.class<@HandleCompareClass>
  %case_ne = moore.handle_case_ne %lhs, %rhs : !moore.class<@HandleCompareClass>, !moore.class<@HandleCompareClass>
  return %eq, %ne, %case_eq, %case_ne : !moore.i1, !moore.i1, !moore.i1, !moore.i1
}
