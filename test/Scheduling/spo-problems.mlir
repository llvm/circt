// RUN: circt-opt %s -test-spo-problem -allow-unregistered-dialect
// RUN: circt-opt %s -test-simplex-scheduler=with=SharedPipelinedOperatorsProblem -allow-unregistered-dialect | FileCheck %s -check-prefix=SIMPLEX

// SIMPLEX-LABEL: full_load
func @full_load(%a0 : i32, %a1 : i32, %a2 : i32, %a3 : i32, %a4 : i32, %a5 : i32) -> i32 attributes {
  operatortypes = [
    { name = "add", latency = 3, limit = 1 },
    { name = "_0", latency = 0 }
  ] } {
  %0 = arith.addi %a0, %a1 { opr = "add", problemStartTime = 0 } : i32
  %1 = arith.addi %a1, %a1 { opr = "add", problemStartTime = 1 } : i32
  %2 = arith.addi %a2, %a3 { opr = "add", problemStartTime = 2 } : i32
  %3 = arith.addi %a3, %a4 { opr = "add", problemStartTime = 3 } : i32
  %4 = arith.addi %a4, %a5 { opr = "add", problemStartTime = 4 } : i32
  %5 = "barrier"(%0, %1, %2, %3, %4) { opr = "_0", problemStartTime = 7 } : (i32, i32, i32, i32, i32) -> i32
  // SIMPLEX: return
  // SIMPLEX-SAME: simplexStartTime = 7
  return { problemStartTime = 7 } %5 : i32
}

// SIMPLEX-LABEL: partial_load
func @partial_load(%a0 : i32, %a1 : i32, %a2 : i32, %a3 : i32, %a4 : i32, %a5 : i32) -> i32 attributes {
  operatortypes = [
    { name = "add", latency = 3, limit = 3},
    { name = "_0", latency = 0 }
  ] } {
  %0 = arith.addi %a0, %a1 { opr = "add", problemStartTime = 0 } : i32
  %1 = arith.addi %a1, %a1 { opr = "add", problemStartTime = 1 } : i32
  %2 = arith.addi %a2, %a3 { opr = "add", problemStartTime = 0 } : i32
  %3 = arith.addi %a3, %a4 { opr = "add", problemStartTime = 2 } : i32
  %4 = arith.addi %a4, %a5 { opr = "add", problemStartTime = 1 } : i32
  %5 = "barrier"(%0, %1, %2, %3, %4) { opr = "_0", problemStartTime = 10 } : (i32, i32, i32, i32, i32) -> i32
  // SIMPLEX: return
  // SIMPLEX-SAME: simplexStartTime = 4
  return { problemStartTime = 10 } %5 : i32
}

// SIMPLEX-LABEL: multiple
func @multiple(%a0 : i32, %a1 : i32, %a2 : i32, %a3 : i32, %a4 : i32, %a5 : i32) -> i32 attributes {
  operatortypes = [
    { name = "slowAdd", latency = 3, limit = 2},
    { name = "fastAdd", latency = 1, limit = 1},
    { name = "_0", latency = 0 }
  ] } {
  %0 = arith.addi %a0, %a1 { opr = "slowAdd", problemStartTime = 0 } : i32
  %1 = arith.addi %a1, %a1 { opr = "slowAdd", problemStartTime = 1 } : i32
  %2 = arith.addi %a2, %a3 { opr = "fastAdd", problemStartTime = 0 } : i32
  %3 = arith.addi %a3, %a4 { opr = "slowAdd", problemStartTime = 1 } : i32
  %4 = arith.addi %a4, %a5 { opr = "fastAdd", problemStartTime = 1 } : i32
  %5 = "barrier"(%0, %1, %2, %3, %4) { opr = "_0", problemStartTime = 10 } : (i32, i32, i32, i32, i32) -> i32
  // SIMPLEX: return
  // SIMPLEX-SAME: simplexStartTime = 4
  return { problemStartTime = 10 } %5 : i32
}
