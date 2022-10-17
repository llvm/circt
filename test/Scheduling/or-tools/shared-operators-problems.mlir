// REQUIRES: or-tools
// RUN: circt-opt %s -test-cpsat-scheduler=with=SharedOperatorsProblem -allow-unregistered-dialect | FileCheck %s -check-prefix=CPSAT

// CPSAT-LABEL: full_load
func.func @full_load(%a0 : i32, %a1 : i32, %a2 : i32, %a3 : i32, %a4 : i32, %a5 : i32) -> i32 attributes {
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
  // CPSAT: return
  // CPSAT-SAME: cpSatStartTime = 7
  return { problemStartTime = 7 } %5 : i32
}

// CPSAT-LABEL: partial_load
func.func @partial_load(%a0 : i32, %a1 : i32, %a2 : i32, %a3 : i32, %a4 : i32, %a5 : i32) -> i32 attributes {
  operatortypes = [
    { name = "add", latency = 3, limit = 3},
    { name = "_0", latency = 0 }
  ] } {
  %0 = arith.addi %a0, %a1 { opr = "add", problemStartTime = 0 } : i32
  %1 = arith.addi %a1, %a1 { opr = "add", problemStartTime = 0 } : i32
  %2 = arith.addi %a2, %a3 { opr = "add", problemStartTime = 0 } : i32
  %3 = arith.addi %a3, %a4 { opr = "add", problemStartTime = 1 } : i32
  %4 = arith.addi %a4, %a5 { opr = "add", problemStartTime = 1 } : i32
  %5 = "barrier"(%0, %1, %2, %3, %4) { opr = "_0", problemStartTime = 4 } : (i32, i32, i32, i32, i32) -> i32
  // CPSAT: return
  // CPSAT-SAME: cpSatStartTime = 4
  return { problemStartTime = 4 } %5 : i32
}

// CPSAT-LABEL: multiple
func.func @multiple(%a0 : i32, %a1 : i32, %a2 : i32, %a3 : i32, %a4 : i32, %a5 : i32) -> i32 attributes {
  operatortypes = [
    { name = "slowAdd", latency = 3, limit = 2},
    { name = "fastAdd", latency = 1, limit = 1},
    { name = "_0", latency = 0 }
  ] } {
  %0 = arith.addi %a0, %a1 { opr = "slowAdd", problemStartTime = 0 } : i32
  %1 = arith.addi %a1, %a1 { opr = "slowAdd", problemStartTime = 0 } : i32
  %2 = arith.addi %a2, %a3 { opr = "fastAdd", problemStartTime = 0 } : i32
  %3 = arith.addi %a3, %a4 { opr = "slowAdd", problemStartTime = 1 } : i32
  %4 = arith.addi %a4, %a5 { opr = "fastAdd", problemStartTime = 1 } : i32
  %5 = "barrier"(%0, %1, %2, %3, %4) { opr = "_0", problemStartTime = 4 } : (i32, i32, i32, i32, i32) -> i32
  // CPSAT: return
  // CPSAT-SAME: cpSatStartTime = 4
  return { problemStartTime = 4 } %5 : i32
}
