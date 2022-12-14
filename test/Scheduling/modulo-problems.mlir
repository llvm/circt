// RUN: circt-opt %s -test-modulo-problem -allow-unregistered-dialect
// RUN: circt-opt %s -test-simplex-scheduler=with=ModuloProblem -allow-unregistered-dialect | FileCheck %s -check-prefix=SIMPLEX

// SIMPLEX-LABEL: canis14_fig2
// SIMPLEX-SAME: simplexInitiationInterval = 4
func.func @canis14_fig2() attributes {
  problemInitiationInterval = 3,
  auxdeps = [ [3,0,1], [3,4] ],
  operatortypes = [
    { name = "mem_port", latency = 1, limit = 1 },
    { name = "add", latency = 1 }
  ] } {
  %0 = "dummy.load_A"() { opr = "mem_port", problemStartTime = 2 } : () -> i32
  %1 = "dummy.load_B"() { opr = "mem_port", problemStartTime = 0 } : () -> i32
  %2 = arith.addi %0, %1 { opr = "add", problemStartTime = 3 } : i32
  "dummy.store_A"(%2) { opr = "mem_port", problemStartTime = 4 } : (i32) -> ()
  // SIMPLEX: return
  // SIMPLEX-SAME: simplexStartTime = 4
  return { problemStartTime = 5 }
}

// SIMPLEX-LABEL: minII_feasible
// SIMPLEX-SAME: simplexInitiationInterval = 4
func.func @minII_feasible() attributes {
  problemInitiationInterval = 3,
  auxdeps = [ [6,1,5], [5,2,3], [6,7] ],
  operatortypes = [
    { name = "const", latency = 0 },
    { name = "phi", latency = 2 },
    { name = "xor", latency = 1 },
    { name = "sub", latency = 3, limit = 1}
  ] } {
  %0 = arith.constant { opr = "const", problemStartTime = 0 } -1 : i32
  %1 = "dummy.phi"() { opr = "phi", problemStartTime = 0 } : () -> i32
  %2 = "dummy.phi"() { opr = "phi", problemStartTime = 1 } : () -> i32
  %3 = arith.xori %1, %0 { opr = "xor", problemStartTime = 2 } : i32
  %4 = arith.subi %3, %2 { opr = "sub", problemStartTime = 3 } : i32
  %5 = arith.subi %0, %4 { opr = "sub", problemStartTime = 7 } : i32
  %6 = arith.subi %4, %5 { opr = "sub", problemStartTime = 11 } : i32
  // SIMPLEX: return
  // SIMPLEX-SAME: simplexStartTime = 14
  return { problemStartTime = 14 }
}

// SIMPLEX-LABEL: minII_infeasible
// SIMPLEX-SAME: simplexInitiationInterval = 4
func.func @minII_infeasible() -> i32 attributes {
  problemInitiationInterval = 4,
  auxdeps = [ [0,1], [5,1,1] ],
  operatortypes = [
    { name = "unlimited", latency = 1 },
    { name = "limited", latency = 1, limit = 2 }
  ] } {
  %0 = arith.constant { opr = "unlimited", problemStartTime = 0 } 42 : i32
  %1 = "dummy.phi"() { opr = "unlimited", problemStartTime = 1 } : () -> i32
  %2 = "dummy.op"(%1) { opr = "limited", problemStartTime = 2 } : (i32) -> i32
  %3 = "dummy.op"(%1) { opr = "limited", problemStartTime = 3 } : (i32) -> i32
  %4 = "dummy.op"(%1) { opr = "limited", problemStartTime = 2 } : (i32) -> i32
  %5 = "dummy.mux"(%2, %3, %4) { opr = "unlimited", problemStartTime = 4 } : (i32, i32, i32) -> i32
  // SIMPLEX: return
  // SIMPLEX-SAME: simplexStartTime = 5
  return { opr = "unlimited", problemStartTime = 5 } %5 : i32
}

func.func @four_read_pipeline() -> i32 attributes {
  problemInitiationInterval = 4,
  auxdeps = [ [0,1] ],
  operatortypes = [
    { name = "unlimited", latency = 1 },
    { name = "limited", latency = 1, limit = 1 }
  ] } {
  %0 = arith.constant { opr = "unlimited", problemStartTime = 0 } 42 : i32
  %1 = "dummy.phi"() { opr = "unlimited", problemStartTime = 1 } : () -> i32
  %2 = "dummy.op"(%1) { opr = "limited", problemStartTime = 2 } : (i32) -> i32
  %3 = "dummy.op"(%1) { opr = "limited", problemStartTime = 3 } : (i32) -> i32
  %4 = "dummy.op"(%1) { opr = "limited", problemStartTime = 4 } : (i32) -> i32
  %5 = "dummy.op"(%1) { opr = "limited", problemStartTime = 5 } : (i32) -> i32
  %6 = "dummy.mux"(%2, %3) { opr = "unlimited", problemStartTime = 4 } : (i32, i32) -> i32
  %7 = "dummy.mux"(%4, %5) { opr = "unlimited", problemStartTime = 6 } : (i32, i32) -> i32
  %8 = "dummy.mux"(%6, %7) { opr = "unlimited", problemStartTime = 7 } : (i32, i32) -> i32
  // SIMPLEX: return
  // SIMPLEX-SAME: simplexStartTime = 8
  return { opr = "unlimited", problemStartTime = 8 } %8 : i32
}
