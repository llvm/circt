// RUN: circt-opt %s -test-modulo-problem -allow-unregistered-dialect

func @canis14_fig2() attributes {
  problemInitiationInterval = 3,
  auxdeps = [ [2,0,1], [3,4] ],
  operatortypes = [
    { name = "mem_port", latency = 1, limit = 1 },
    { name = "add", latency = 1 }
  ] } {
  %0 = "dummy.load_A"() { opr = "mem_port", problemStartTime = 2 } : () -> i32
  %1 = "dummy.load_B"() { opr = "mem_port", problemStartTime = 0 } : () -> i32
  %2 = addi %0, %1 { opr = "add", problemStartTime = 3 } : i32
  "dummy.store_A"(%2) { opr = "mem_port", problemStartTime = 4 } : (i32) -> ()
  return { problemStartTime = 5 }
}

func @minII_feasible() attributes {
  problemInitiationInterval = 3,
  auxdeps = [ [6,1,5], [5,2,3], [6,7] ],
  operatortypes = [
    { name = "const", latency = 0 },
    { name = "phi", latency = 2 },
    { name = "xor", latency = 1 },
    { name = "sub", latency = 3, limit = 1}
  ] } {
  %0 = constant { opr = "const", problemStartTime = 0 } -1 : i32
  %1 = "dummy.phi"() { opr = "phi", problemStartTime = 0 } : () -> i32
  %2 = "dummy.phi"() { opr = "phi", problemStartTime = 1 } : () -> i32
  %3 = xor %1, %0 { opr = "xor", problemStartTime = 2 } : i32
  %4 = subi %3, %2 { opr = "sub", problemStartTime = 3 } : i32
  %5 = subi %0, %4 { opr = "sub", problemStartTime = 7 } : i32
  %6 = subi %4, %5 { opr = "sub", problemStartTime = 11 } : i32
  return { problemStartTime = 14 }
}

func @minII_infeasible() -> i32 attributes {
  problemInitiationInterval = 4,
  auxdeps = [ [0,1], [5,1,1] ],
  operatortypes = [
    { name = "unlimited", latency = 1 },
    { name = "limited", latency = 1, limit = 2 }
  ] } {
  %0 = constant { opr = "unlimited", problemStartTime = 0 } 42 : i32
  %1 = "dummy.phi"() { opr = "unlimited", problemStartTime = 1 } : () -> i32
  %2 = "dummy.op"(%1) { opr = "limited", problemStartTime = 2 } : (i32) -> i32
  %3 = "dummy.op"(%1) { opr = "limited", problemStartTime = 3 } : (i32) -> i32
  %4 = "dummy.op"(%1) { opr = "limited", problemStartTime = 2 } : (i32) -> i32
  %5 = "dummy.mux"(%2, %3, %4) { opr = "unlimited", problemStartTime = 4 } : (i32, i32, i32) -> i32
  return { opr = "unlimited", problemStartTime = 5 } %5 : i32
}
