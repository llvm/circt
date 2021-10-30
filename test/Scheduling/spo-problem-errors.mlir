// RUN: circt-opt %s -test-spo-problem -verify-diagnostics -split-input-file

// expected-error@+2 {{Limited operator type '_0' has zero latency}}
// expected-error@+1 {{problem check failed}}
func @limited_but_zero_latency() attributes {
  operatortypes = [ { name = "_0", latency = 0, limit = 1} ]
  } {
  return { opr = "_0" }
}

// -----

// expected-error@+2 {{Operator type 'limited' is oversubscribed}}
// expected-error@+1 {{problem verification failed}}
func @oversubscribed(%a0 : i32, %a1 : i32, %a2 : i32) -> i32 attributes {
  operatortypes = [ { name = "limited", latency = 1, limit = 2} ]
  } {
  %0 = arith.addi %a0, %a0 { problemStartTime = 0 } : i32
  %1 = arith.addi %a1, %0 { opr = "limited", problemStartTime = 1 } : i32
  %2 = arith.addi %0, %a2 { opr = "limited", problemStartTime = 1 } : i32
  %3 = arith.addi %0, %0 { opr = "limited", problemStartTime = 1 } : i32
  return { problemStartTime = 2 } %3 : i32
}
