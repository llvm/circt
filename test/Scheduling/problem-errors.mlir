// RUN: circt-opt -test-scheduling-problem -verify-diagnostics -split-input-file %s

// expected-error@+2 {{Operator type 'foo' has no latency}}
// expected-error@+1 {{problem check failed}}
func @no_latency() {
  %0 = constant 0 : i32
  %1 = constant { startTime = 0, opr = "foo" } 1 : i32
  return
}

// -----

// expected-error@+1 {{problem verification failed}}
func @no_starttime() {
  %0 = constant { startTime = 0 } 0 : i32
  %1 = constant 1 : i32 // expected-error {{Operation has no start time}}
  return { startTime = 0 }
}

// -----

// expected-error@+2 {{Precedence violated for dependence}}
// expected-error@+1 {{problem verification failed}}
func @ssa_dep_violated(%a0 : i32, %a1 : i32, %a2 : i32) -> i32 attributes {
  operatortypes = [
    { name = "_0", latency = 0 },
    { name = "_1", latency = 1 },
    { name = "_3", latency = 3 }
  ] } {
  %0 = addi %a0, %a0 {opr = "_3", startTime = 0} : i32
  %1 = addi %a1, %0 {opr = "_1", startTime = 3} : i32
  %2 = addi %1, %a2 {opr = "_0", startTime = 3} : i32
  return {opr = "_1", startTime = 4} %2 : i32
}

// -----

// expected-error@+2 {{Precedence violated for dependence}}
// expected-error@+1 {{problem verification failed}}
func @aux_dep_violated() attributes { auxdeps = [ [0,1], [1,2], [2,3] ] } {
  %0 = constant { startTime = 123 } 0 : i32
  %1 = constant { startTime = 456 } 1 : i32
  %2 = constant { startTime = 123 } 2 : i32
  return { startTime = 456 }
}
