// RUN: circt-opt %s -test-cyclic-problem -verify-diagnostics -split-input-file

// expected-error@+2 {{Invalid initiation interval}}
// expected-error@+1 {{problem verification failed}}
func @no_II() {
  return { problemStartTime = 0 }
}

// -----

// expected-error@+2 {{Precedence violated for dependence}}
// expected-error@+1 {{problem verification failed}}
func @backedge_violated(%a1 : i32, %a2 : i32) -> i32 attributes {
  problemInitiationInterval = 2,
  auxdeps = [ [2,0,1] ]
  } {
  %0 = addi %a1, %a2 { problemStartTime = 0 } : i32
  %1 = addi %0, %0 { problemStartTime = 1 } : i32
  %2 = addi %1, %1 { problemStartTime = 2 } : i32
  return { problemStartTime = 3 } %2 : i32
}
