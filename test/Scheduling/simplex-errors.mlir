// RUN: circt-opt %s -test-simplex-scheduler=with=CyclicProblem -verify-diagnostics -split-input-file

// expected-error@+2 {{problem is infeasible}}
// expected-error@+1 {{scheduling failed}}
func @cyclic_graph() attributes {
  auxdeps = [ [0,1], [1,2], [2,3], [3,1] ]
  } {
  %0 = constant 0 : i32
  %1 = constant 1 : i32
  %2 = constant 2 : i32
  %3 = constant 3 : i32
  return
}
