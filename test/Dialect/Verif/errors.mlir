// RUN: circt-opt %s --split-input-file --verify-diagnostics

// expected-error @below {{types of the yielded values of both regions must match}}
verif.lec first {
^bb0(%arg0: i32):
  verif.yield %arg0 : i32
} second {
^bb0(%arg0: i32):
  verif.yield
}

// -----

// expected-error @below {{block argument types of both regions must match}}
verif.lec first {
^bb0(%arg0: i32, %arg1: i32):
  verif.yield %arg0 : i32
} second {
^bb0(%arg0: i32):
  verif.yield %arg0 : i32
}
