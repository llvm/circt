// RUN: circt-opt --arc-lower-verif-simulations --verify-diagnostics --split-input-file %s

// expected-error @below {{op expected to be a `func.func`}}
hw.module @exit() {}

// -----

// expected-error @below {{op expected to have function type '(i32) -> ()', got '(i42) -> i9001' instead}}
func.func private @exit(%arg0: i42) -> (i9001)
