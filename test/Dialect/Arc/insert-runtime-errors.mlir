// RUN: circt-opt %s --arc-insert-runtime --verify-diagnostics

hw.module @hwMod(in %foo: i1) {}

// expected-error @+1 {{does not refer to a known Arc model.}}
arc.sim.instantiate @hwMod as %model {}
