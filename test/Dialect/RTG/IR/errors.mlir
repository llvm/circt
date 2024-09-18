// RUN: circt-opt %s --split-input-file --verify-diagnostics


// expected-error @below {{region should have no arguments}}
rtg.snippet {
^bb0(%arg0: i32):
}
