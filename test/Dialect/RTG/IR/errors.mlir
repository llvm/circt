// RUN: circt-opt %s --split-input-file --verify-diagnostics


func.func @seq0() {
  return
}

// expected-error @below {{'seq0' does not reference a valid 'rtg.sequence' operation}}
rtg.sequence_closure @seq0

// -----

rtg.sequence @seq0 {
^bb0(%arg0: i32):
}

// expected-error @below {{referenced 'rtg.sequence' op's argument types must match 'args' types}}
rtg.sequence_closure @seq0
