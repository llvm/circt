// RUN: circt-opt %s --split-input-file --verify-diagnostics


// expected-error @below {{argument types must match sequence type}}
rtg.sequence {
^bb0(%arg0: i32):
} -> !rtg.sequence

// -----

%0 = rtg.sequence {
^bb0(%arg0: i32):
} -> !rtg.sequence<i32>

rtg.sequence {
  %1 = arith.constant 0 : i32
  // expected-error @below {{sequences and ratios must match}}
  rtg.select_random [%0]((%1) : (i32)), [] : !rtg.sequence<i32>
} -> !rtg.sequence

// -----

%0 = rtg.sequence {
^bb0(%arg0: i32):
} -> !rtg.sequence<i32>

rtg.sequence {
  %1 = arith.constant 0 : i32
  // expected-error @below {{sequences and sequence arg lists must match}}
  rtg.select_random [%0](:), [%1] : !rtg.sequence<i32>
} -> !rtg.sequence

// -----

%0 = rtg.sequence {
^bb0(%arg0: i32):
} -> !rtg.sequence<i32>

rtg.sequence {
  %1 = arith.constant 0 : i64
  %2 = arith.constant 0 : i32
  // expected-error @below {{sequence argument types do not match sequence requirements}}
  rtg.select_random [%0]((%1) : (i64)), [%2] : !rtg.sequence<i32>
} -> !rtg.sequence
