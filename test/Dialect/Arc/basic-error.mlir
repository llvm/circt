// RUN: circt-opt %s --split-input-file --verify-diagnostics

arc.define @lut () -> () {
  // expected-error @+1 {{requires one result}}
  arc.lut () : () -> () {
    arc.output
  }
  arc.output
}

// -----

arc.define @lut (%arg0: i32, %arg1: i8) -> () {
  // expected-note @+1 {{required by region isolation constraints}}
  %1 = arc.lut (%arg1, %arg0) : (i8, i32) -> i32 {
    ^bb0(%arg2: i8, %arg3: i32):
      // expected-error @+1 {{using value defined outside the region}}
      arc.output %arg0 : i32
  }
  arc.output
}

// -----

arc.define @lutSideEffects () -> i32 {
  // expected-error @+1 {{no operations with side-effects allowed inside a LUT}}
  %0 = arc.lut () : () -> i32 {
    %true = hw.constant true
    // expected-note @+1 {{first operation with side-effects here}}
    %1 = arc.memory !arc.memory<20 x i32>
    %2 = arc.memory_read %1[%true], %true, %true : !arc.memory<20 x i32>, i1
    arc.output %2 : i32
  }
  arc.output %0 : i32
}
