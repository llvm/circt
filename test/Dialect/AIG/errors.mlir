// RUN: circt-opt %s -split-input-file -verify-diagnostics

hw.module @InputNum(in %a: i1, in %b: i1) {
  // expected-error @+1 {{the number of inputs and the number of block arguments do not match. Expected 2 but got 1}}
  %0 = aig.cut %a, %b : (i1, i1) -> (i1) {
  ^bb0(%arg0: i1):
    aig.output %arg0 : i1
  }
}

// -----

hw.module @InputType(in %a: i1, in %b: i1) {
  // expected-error @+1 {{'aig.cut' op input type 'i1' does not match block argument type 'i2'}}
  %0 = aig.cut %a, %b : (i1, i1) -> (i1) {
  ^bb0(%arg0: i2, %arg1: i1):
    aig.output %arg1 : i1
  }
}

// -----

hw.module @OutputNum(in %a: i1, in %b: i1) {
  // expected-error @+1 {{the number of results and the number of terminator operands do not match. Expected 1 but got 2}}
  %0 = aig.cut %a, %b : (i1, i1) -> (i1) {
  ^bb0(%arg0: i1, %arg1: i1):
    aig.output %arg0, %arg1 : i1, i1
  }
}

// -----

hw.module @OutputType(in %a: i2, in %b: i1) {
  // expected-error @+1 {{'aig.cut' op result type 'i1' does not match terminator operand type 'i2'}}
  %0 = aig.cut %a, %b : (i2, i1) -> (i1) {
  ^bb0(%arg0: i2, %arg1: i1):
    aig.output %arg0 : i2
  }
}
