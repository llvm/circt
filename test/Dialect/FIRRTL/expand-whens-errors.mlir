// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl.module(firrtl-expand-whens))' -verify-diagnostics --split-input-file %s

firrtl.circuit "simple" {

firrtl.module @simple(in %in : !firrtl.uint<1>, out %out : !firrtl.uint<1>) {
    firrtl.connect %out, %in : !firrtl.uint<1>, !firrtl.uint<1>
}

// This test is checking each kind of declaration to ensure that it is caught
// by the initialization coverage check. This is also testing that we can emit
// all errors in a module at once.
firrtl.module @CheckInitialization(in %clock : !firrtl.clock, in %en : !firrtl.uint<1>, in %p : !firrtl.uint<1>, out %out : !firrtl.uint<2>) {
  // expected-error @-1 {{module port "out" not fully initialized}}
  // expected-error @+1 {{sink not fully initialized}}
  %w = firrtl.wire : !firrtl.uint<2>
  // expected-error @+1 {{instance port "in" not fully initialized}}
  %simple_out, %simple_in = firrtl.instance @simple {name = "test", portNames=["in", "out"]}: !firrtl.flip<uint<1>>, !firrtl.uint<1>
}
}

// -----

firrtl.circuit "declaration_in_when" {
// Check that wires declared inside of a when are detected as uninitialized.
firrtl.module @declaration_in_when(in %p : !firrtl.uint<1>) {
  firrtl.when %p {
    // expected-error @+1 {{sink not fully initialized}}
    %w_then = firrtl.wire : !firrtl.uint<2>
  } else {
    // expected-error @+1 {{sink not fully initialized}}
    %w_else = firrtl.wire : !firrtl.uint<2>
  }
}
}

// -----

firrtl.circuit "complex" {
// Test that a wire set across separate when statements is detected as not
// completely initialized.
firrtl.module @complex(in %p : !firrtl.uint<1>, in %q : !firrtl.uint<1>) {
  // expected-error @+1 {{sink not fully initialized}}
  %w = firrtl.wire : !firrtl.uint<2>

  firrtl.when %p {
    %c1_ui2 = firrtl.constant(1 : ui2) : !firrtl.uint<2>
    firrtl.connect %w, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }

  firrtl.when %q {
  } else {
    %c1_ui2 = firrtl.constant(1 : ui2) : !firrtl.uint<2>
    firrtl.connect %w, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
}

}
