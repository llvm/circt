// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl.module(firrtl-expand-whens))' -verify-diagnostics --split-input-file %s

// This test is checking each kind of declaration to ensure that it is caught
// by the initialization coverage check. This is also testing that we can emit
// all errors in a module at once.
firrtl.circuit "CheckInitialization" {
firrtl.module @CheckInitialization(in %clock : !firrtl.clock, in %en : !firrtl.uint<1>, in %p : !firrtl.uint<1>, in %in0 : !firrtl.bundle<a  flip: uint<1>>, out %out0 : !firrtl.uint<2>, out %out1 : !firrtl.bundle<a flip: uint<1>>) {
  // expected-error @-1 {{sink "in0.a" not fully initialized}}
}
}

// -----

firrtl.circuit "CheckInitialization" {
firrtl.module @CheckInitialization() {
  // expected-error @+1 {{sink "w.a" not fully initialized}}
  %w = firrtl.wire : !firrtl.bundle<a : uint<1>, b  flip: uint<1>>
}
}

// -----

firrtl.circuit "CheckInitialization" {
firrtl.module @simple(in %in : !firrtl.uint<1>, out %out : !firrtl.uint<1>) {
    firrtl.connect %out, %in : !firrtl.uint<1>, !firrtl.uint<1>
}
firrtl.module @CheckInitialization() {
  // expected-error @+1 {{sink "test.in" not fully initialized}}
  %simple_out, %simple_in = firrtl.instance test @simple(in in : !firrtl.uint<1>, out out : !firrtl.uint<1>)
}
}

// -----

firrtl.circuit "CheckInitialization" {
firrtl.module @CheckInitialization() {
  // expected-error @+1 {{sink "memory.r.addr" not fully initialized}}
  %memory_r = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>
}
}

// -----

firrtl.circuit "declaration_in_when" {
// Check that wires declared inside of a when are detected as uninitialized.
firrtl.module @declaration_in_when(in %p : !firrtl.uint<1>) {
  firrtl.when %p {
    // expected-error @+1 {{sink "w_then" not fully initialized}}
    %w_then = firrtl.wire : !firrtl.uint<2>
  }
}
}

// -----

firrtl.circuit "declaration_in_when" {
// Check that wires declared inside of a when are detected as uninitialized.
firrtl.module @declaration_in_when(in %p : !firrtl.uint<1>) {
  firrtl.when %p {
  } else {
    // expected-error @+1 {{sink "w_else" not fully initialized}}
    %w_else = firrtl.wire : !firrtl.uint<2>
  }
}
}

// -----

firrtl.circuit "complex" {
// Test that a wire set across separate when statements is detected as not
// completely initialized.
firrtl.module @complex(in %p : !firrtl.uint<1>, in %q : !firrtl.uint<1>) {
  // expected-error @+1 {{sink "w" not fully initialized}}
  %w = firrtl.wire : !firrtl.uint<2>

  firrtl.when %p {
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    firrtl.connect %w, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }

  firrtl.when %q {
  } else {
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    firrtl.connect %w, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
}

}
