// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(any(firrtl-expand-whens)))' -verify-diagnostics --split-input-file %s

// This test is checking each kind of declaration to ensure that it is caught
// by the initialization coverage check. This is also testing that we can emit
// all errors in a module at once.
firrtl.circuit "CheckInitialization" {
firrtl.module @CheckInitialization(in %clock : !firrtl.clock, in %en : !firrtl.uint<1>, in %p : !firrtl.uint<1>, in %in0 : !firrtl.bundle<a  flip: uint<1>>, out %out0 : !firrtl.uint<2>, out %out1 : !firrtl.bundle<a flip: uint<1>>) {
  // expected-error @above {{port "in0.a" not fully initialized in "CheckInitialization"}}
  // expected-error @above {{port "out0" not fully initialized in "CheckInitialization"}}
}
}

// -----

firrtl.circuit "CheckInitialization" {
firrtl.module @CheckInitialization() {
  // expected-error @below {{sink "w.a" not fully initialized in "CheckInitialization"}}
  // expected-error @below {{sink "w.b" not fully initialized in "CheckInitialization"}}
  %w = firrtl.wire : !firrtl.bundle<a : uint<1>, b  flip: uint<1>>
}
}

// -----

firrtl.circuit "CheckInitialization" {
firrtl.module @simple(in %in : !firrtl.uint<1>, out %out : !firrtl.uint<1>) {
    firrtl.connect %out, %in : !firrtl.uint<1>, !firrtl.uint<1>
}
firrtl.module @CheckInitialization() {
  // expected-error @below {{sink "test.in" not fully initialized}}
  %simple_out, %simple_in = firrtl.instance test @simple(in in : !firrtl.uint<1>, out out : !firrtl.uint<1>)
}
}

// -----

firrtl.circuit "CheckInitialization" {
firrtl.module @CheckInitialization() {
  // expected-error @below {{sink "memory.r.addr" not fully initialized}}
  // expected-error @below {{sink "memory.r.en" not fully initialized}}
  // expected-error @below {{sink "memory.r.clk" not fully initialized}}
  %memory_r = firrtl.mem Undefined {depth = 16 : i64, name = "memory", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<a: uint<8>, b: uint<8>>>
}
}

// -----

firrtl.circuit "declaration_in_when" {
// Check that wires declared inside of a when are detected as uninitialized.
firrtl.module @declaration_in_when(in %p : !firrtl.uint<1>) {
  firrtl.when %p : !firrtl.uint<1> {
    // expected-error @below {{sink "w_then" not fully initialized}}
    %w_then = firrtl.wire : !firrtl.uint<2>
  }
}
}

// -----

firrtl.circuit "declaration_in_when" {
// Check that wires declared inside of a when are detected as uninitialized.
firrtl.module @declaration_in_when(in %p : !firrtl.uint<1>) {
  firrtl.when %p : !firrtl.uint<1> {
  } else {
    // expected-error @below {{sink "w_else" not fully initialized}}
    %w_else = firrtl.wire : !firrtl.uint<2>
  }
}
}

// -----

firrtl.circuit "complex" {
// Test that a wire set across separate when statements is detected as not
// completely initialized.
firrtl.module @complex(in %p : !firrtl.uint<1>, in %q : !firrtl.uint<1>) {
  // expected-error @below {{sink "w" not fully initialized}}
  %w = firrtl.wire : !firrtl.uint<2>

  firrtl.when %p : !firrtl.uint<1> {
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    firrtl.connect %w, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }

  firrtl.when %q : !firrtl.uint<1> {
  } else {
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    firrtl.connect %w, %c1_ui2 : !firrtl.uint<2>, !firrtl.uint<2>
  }
}

}

// -----

firrtl.circuit "CheckInitialization" {
firrtl.module @CheckInitialization(out %out : !firrtl.vector<uint<1>, 1>) {
  // expected-error @above {{port "out[0]" not fully initialized in "CheckInitialization"}}
}
}

// -----

firrtl.circuit "CheckInitialization" {
firrtl.module @CheckInitialization() {
  // expected-error @below {{sink "w[0]" not fully initialized}}
  // expected-error @below {{sink "w[1]" not fully initialized}}
  %w = firrtl.wire : !firrtl.vector<uint<1>, 2>
}
}

// -----

firrtl.circuit "CheckInitialization" {
firrtl.module @CheckInitialization(in %in : !firrtl.uint<1>, out %out : !firrtl.vector<uint<1>, 2>) {
  // expected-error @above {{port "out[1]" not fully initialized in "CheckInitialization"}}
  %0 = firrtl.subindex %out[0] : !firrtl.vector<uint<1>, 2>
  firrtl.connect %0, %in : !firrtl.uint<1>, !firrtl.uint<1>
}
}

// -----

firrtl.circuit "CheckInitialization" {
firrtl.module @CheckInitialization(in %in : !firrtl.uint<1>, out %out : !firrtl.vector<vector<uint<1>, 1>, 1>) {
  // expected-error @above {{port "out[0][0]" not fully initialized in "CheckInitialization"}}
}
}

// -----

firrtl.circuit "CheckInitialization" {
firrtl.module @CheckInitialization(in %p : !firrtl.uint<1>, out %out: !firrtl.vector<bundle<a:uint<1>, b:uint<1>>, 1>) {
  // expected-error @above {{port "out[0].a" not fully initialized in "CheckInitialization"}}
  // expected-error @above {{port "out[0].b" not fully initialized in "CheckInitialization"}}
}
}

// -----

// Check initialization error is produced for out-references
firrtl.circuit "RefInitOut" {
firrtl.module @RefInitOut(out %out : !firrtl.probe<uint<1>>) {
  // expected-error @above {{port "out" not fully initialized in "RefInitOut"}}
}
}

// -----

// Check initialization error is produced for output property ports on modules.
firrtl.circuit "PropInitOut" {
firrtl.module @PropInitOut(out %out : !firrtl.string) {
  // expected-error @above {{port "out" not fully initialized in "PropInitOut"}}
}
}

// -----

// Check initialization error is produced for output property ports on classes.
firrtl.circuit "PropInitOut" {
  // expected-error @below {{port "out" not fully initialized in "Class"}}
  firrtl.class @Class(out %out: !firrtl.string) {}
  firrtl.module @PropInitOut() {}
}

// -----

// Check initialization error is produced for input property ports on instances.
firrtl.circuit "PropInitIn" {
firrtl.module @Child(in %in: !firrtl.string) {}
firrtl.module @PropInitIn() {
  %child_in = firrtl.instance child @Child(in in : !firrtl.string)
  // expected-error @above {{sink "child.in" not fully initialized in "PropInitIn"}}
}
}

// -----

// Check initialization error is produced for input property ports on local objects.
firrtl.circuit "PropInitIn" {
firrtl.class @Class(in %in: !firrtl.string) {}
firrtl.module @PropInitIn() {
  // expected-error @below {{sink "obj.in" not fully initialized in "PropInitIn"}}
  %obj = firrtl.object @Class(in in : !firrtl.string)
}
}

// -----

// Check initialization error is produced for output object ports on modules.
firrtl.circuit "Test" {
firrtl.class @Class() {}
// expected-error @below {{port "out" not fully initialized in "Test"}}
firrtl.module @Test(out %out: !firrtl.class<@Class()>) {}
}

// -----

// Check initialization error is produced for output object ports on classes.
firrtl.circuit "Test" {
firrtl.class @Class1() {}
// expected-error @below {{port "out" not fully initialized in "Class2"}}
firrtl.class @Class2(out %out: !firrtl.class<@Class1()>) {}
firrtl.module @Test() {}
}

// -----

// Check initialization error is produced for input object ports on instances.
firrtl.circuit "Test" {
firrtl.class @Class() {}
firrtl.module @Module(in %in: !firrtl.class<@Class()>) {}
firrtl.module @Test() {
  // expected-error @below {{sink "mod.in" not fully initialized in "Test"}}
  %mod_in = firrtl.instance mod @Module(in in : !firrtl.class<@Class()>)
}
}

// -----

// Check initialization error is produced for input object ports on local objects.
firrtl.circuit "Test" {
firrtl.class @Class1() { }
firrtl.class @Class2(in %in: !firrtl.class<@Class1()>) {}
firrtl.module @Test() {
  // expected-error @below {{sink "obj.in" not fully initialized in "Test"}}
  %obj = firrtl.object @Class2(in in : !firrtl.class<@Class1()>)
}
}

// -----

// Check initialization errors for local objects are produced under firrtl classes.
firrtl.circuit "Test" {
firrtl.class @Class1(in %in: !firrtl.string) {}
firrtl.class @Class2() {
  // expected-error @below {{sink "obj.in" not fully initialized in "Class2"}}
  %obj = firrtl.object @Class1(in in: !firrtl.string)
}
firrtl.module @Test() {}
}

// -----

// Check initialization errors for objects in wires are produced.
firrtl.circuit "Test" {
firrtl.class @Class(in %in: !firrtl.string) {}
firrtl.module @Test() {
  // expected-error @below {{sink "w" not fully initialized in "Test"}}
  %w = firrtl.wire : !firrtl.class<@Class()>
}
}
