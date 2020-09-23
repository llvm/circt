// RUN: circt-opt %s -split-input-file -verify-diagnostics

firrtl.circuit "X" {

firrtl.module @X(%b : !firrtl.unknowntype) {
  // expected-error @-1 {{unknown firrtl type}}
}

}

// -----

firrtl.circuit "X" {

firrtl.module @X(%b : !firrtl.uint<32>, %d : !firrtl.uint<16>, %out : !firrtl.uint) {
  // expected-error @+1 {{'firrtl.add' op expected 2 operands, but found 3}}
  %3 = "firrtl.add"(%b, %d, %out) : (!firrtl.uint<32>, !firrtl.uint<16>, !firrtl.uint) -> !firrtl.uint<32>
}

}

// -----

firrtl.circuit "MyModule" {

// expected-error @+2 {{'firrtl.module' op expects regions to end with 'firrtl.done'}}
// expected-note @+1 {{implies 'firrtl.done'}}
"firrtl.module"() ( {
^bb0(%a: !firrtl.uint<32>):
  firrtl.invalid %a : !firrtl.uint<32>

}) {sym_name = "MyModule", type = (!firrtl.uint<32>) -> ()} : () -> ()

}

// -----

// expected-error @+1 {{'firrtl.circuit' op must contain one module that matches main name 'MyCircuit'}}
firrtl.circuit "MyCircuit" {

"firrtl.module"() ( {
  "firrtl.done"() : () -> ()
}) { type = () -> ()} : () -> ()

}

// -----


// expected-error @+1 {{'firrtl.module' op should be embedded into a firrtl.circuit}}
firrtl.module @X() {}

// -----

// expected-error @+1 {{'firrtl.circuit' op must contain one module that matches main name 'Foo'}}
firrtl.circuit "Foo" {

firrtl.module @Bar() {}

}

// -----

// expected-error @+1 {{'firrtl.circuit' op must have a non-empty name}}
firrtl.circuit "" {
}

// -----

firrtl.circuit "Foo" {
  firrtl.module @Foo(%clk: !firrtl.uint<1>, %reset: !firrtl.uint<1>) {
    // expected-error @+1 {{'firrtl.reg' op operand #0 must be clock, but got '!firrtl.uint<1>'}}
    %a = firrtl.reg %clk {name = "a"} : (!firrtl.uint<1>) -> !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "Foo" {
  firrtl.module @Foo(%clk: !firrtl.uint<1>, %reset: !firrtl.uint<1>) {
    %zero = firrtl.constant(0 : ui1) : !firrtl.uint<1>
    // expected-error @+1 {{'firrtl.reginit' op operand #0 must be clock, but got '!firrtl.uint<1>'}}
    %a = firrtl.reginit %clk, %reset, %zero {name = "a"} : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "Foo" {
  firrtl.module @Foo(%clk: !firrtl.clock, %reset: !firrtl.uint<2>) {
    %zero = firrtl.constant(0 : ui1) : !firrtl.uint<1>
    // expected-error @+1 {{'firrtl.reginit' op operand #1 must be Reset, AsyncReset, or UInt<1>, but got '!firrtl.uint<2>'}}
    %a = firrtl.reginit %clk, %reset, %zero {name = "a"} : (!firrtl.clock, !firrtl.uint<2>, !firrtl.uint<1>) -> !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "Foo" {
  firrtl.module @Foo() {
  // expected-error @+1 {{'firrtl.mem' op attribute 'writeLatency' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 1}}
    %m = firrtl.mem "Undefined" {depth = 32 : i64, name = "m", readLatency = 0 : i32, writeLatency = 0 : i32} : !firrtl.bundle<>
  }
}
