// RUN: spt-opt %s -split-input-file -verify-diagnostics

firrtl.module @X() {
  // expected-error @+1 {{unknown firrtl type}}
  %0 = firrtl.invalid : !firrtl.unknowntype
}

// -----

firrtl.module @X(%b : !firrtl.uint<32>, %d : !firrtl.uint<16>, %out : !firrtl.uint { firrtl.output }) {
  // expected-error @+1 {{'firrtl.add' op expected 2 operands, but found 3}}
  %3 = "firrtl.add"(%b, %d, %out) : (!firrtl.uint<32>, !firrtl.uint<16>, !firrtl.uint) -> !firrtl.uint<32>
}

// -----

// expected-error @+2 {{'firrtl.module' op expects regions to end with 'firrtl.done'}}
// expected-note @+1 {{implies 'firrtl.done'}}
"firrtl.module"() ( {
  %0 = firrtl.invalid : !firrtl.sint<32>

}) {sym_name = "MyModule", type = () -> ()} : () -> ()

// -----


// expected-error @+1 {{'firrtl.module' op requires string attribute 'sym_name'}}
"firrtl.module"() ( {
  "firrtl.done"() : () -> ()
}) { type = () -> ()} : () -> ()
