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
  %0 = firrtl.add %a, %a : (!firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<33>

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


// expected-error @+1 {{'firrtl.module' op should be embedded into a 'firrtl.circuit'}}
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


// -----

firrtl.circuit "Foo" {
  firrtl.module @Foo(%clk: !firrtl.clock) {
    // expected-error @+1 {{'firrtl.reg' op result #0 must be a passive type (contain no flips)}}
    %a = firrtl.reg %clk {name = "a"} : (!firrtl.clock) -> !firrtl.flip<uint<1>>
  }
}

// -----

firrtl.circuit "Foo" {
  firrtl.module @Foo(%clk: !firrtl.clock, %reset: !firrtl.uint<1>) {
    %zero = firrtl.constant(0 : ui1) : !firrtl.uint<1>
    // expected-error @+1 {{'firrtl.reginit' op result #0 must be a passive type (contain no flips)}}
    %a = firrtl.reginit %clk, %reset, %zero {name = "a"} : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.flip<uint<1>>
  }
}

// -----

firrtl.circuit "Foo" {

  // expected-error @+1 {{'firrtl.extmodule' op attribute 'defname' with value "Bar" conflicts with the name of another module in the circuit}}
  firrtl.extmodule @Foo() attributes { defname = "Bar" }
  // expected-note @+1 {{previous module declared here}}
  firrtl.module @Bar() {}
  // Allow an extmodule to conflict with its own symbol name
  firrtl.extmodule @Baz() attributes { defname = "Baz" }

}

// -----

firrtl.circuit "Foo" {

  // expected-note @+1 {{previous extmodule definition occurred here}}
  firrtl.extmodule @Foo(%a : !firrtl.uint<1>) attributes { defname = "Foo" }
  // expected-error @+1 {{'firrtl.extmodule' op with 'defname' attribute "Foo" has 0 ports which is different from a previously defined extmodule with the same 'defname' which has 1 ports}}
  firrtl.extmodule @Bar() attributes { defname = "Foo" }

}

// -----

firrtl.circuit "Foo" {

  // expected-note @+1 {{previous extmodule definition occurred here}}
  firrtl.extmodule @Foo(%a : !firrtl.uint<1>) attributes { defname = "Foo" }
  // expected-error @+1 {{'firrtl.extmodule' op with 'defname' attribute "Foo" has a port with name "b" which does not match the name of the port in the same position of a previously defined extmodule with the same 'defname', expected port to have name "a"}}
  firrtl.extmodule @Foo_(%b : !firrtl.uint<1>) attributes { defname = "Foo" }

}

// -----

firrtl.circuit "Foo" {

  firrtl.extmodule @Foo(%a : !firrtl.uint<2>) attributes { defname = "Foo", parameters = { width = 2 : i32 } }
  // expected-note @+1 {{previous extmodule definition occurred here}}
  firrtl.extmodule @Bar(%a : !firrtl.uint<1>) attributes { defname = "Foo" }
  // expected-error @+1 {{'firrtl.extmodule' op with 'defname' attribute "Foo" has a port with name "a" which has a different type '!firrtl.uint<2>' which does not match the type of the port in the same position of a previously defined extmodule with the same 'defname', expected port to have type '!firrtl.uint<1>'}}
  firrtl.extmodule @Baz(%a : !firrtl.uint<2>) attributes { defname = "Foo" }

}

// -----

firrtl.circuit "Foo" {

  // expected-note @+1 {{previous extmodule definition occurred here}}
  firrtl.extmodule @Foo(%a : !firrtl.uint<1>) attributes { defname = "Foo" }
  // expected-error @+1 {{'firrtl.extmodule' op with 'defname' attribute "Foo" has a port with name "a" which has a different type '!firrtl.sint<1>' which does not match the type of the port in the same position of a previously defined extmodule with the same 'defname', expected port to have type '!firrtl.uint<1>'}}
  firrtl.extmodule @Foo_(%a : !firrtl.sint<1>) attributes { defname = "Foo" }

}

// -----

firrtl.circuit "Foo" {

  // expected-note @+1 {{previous extmodule definition occurred here}}
  firrtl.extmodule @Foo(%a : !firrtl.uint<2>) attributes { defname = "Foo", parameters = { width = 2 : i32 } }
  // expected-error @+1 {{'firrtl.extmodule' op with 'defname' attribute "Foo" has a port with name "a" which has a different type '!firrtl.sint' which does not match the type of the port in the same position of a previously defined extmodule with the same 'defname', expected port to have type '!firrtl.uint'}}
  firrtl.extmodule @Bar(%a : !firrtl.sint<1>) attributes { defname = "Foo" }

}

// -----

firrtl.circuit "Foo" {

  // expected-error @+1 {{has unknown extmodule parameter value 'width' = @Foo}}
  firrtl.extmodule @Foo(%a : !firrtl.uint<2>) attributes { defname = "Foo", parameters = { width = @Foo } }

}

// -----

firrtl.circuit "Foo" {

  firrtl.extmodule @Foo()
  // expected-error @+1 {{'firrtl.instance' op should be embedded in a 'firrtl.module'}}
  %a = firrtl.instance @Foo : !firrtl.bundle<>

}

// -----

firrtl.circuit "Foo" {

  // expected-note @+1 {{containing module declared here}}
  firrtl.module @Foo() {
    // expected-error @+1 {{'firrtl.instance' op is a recursive instantiation of its containing module}}
    %a = firrtl.instance @Foo : !firrtl.bundle<>
  }

}

// -----

firrtl.circuit "Foo" {

  // expected-note @+1 {{original module declared here}}
  firrtl.module @Callee(%arg0: !firrtl.uint<1>) { }
  firrtl.module @Foo() {
    // expected-error @+1 {{'firrtl.instance' op output bundle type must match module. In element 0, expected '!firrtl.uint<1>', but got '!firrtl.uint<2>'.}}
    %a = firrtl.instance @Callee : !firrtl.bundle<arg0: uint<2>>
  }
}

// -----

firrtl.circuit "Foo" {

  // expected-note @+1 {{original module declared here}}
  firrtl.module @Callee(%arg0: !firrtl.uint<1> ) { }
  firrtl.module @Foo() {
    // expected-error @+1 {{'firrtl.instance' op has a wrong size of bundle type, expected size is 1 but got 0}}
    %a = firrtl.instance @Callee : !firrtl.bundle<>
  }
}

// -----

firrtl.circuit "Foo" {

  // expected-note @+1 {{original module declared here}}
  firrtl.module @Callee(%arg0: !firrtl.uint<1>, %arg1: !firrtl.bundle<valid: uint<1>>) { }
  firrtl.module @Foo() {
    // expected-error @+1 {{'firrtl.instance' op output bundle type must match module. In element 1, expected '!firrtl.bundle<valid: uint<1>>', but got '!firrtl.bundle<valid: uint<2>>'.}}
    %a = firrtl.instance @Callee : !firrtl.bundle<arg0: uint<1>, arg1: bundle<valid: uint<2>>>
  }
}

// ----- 

firrtl.circuit "X" {

firrtl.module @X(%a : !firrtl.uint<4>) {
  // expected-error @+1 {{high must be equal or greater than low, but got high = 3, low = 4}}
  %0 = firrtl.bits %a 3 to 4 : (!firrtl.uint<4>) -> !firrtl.uint<2>
}

}

// -----

firrtl.circuit "X" {

firrtl.module @X(%a : !firrtl.uint<4>) {
  // expected-error @+1 {{high must be smaller than the width of input, but got high = 4, width = 4}}
  %0 = firrtl.bits %a 4 to 3 : (!firrtl.uint<4>) -> !firrtl.uint<2>
}

}

// -----

firrtl.circuit "X" {

firrtl.module @X(%a : !firrtl.uint<4>) {
  // expected-error @+1 {{'firrtl.bits' op result type should be '!firrtl.uint<3>'}}
  %0 = firrtl.bits %a 3 to 1 : (!firrtl.uint<4>) -> !firrtl.uint<2>
}

}

// -----

firrtl.circuit "TopModule" {

firrtl.module @SubModule(%a : !firrtl.uint<1>) {
}

firrtl.module @TopModule() {
  // expected-error @+1 {{'firrtl.instance' op has invalid result type of '!firrtl.uint<1>'}}
  %0 = firrtl.instance @SubModule : !firrtl.uint<1>
}

}

// -----

firrtl.circuit "TopModule" {

firrtl.module @SubModule(%a : !firrtl.uint<1>) {
}

firrtl.module @TopModule() {
  // expected-error @+1 {{'firrtl.instance' op has invalid result type of '!firrtl.flip<uint<1>>'}}
  %0 = firrtl.instance @SubModule : !firrtl.flip<uint<1>>
}

}

// -----

firrtl.circuit "BadPort" {
  // expected-error @+1 {{'firrtl.module' op all module ports must be firrtl types}}
  firrtl.module @BadPort(%in1 : i1) {
  }
}


// -----

firrtl.circuit "BadPort" {
  firrtl.module @BadPort(%a : !firrtl.uint<1>) {
    // expected-error @+1 {{'firrtl.attach' op operand #0 must be analog type, but got '!firrtl.uint<1>'}}
    firrtl.attach %a, %a : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "BadAdd" {
  firrtl.module @BadAdd(%a : !firrtl.uint<1>) {
    // expected-error @+1 {{'firrtl.add' op result type should be '!firrtl.uint<2>'}}
    firrtl.add %a, %a : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  }
}
