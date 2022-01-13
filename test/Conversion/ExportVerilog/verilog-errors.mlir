// RUN: circt-opt -export-verilog -verify-diagnostics --split-input-file -mlir-print-op-on-diagnostic=false %s

// expected-error @+1 {{value has an unsupported verilog type 'f32'}}
hw.module @Top(%out: f32) {
}

// -----

// expected-error @+2 {{unknown style option 'badOption'}}
// expected-error @+1 {{unknown style option 'anotherOne'}}
module attributes {circt.loweringOptions = "badOption,anotherOne"} {}

// -----

hw.module.extern @A<width: none> ()

hw.module @B() {
  // expected-error @+1 {{op invalid parameter value @Foo}}
  hw.instance "foo" @A<width: none = @Foo>() -> ()
}

// -----

// expected-error @+1 {{name "parameter" is not allowed in Verilog output}}
hw.module.extern @parameter ()

// -----
hw.module @invalid_probe_verbatim(%a: i1) {
  // expected-error @+2 {{must have exactly one operand to use in verbatim substitution}}
  // expected-error @+1 {{cannot get name for symbol #hw.innerNameRef<@invalid_probe_verbatim::@probe>}}
  hw.probe @probe, %a, %a : i1, i1
}

hw.module @invalid() -> (a: i1) {
  %0 = sv.verbatim.expr "{{0}}" : () -> i1
       {symbols = [#hw.innerNameRef<@invalid_probe_verbatim::@probe>]}
  hw.output %0 : i1
}