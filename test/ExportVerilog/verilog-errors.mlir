// RUN: circt-translate -export-verilog -verify-diagnostics --split-input-file -mlir-print-op-on-diagnostic=false %s

// expected-error @+1 {{value has an unsupported verilog type 'f32'}}
hw.module @Top(%out: f32) {
}

// -----

// expected-error @+1 {{'hw.module' op name "parameter" is not allowed in Verilog output}}
hw.module @parameter() {
}

// -----

// expected-error @+2 {{unknown style option 'badOption'}}
// expected-error @+1 {{unknown style option 'anotherOne'}}
module attributes {circt.loweringOptions = "badOption,anotherOne"} {}

// -----

// expected-error @+2 {{name 'casex' changed during emission}}
// expected-error @+1 {{name 'if' changed during emission}}
hw.module @namechange(%casex: i4) -> (if: i4) {
  hw.output %casex : i4
}

// -----

hw.module.extern @A<width: none> ()

hw.module @B() {
  // expected-error @+1 {{op invalid parameter value @Foo}}
  hw.instance "foo" @A<width: none = @Foo>() -> ()
}
