// RUN: circt-translate -emit-verilog -verify-diagnostics --split-input-file %s

// expected-error @+1 {{'rtl.module' op Found port without a name. Port names are required for Verilog synthesis.}}
rtl.module @A() -> (i1) {
  // expected-error @+1 {{'std.constant' op cannot emit this operation to Verilog}}
  %0 = constant 0 : i1
  rtl.output %0 : i1
}

// -----

// expected-error @+1 {{value has an unsupported verilog type 'vector<3xi1>'}}
rtl.module @A(%a: vector<3 x i1>) -> () { }
