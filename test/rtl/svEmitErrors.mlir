// RUN: circt-translate %s -emit-verilog -verify-diagnostics --split-input-file

firrtl.circuit "A" {
  // expected-error @+2 {{'rtl.module' op Found port without a name. Port names are required for Verilog synthesis.}}
  // expected-note @+1 {{see current operation: "rtl.module"()}}
  rtl.module @A() -> (i1) {
    // expected-error @+2 {{'std.constant' op cannot emit this operation to Verilog}}
    // expected-note @+1 {{see current operation: %false = "std.constant"()}}
    %0 = constant 0 : i1
    rtl.output %0 : i1
  }
}

// -----

firrtl.circuit "A" {
  // expected-error @+2 {{value has an unsupported verilog type 'vector<3xi1>'}}
  // expected-note @+1 {{see current operation: "rtl.module"()}}
  rtl.module @A(%a: vector<3 x i1>) -> () { }
}
