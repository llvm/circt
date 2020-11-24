// RUN: circt-translate -emit-verilog -verify-diagnostics --split-input-file -mlir-print-op-on-diagnostic=false %s

func @foo() attributes {sym_visibility = "private"} { // expected-error {{unknown operation}}
}

// -----

firrtl.circuit "Top" {
  // expected-error @+1 {{value has an unsupported verilog type '!firrtl.uint'}}
  firrtl.module @Top(%out: !firrtl.uint) {
  }
}
