// RUN: spt-translate -emit-verilog -verify-diagnostics --split-input-file -mlir-print-op-on-diagnostic=false %s

func @foo() { // expected-error {{unknown operation}}
}

// -----

firrtl.circuit "Top" {
  firrtl.module @Top(%out: !firrtl.uint) {
  }
}
