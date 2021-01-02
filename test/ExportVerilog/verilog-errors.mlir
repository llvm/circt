// RUN: circt-translate -emit-firrtl-verilog -verify-diagnostics --split-input-file -mlir-print-op-on-diagnostic=false %s

firrtl.circuit "Top" {
  // expected-error @+1 {{value has an unsupported verilog type '!firrtl.uint'}}
  firrtl.module @Top(%out: !firrtl.uint) {
  }
}
