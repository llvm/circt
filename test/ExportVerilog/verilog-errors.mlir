// RUN: circt-translate -emit-verilog -verify-diagnostics --split-input-file -mlir-print-op-on-diagnostic=false %s

// expected-error @+1 {{value has an unsupported verilog type 'f32'}}
rtl.module @Top(%out: f32) {
}

// -----

// expected-error @+1 {{value has an unsupported verilog type 'i0'}}
rtl.module @Top(%out: i0) {
}
