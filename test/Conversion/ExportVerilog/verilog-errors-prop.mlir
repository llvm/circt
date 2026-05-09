
// Make sure error is propagated to the pass failure when running `export-verilog`.
// RUN: not circt-opt -export-verilog %s

// Check diagnostics.
// RUN: circt-opt -export-verilog -verify-diagnostics %s

// Make sure error is propagated to the pass failure when running `export-split-verilog`.
// RUN: rm -rf %t && mkdir -p %t
// RUN: not circt-opt -export-split-verilog='dir-name=%t' %s

// Check diagnostics for split verilog.
// RUN: rm -rf %t && mkdir -p %t
// RUN: circt-opt -export-split-verilog='dir-name=%t' %s --verify-diagnostics

// expected-error @+1 {{value has an unsupported verilog type 'f32'}}
hw.module @Top(in %out: f32) {
}
