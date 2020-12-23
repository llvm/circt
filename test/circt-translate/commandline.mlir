// RUN: circt-translate --help | FileCheck %s

// TODO: this should be CIRCT, not MLIR
// CHECK: OVERVIEW: MLIR translation driver

// CHECK: Translation to perform
// CEHCK:     --emit-verilog
// CHECK:     --llhd-to-verilog
// CHECK:     --parse-fir
