// RUN: circt-verilog -h | FileCheck %s --check-prefix=CHECK-HELP
// RUN: circt-verilog --version | FileCheck %s --check-prefix=CHECK-VERSION
// REQUIRES: slang

// CHECK-HELP: OVERVIEW: Verilog and SystemVerilog frontend
// CHECK-VERSION: slang version 8.
