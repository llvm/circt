// RUN: not circt-verilog %s -Xslang=--std=foo 2>&1 | FileCheck %s
// RUN: circt-verilog %s -Xslang=--std=latest

// CHECK: invalid value for --std option: 'foo'
