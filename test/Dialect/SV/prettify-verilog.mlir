// RUN: circt-opt -prettify-verilog %s | FileCheck %s
// RUN: circt-opt -prettify-verilog %s | circt-translate  --export-verilog | FileCheck %s --check-prefix=VERILOG

// CHECK-LABEL: hw.module @unary_ops
hw.module @unary_ops(%arg0: i8, %arg1: i8, %arg2: i8) -> (%a: i8, %b: i8) {
  %c-1_i8 = hw.constant -1 : i8

  // CHECK: [[XOR1:%.+]] = comb.xor %arg0
  %unary = comb.xor %arg0, %c-1_i8 : i8
  // CHECK: comb.add [[XOR1]], %arg1
  %a = comb.add %unary, %arg1 : i8

  // CHECK: [[XOR2:%.+]] = comb.xor %arg0
  // CHECK: comb.add [[XOR2]], %arg2
  %b = comb.add %unary, %arg2 : i8
  hw.output %a, %b : i8, i8
}

// VERILOG: assign a = ~arg0 + arg1;
// VERILOG: assign b = ~arg0 + arg2;