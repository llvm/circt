// REQUIRES: verilator
// RUN: circt-opt %s --cellift-instrument -export-verilog -o %t.mlir > %t.sv
// RUN: circt-rtl-sim.py %t.sv %S/cellift-runtime-driver.cpp --no-default-driver 2>&1 | FileCheck %s

hw.module @adder(in %a : i4, in %b : i4, out sum : i4) {
  %sum = comb.add %a, %b : i4
  hw.output %sum : i4
}

hw.module @top(in %a : i4, in %b : i4, out sum : i4) {
  %sum = hw.instance "dut" @adder(a: %a: i4, b: %b: i4) -> (sum: i4)
  hw.output %sum : i4
}

// CHECK: case0 sum=3 taint=0
// CHECK-NEXT: case1 sum=4 taint=7
// CHECK-NEXT: case2 sum=0 taint=15