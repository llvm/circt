// REQUIRES: rtl-sim
// RUN: circt-opt %s -create-dataflow -canonicalize-dataflow -canonicalize -cse > %t0.mlir
// RUN: circt-opt %t0.mlir -lower-handshake-to-firrtl -canonicalize -cse > %t1.mlir
// RUN: circt-opt %t1.mlir -pass-pipeline='firrtl.circuit(firrtl.module(firrtl-lower-types))' > %t2.mlir
// RUN: circt-translate %t2.mlir -emit-firrtl-verilog > %t3.sv
// RUN: circt-rtl-sim.py %t3.sv %S/basic.cpp --no-default-driver

func @top(%arg0: i32, %arg1: i32) -> i32 {
  %0 = addi %arg0, %arg1 : i32
  return %0 : i32
}
