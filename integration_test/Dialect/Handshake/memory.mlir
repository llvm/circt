// REQUIRES: ieee-sim
// RUN: circt-opt %s --create-dataflow --simple-canonicalizer --cse --handshake-insert-buffer > %memory-handshake.mlir
// RUN: circt-opt %memory-handshake.mlir --lower-handshake-to-firrtl --firrtl-lower-types --firrtl-imconstprop --lower-firrtl-to-rtl --rtl-memory-sim --rtl-cleanup --simple-canonicalizer --cse --rtl-legalize-names > %memory-hw.mlir
// RUN: circt-translate %memory-hw.mlir --export-verilog > %memory-export.sv
// RUN: circt-rtl-sim.py %memory-export.sv %S/driver.sv --sim %ieee-sim --no-default-driver --top driver | FileCheck %s
// CHECK: Result={{.*}}34

module  {
  func @top() -> i32 {
    %c34_i32 = constant 34 : i32
    %c1 = constant 0 : index
    %0 = memref.alloc() : memref<1xi32>
    memref.store %c34_i32, %0[%c1] : memref<1xi32>
    %1 = memref.load %0[%c1] : memref<1xi32>
    return %1 : i32
  }
}
