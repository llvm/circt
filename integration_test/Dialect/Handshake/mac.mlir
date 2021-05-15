// REQUIRES: ieee-sim
// RUN: circt-opt %s --create-dataflow --simple-canonicalizer --cse --handshake-insert-buffer > %mac-handshake.mlir
// RUN: circt-opt %mac-handshake.mlir --lower-handshake-to-firrtl --firrtl-lower-types --firrtl-imconstprop --lower-firrtl-to-rtl --rtl-memory-sim --rtl-cleanup --simple-canonicalizer --cse --rtl-legalize-names > %mac-hw.mlir
// RUN: circt-translate %mac-hw.mlir --export-verilog > %mac-export.sv
// RUN: circt-rtl-sim.py %mac-export.sv %S/driver.sv --sim %ieee-sim --no-default-driver --top driver | FileCheck %s
// CHECK: Result={{.*}}912

module  {
  func @top() -> i32 {
    %c24_i32 = constant 24 : i32
    %c36_i32 = constant 36 : i32
    %c3_i32 = constant 48 : i32
    %c0 = constant 0 : index
    %0 = memref.alloc() : memref<1xi32>
    memref.store %c3_i32, %0[%c0] : memref<1xi32>
    %1 = memref.load %0[%c0] : memref<1xi32>
    %2 = muli %c24_i32, %c36_i32 : i32
    %3 = addi %2, %1 : i32
    memref.store %3, %0[%c0] : memref<1xi32>
    %4 = memref.load %0[%c0] : memref<1xi32>
    return %4 : i32
  }
}
