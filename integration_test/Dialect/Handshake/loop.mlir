// REQUIRES: ieee-sim
// RUN: circt-opt %s --create-dataflow --simple-canonicalizer --cse --handshake-insert-buffer=strategies=all > %loop-handshake.mlir
// RUN: circt-opt %loop-handshake.mlir --lower-handshake-to-firrtl --firrtl-lower-types --firrtl-imconstprop --lower-firrtl-to-hw --hw-cleanup --simple-canonicalizer --cse --hw-legalize-names > %loop-hw.mlir
// RUN: circt-translate %loop-hw.mlir --export-verilog > %loop-export.sv
// RUN: circt-rtl-sim.py %loop-export.sv %S/driver.sv --sim %ieee-sim --no-default-driver --top driver | FileCheck %s
// CHECK: Result={{.*}}42

module  {
  func @top() -> i32 {
    %c42_i32 = constant 42 : i32
    %c1_i32 = constant 1 : i32
    br ^bb1(%c1_i32 : i32)
  ^bb1(%0: i32):  // 2 preds: ^bb0, ^bb2
    %1 = cmpi slt, %0, %c42_i32 : i32
    cond_br %1, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %2 = addi %0, %c1_i32 : i32
    br ^bb1(%2 : i32)
  ^bb3:  // pred: ^bb1
    return %0 : i32
  }
}
