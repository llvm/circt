// XFAIL: *
// REQUIRES: ieee-sim
// RUN: circt-opt %s --create-dataflow --simple-canonicalizer --handshake-insert-buffer=strategies=control > %matmul-handshake.mlir
// RUN: circt-opt %matmul-handshake.mlir --lower-handshake-to-firrtl --firrtl-lower-types --firrtl-imconstprop --lower-firrtl-to-rtl --rtl-memory-sim --rtl-cleanup --simple-canonicalizer --cse --rtl-legalize-names > %matmul-rtl.mlir
// RUN: circt-translate %matmul-rtl.mlir --export-verilog > %matmul-export.sv
// RUN: circt-rtl-sim.py %matmul-export.sv %S/driver.sv --sim %ieee-sim --no-default-driver --top driver | FileCheck %s
// CHECK: Result={{.*}}200

module  {
  func @top() -> i32 {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c8 = constant 8 : index
    // %c63 = constant 63 : index
    %c64 = constant 64 : index
    // %c0_i32 = constant 0 : i32
    %c5_i32 = constant 5 : i32
    // %0 = memref.alloc() : memref<64xi32>
    // %1 = memref.alloc() : memref<64xi32>
    // %2 = memref.alloc() : memref<64xi32>
    // %3 = memref.alloc() : memref<1xi32>
    br ^bb1(%c0 : index)
  ^bb1(%4: index):  // 2 preds: ^bb0, ^bb2
    %5 = cmpi slt, %4, %c64 : index
    cond_br %5, ^bb2, ^bb3(%c0 : index)
  ^bb2:  // pred: ^bb1
    // memref.store %c5_i32, %0[%4] : memref<64xi32>
    // memref.store %c5_i32, %1[%4] : memref<64xi32>
    %6 = addi %4, %c1 : index
    br ^bb1(%6 : index)
  ^bb3(%7: index):  // 2 preds: ^bb1, ^bb9
    %8 = cmpi slt, %7, %c8 : index
    cond_br %8, ^bb4(%c0 : index), ^bb10
  ^bb4(%9: index):  // 2 preds: ^bb3, ^bb8
    %10 = cmpi slt, %9, %c8 : index
    cond_br %10, ^bb5, ^bb9
  ^bb5:  // pred: ^bb4
    // memref.store %c0_i32, %3[%c0] : memref<1xi32>
    %11 = muli %7, %c8 : index
    br ^bb6(%c0 : index)
  ^bb6(%12: index):  // 2 preds: ^bb5, ^bb7
    %13 = cmpi slt, %12, %c8 : index
    cond_br %13, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %14 = addi %11, %12 : index
    // %15 = memref.load %0[%14] : memref<64xi32>
    %16 = muli %12, %c8 : index
    %17 = addi %16, %9 : index
    // %18 = memref.load %1[%17] : memref<64xi32>
    // %19 = memref.load %3[%c0] : memref<1xi32>
    // %20 = muli %15, %18 : i32
    // %21 = addi %19, %20 : i32
    // memref.store %21, %3[%c0] : memref<1xi32>
    %22 = addi %12, %c1 : index
    br ^bb6(%22 : index)
  ^bb8:  // pred: ^bb6
    %23 = addi %11, %9 : index
    // %24 = memref.load %3[%c0] : memref<1xi32>
    // memref.store %24, %2[%23] : memref<64xi32>
    %25 = addi %9, %c1 : index
    br ^bb4(%25 : index)
  ^bb9:  // pred: ^bb4
    %26 = addi %7, %c1 : index
    br ^bb3(%26 : index)
  ^bb10:  // pred: ^bb3
    // %27 = memref.load %2[%c63] : memref<64xi32>
    return %c5_i32 : i32
  }
}
