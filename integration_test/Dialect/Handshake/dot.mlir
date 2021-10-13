// REQUIRES: ieee-sim
// RUN: circt-opt %s --lower-std-to-handshake --canonicalize='top-down=true region-simplify=true' --handshake-insert-buffer=strategies=all --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --lower-to-hw --verilog > %dot-export.sv
// RUN: circt-rtl-sim.py %dot-export.sv %S/driver.sv --sim %ieee-sim --no-default-driver --top driver | FileCheck %s
// CHECK: Result={{.*}}3589632

module {
  func @top() -> i32 {
    %c123_i32 = constant 123 : i32
    %c456_i32 = constant 456 : i32
    %c0_i32 = constant 0 : i32
    %c0 = constant 0 : index
    %c64 = constant 64 : index
    %c1 = constant 1 : index
    %0 = memref.alloc() : memref<64xi32>
    %1 = memref.alloc() : memref<64xi32>
    br ^bb1(%c0 : index)
  ^bb1(%2: index):  // 2 preds: ^bb0, ^bb2
    %3 = cmpi slt, %2, %c64 : index
    cond_br %3, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    memref.store %c123_i32, %0[%2] : memref<64xi32>
    memref.store %c456_i32, %1[%2] : memref<64xi32>
    %4 = addi %2, %c1 : index
    br ^bb1(%4 : index)
  ^bb3:  // pred: ^bb1
    br ^bb4(%c0, %c0_i32 : index, i32)
  ^bb4(%5: index, %6: i32):  // 2 preds: ^bb3, ^bb5
    %7 = cmpi slt, %5, %c64 : index
    cond_br %7, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %8 = memref.load %0[%5] : memref<64xi32>
    %9 = memref.load %1[%5] : memref<64xi32>
    %10 = muli %8, %9 : i32
    %11 = addi %6, %10 : i32
    %12 = addi %5, %c1 : index
    br ^bb4(%12, %11 : index, i32)
  ^bb6:  // pred: ^bb4
    return %6 : i32
  }
}

