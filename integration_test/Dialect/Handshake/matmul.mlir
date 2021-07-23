// REQUIRES: ieee-sim
// RUN: circt-opt %s --create-dataflow --canonicalize='top-down=true region-simplify=true' --handshake-insert-buffer=strategies=all --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --lower-to-hw --verilog > %matmul-export.sv
// RUN: circt-rtl-sim.py %matmul-export.sv %S/driver.sv --sim %ieee-sim --no-default-driver --top driver | FileCheck %s
// CHECK: Result={{.*}}448704

module {
  func @top() -> i32 {
    %c123_i32 = constant 123 : i32
    %c456_i32 = constant 456 : i32
    %c0_i32 = constant 0 : i32
    %c64 = constant 64 : index
    %c8 = constant 8 : index
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c63 = constant 63 : index
    %0 = memref.alloc() : memref<64xi32>
    %1 = memref.alloc() : memref<64xi32>
    %2 = memref.alloc() : memref<64xi32>
    br ^bb1(%c0 : index)
  ^bb1(%3: index):  // 2 preds: ^bb0, ^bb2
    %4 = cmpi slt, %3, %c64 : index
    cond_br %4, ^bb2, ^bb3(%c0 : index)
  ^bb2:  // pred: ^bb1
    memref.store %c123_i32, %0[%3] : memref<64xi32>
    memref.store %c456_i32, %1[%3] : memref<64xi32>
    memref.store %c0_i32, %2[%3] : memref<64xi32>
    %5 = addi %3, %c1 : index
    br ^bb1(%5 : index)
  ^bb3(%6: index):  // 2 preds: ^bb1, ^bb9
    %7 = cmpi slt, %6, %c8 : index
    cond_br %7, ^bb4(%c0 : index), ^bb10
  ^bb4(%8: index):  // 2 preds: ^bb3, ^bb8
    %9 = cmpi slt, %8, %c8 : index
    cond_br %9, ^bb5, ^bb9
  ^bb5:  // pred: ^bb4
    %10 = muli %6, %c8 : index
    br ^bb6(%c0, %c0_i32 : index, i32)
  ^bb6(%11: index, %12: i32):  // 2 preds: ^bb5, ^bb7
    %13 = cmpi slt, %11, %c8 : index
    cond_br %13, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %14 = addi %10, %11 : index
    %15 = memref.load %0[%14] : memref<64xi32>
    %16 = muli %11, %c8 : index
    %17 = addi %16, %8 : index
    %18 = memref.load %1[%17] : memref<64xi32>
    %19 = muli %15, %18 : i32
    %20 = addi %12, %19 : i32
    %21 = addi %11, %c1 : index
    br ^bb6(%21, %20 : index, i32)
  ^bb8:  // pred: ^bb6
    %22 = addi %10, %8 : index
    memref.store %12, %2[%22] : memref<64xi32>
    %23 = addi %8, %c1 : index
    br ^bb4(%23 : index)
  ^bb9:  // pred: ^bb4
    %24 = addi %6, %c1 : index
    br ^bb3(%24 : index)
  ^bb10:  // pred: ^bb3
    %25 = memref.load %2[%c63] : memref<64xi32>
    return %25 : i32
  }
}
