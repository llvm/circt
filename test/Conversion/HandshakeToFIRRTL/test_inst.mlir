// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s
// CHECK:       module {
// CHECK-LABEL:   firrtl.module @handshake.merge_1ins_1outs(
// CHECK:         }
// CHECK-LABEL:   firrtl.module @std.addi_2ins_1outs(
// CHECK:         }
// CHECK-LABEL:   firrtl.module @test_inst(
// CHECK:         }
// CHECK:       }

module {
  handshake.func @test_inst(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: none, ...) -> (i32, none) {
    %0 = "handshake.merge"(%arg0) : (i32) -> i32
    %1 = "handshake.merge"(%arg1) : (i32) -> i32
    %2 = "handshake.merge"(%arg2) : (i32) -> i32
    %3 = addi %0, %1 : i32
    %4 = addi %2, %3 : i32
    handshake.return %4, %arg3 : i32, none
  }
}