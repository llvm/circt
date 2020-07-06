// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s
// CHECK:       module {
// CHECK-LABEL:   firrtl.module @handshake.merge_1ins_1outs(
// CHECK:         }
// CHECK-LABEL:   firrtl.module @foo(
// CHECK:         }
// CHECK:       }

module {
  handshake.func @foo(%arg0: si32, %arg1: none, ...) -> (si32, none) {
    %0 = "handshake.merge"(%arg0) : (si32) -> si32
    handshake.return %0, %arg1 : si32, none
  }
}