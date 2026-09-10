// RUN: circt-opt --exclude-exec-region-canonicalize --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                    %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<4xi32>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           scf.execute_region {
// CHECK:             memref.store %[[VAL_1]], %[[VAL_0]]{{\[}}%[[VAL_2]]] : memref<4xi32>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           return
// CHECK:         }

module {
  func.func @main(%arg0 : memref<4xi32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %idx = arith.constant 0 : index
    %true = arith.constant true
    %false = arith.constant false

    %val = scf.if %true -> i32 {
      %0 = arith.addi %c1, %c1 : i32
      scf.yield %0 : i32
    } else {
      %0 = arith.addi %c0, %c0 : i32
      scf.yield %0 : i32
    }

    scf.execute_region {
      scf.if %false {
        %0 = arith.addi %val, %c1 : i32
        memref.store %0, %arg0[%idx] : memref<4xi32>
      } else {
        %0 = arith.addi %c0, %c0 : i32
        memref.store %0, %arg0[%idx] : memref<4xi32>
      }
      scf.yield
    }

    return
  }
}
