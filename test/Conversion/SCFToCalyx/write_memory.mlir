// RUN: circt-opt %s --lower-scf-to-calyx="write-json=data" -canonicalize>/dev/null && cat $(dirname %s)/data.json | FileCheck %s

// CHECK-LABEL:   "mem_0": {
// CHECK-DAG:           "data": [
// CHECK-DAG:             0,
// CHECK-DAG:             0,
// CHECK-DAG:             0,
// CHECK-DAG:             0
// CHECK-DAG:           ],
// CHECK-DAG:           "format": {
// CHECK-DAG:             "is_signed": false,
// CHECK-DAG:             "numeric_type": "ieee754_float",
// CHECK-DAG:             "width": 32
// CHECK-DAG:           }
// CHECK-DAG:         },

// CHECK-LABEL:   "mem_1": {
// CHECK-DAG:           "data": [
// CHECK-DAG:             43,
// CHECK-DAG:             8,
// CHECK-DAG:             -39,
// CHECK-DAG:             -19,
// CHECK-DAG:             70,
// CHECK-DAG:             -64,
// CHECK-DAG:             -7,
// CHECK-DAG:             -27,
// CHECK-DAG:             -57,
// CHECK-DAG:             5
// CHECK-DAG:           ],
// CHECK-DAG:           "format": {
// CHECK-DAG:             "is_signed": true,
// CHECK-DAG:             "numeric_type": "bitnum",
// CHECK-DAG:             "width": 32
// CHECK-DAG:           }
// CHECK-DAG:         },

// CHECK-LABEL:   "mem_2": {
// CHECK-DAG:           "data": [
// CHECK-DAG:             0.69999998807907104,
// CHECK-DAG:             -4.1999998092651367,
// CHECK-DAG:             0
// CHECK-DAG:           ],
// CHECK-DAG:           "format": {
// CHECK-DAG:             "is_signed": true,
// CHECK-DAG:             "numeric_type": "ieee754_float",
// CHECK-DAG:             "width": 32
// CHECK-DAG:           }
// CHECK-DAG:         }

module {
  memref.global "private" constant @constant_10xi32_0 : memref<10xi32> = dense<[43, 8, -39, -19, 70, -64, -7, -27, -57, 5]>
  memref.global "private" constant @constant_3xf32 : memref<3xf32> = dense<[0.7, -4.2, 0.0]>
  func.func @main(%arg_idx : index) -> i32 {
    %alloc = memref.alloc() : memref<4xf32>
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = memref.get_global @constant_10xi32_0 : memref<10xi32>
    %ret = memref.load %0[%arg_idx] : memref<10xi32>
    %1 = memref.get_global @constant_3xf32 : memref<3xf32>
    %2 = memref.load %1[%c1] : memref<3xf32>
    memref.store %2, %alloc[%c2] : memref<4xf32>
    return %ret : i32
  }
}

