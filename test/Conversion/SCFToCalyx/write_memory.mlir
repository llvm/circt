// RUN: circt-opt %s --lower-scf-to-calyx="write-json=data" -canonicalize -split-input-file>/dev/null && cat ./data.json | FileCheck %s

// CHECK-LABEL:   "mem_0": {
// CHECK:           "data": [
// CHECK:             0,
// CHECK:             0,
// CHECK:             0,
// CHECK:             0
// CHECK:           ],
// CHECK:           "format": {
// CHECK:             "is_signed": true,
// CHECK:             "numeric_type": "ieee754_float",
// CHECK:             "width": 32
// CHECK:           }
// CHECK:         },

// CHECK-LABEL:   "mem_1": {
// CHECK:           "data": [
// CHECK:             43,
// CHECK:             8,
// CHECK:             -39,
// CHECK:             -19,
// CHECK:             70,
// CHECK:             -64,
// CHECK:             -7,
// CHECK:             -27,
// CHECK:             -57,
// CHECK:             5
// CHECK:           ],
// CHECK:           "format": {
// CHECK:             "is_signed": false,
// CHECK:             "numeric_type": "bitnum",
// CHECK:             "width": 32
// CHECK:           }
// CHECK:         },

// CHECK-LABEL:   "mem_2": {
// CHECK:           "data": [
// CHECK:             0.69999998807907104,
// CHECK:             -4.1999998092651367,
// CHECK:             0
// CHECK:           ],
// CHECK:           "format": {
// CHECK:             "is_signed": true,
// CHECK:             "numeric_type": "ieee754_float",
// CHECK:             "width": 32
// CHECK:           }
// CHECK:         }

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

