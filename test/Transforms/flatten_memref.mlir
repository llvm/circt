// RUN: circt-opt -split-input-file --flatten-memref %s | FileCheck %s

// CHECK-LABEL:   func @as_func_arg(
// CHECK-SAME:                           %[[VAL_0:.*]]: memref<16xi32>,
// CHECK-SAME:                           %[[VAL_1:.*]]: index) -> i32 {
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_3:.*]] = arith.shli %[[VAL_1]], %[[VAL_2]] : index
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_3]], %[[VAL_1]] : index
// CHECK:           %[[VAL_5:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_4]]] : memref<16xi32>
// CHECK:           %[[VAL_6:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_7:.*]] = arith.shli %[[VAL_1]], %[[VAL_6]] : index
// CHECK:           %[[VAL_8:.*]] = arith.addi %[[VAL_7]], %[[VAL_1]] : index
// CHECK:           memref.store %[[VAL_5]], %[[VAL_0]]{{\[}}%[[VAL_8]]] : memref<16xi32>
// CHECK:           return %[[VAL_5]] : i32
// CHECK:         }
func.func @as_func_arg(%a : memref<4x4xi32>, %i : index) -> i32 {
  %0 = memref.load %a[%i, %i] : memref<4x4xi32>
  memref.store %0, %a[%i, %i] : memref<4x4xi32>
  return %0 : i32
}

// -----

// CHECK-LABEL:   func @multidim3(
// CHECK:                    %[[VAL_0:.*]]: memref<210xi32>, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index, %[[VAL_3:.*]]: index) -> i32 {
// CHECK:           %[[VAL_4:.*]] = arith.constant 42 : index
// CHECK:           %[[VAL_5:.*]] = arith.muli %[[VAL_1]], %[[VAL_4]] : index
// CHECK:           %[[VAL_6:.*]] = arith.addi %[[VAL_5]], %[[VAL_2]] : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 7 : index
// CHECK:           %[[VAL_8:.*]] = arith.muli %[[VAL_6]], %[[VAL_7]] : index
// CHECK:           %[[VAL_9:.*]] = arith.addi %[[VAL_8]], %[[VAL_3]] : index
// CHECK:           %[[VAL_10:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_9]]] : memref<210xi32>
// CHECK:           return %[[VAL_10]] : i32
// CHECK:         }
func.func @multidim3(%a : memref<5x6x7xi32>, %i1 : index, %i2 : index, %i3 : index) -> i32 {
  %0 = memref.load %a[%i1, %i2, %i3] : memref<5x6x7xi32>
  return %0 : i32
}

// -----

// CHECK-LABEL:   func @multidim5(
// CHECK:                    %[[VAL_0:.*]]: memref<18900xi32>, %[[VAL_1:.*]]: index) -> i32 {
// CHECK:           %[[VAL_2:.*]] = arith.constant 3780 : index
// CHECK:           %[[VAL_3:.*]] = arith.muli %[[VAL_1]], %[[VAL_2]] : index
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_3]], %[[VAL_1]] : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 630 : index
// CHECK:           %[[VAL_6:.*]] = arith.muli %[[VAL_4]], %[[VAL_5]] : index
// CHECK:           %[[VAL_7:.*]] = arith.addi %[[VAL_6]], %[[VAL_1]] : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 90 : index
// CHECK:           %[[VAL_9:.*]] = arith.muli %[[VAL_7]], %[[VAL_8]] : index
// CHECK:           %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[VAL_1]] : index
// CHECK:           %[[VAL_11:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_12:.*]] = arith.muli %[[VAL_10]], %[[VAL_11]] : index
// CHECK:           %[[VAL_13:.*]] = arith.addi %[[VAL_12]], %[[VAL_1]] : index
// CHECK:           %[[VAL_14:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_13]]] : memref<18900xi32>
// CHECK:           return %[[VAL_14]] : i32
// CHECK:         }
func.func @multidim5(%a : memref<5x6x7x9x10xi32>, %i : index) -> i32 {
  %0 = memref.load %a[%i, %i, %i, %i, %i] : memref<5x6x7x9x10xi32>
  return %0 : i32
}

// -----

// CHECK-LABEL:   func @multidim5_p2(
// CHECK:                    %[[VAL_0:.*]]: memref<512xi32>, %[[VAL_1:.*]]: index) -> i32 {
// CHECK:           %[[VAL_2:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_3:.*]] = arith.shli %[[VAL_1]], %[[VAL_2]] : index
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_3]], %[[VAL_1]] : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 6 : index
// CHECK:           %[[VAL_6:.*]] = arith.shli %[[VAL_4]], %[[VAL_5]] : index
// CHECK:           %[[VAL_7:.*]] = arith.addi %[[VAL_6]], %[[VAL_1]] : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_9:.*]] = arith.shli %[[VAL_7]], %[[VAL_8]] : index
// CHECK:           %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[VAL_1]] : index
// CHECK:           %[[VAL_11:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_12:.*]] = arith.shli %[[VAL_10]], %[[VAL_11]] : index
// CHECK:           %[[VAL_13:.*]] = arith.addi %[[VAL_12]], %[[VAL_1]] : index
// CHECK:           %[[VAL_14:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_13]]] : memref<512xi32>
// CHECK:           return %[[VAL_14]] : i32
// CHECK:         }
func.func @multidim5_p2(%a : memref<2x4x8x2x4xi32>, %i : index) -> i32 {
  %0 = memref.load %a[%i, %i, %i, %i, %i] : memref<2x4x8x2x4xi32>
  return %0 : i32
}

// -----

// CHECK-LABEL:   func @as_func_ret(
// CHECK:                      %[[VAL_0:.*]]: memref<16xi32>) -> memref<16xi32> {
// CHECK:           return %[[VAL_0]] : memref<16xi32>
// CHECK:         }
func.func @as_func_ret(%a : memref<4x4xi32>) -> memref<4x4xi32> {
  return %a : memref<4x4xi32>
}

// -----

// CHECK-LABEL:   func @allocs() -> memref<16xi32> {
// CHECK:           %[[VAL_0:.*]] = memref.alloc() : memref<16xi32>
// CHECK:           return %[[VAL_0]] : memref<16xi32>
// CHECK:         }
func.func @allocs() -> memref<4x4xi32> {
  %0 = memref.alloc() : memref<4x4xi32>
  return %0 : memref<4x4xi32>
}

// -----

// CHECK-LABEL:   func @across_bbs(
// CHECK:                     %[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: i1) {
// CHECK:           %[[VAL_3:.*]] = memref.alloc() : memref<16xi32>
// CHECK:           %[[VAL_4:.*]] = memref.alloc() : memref<16xi32>
// CHECK:           cf.cond_br %[[VAL_2]], ^bb1(%[[VAL_3]], %[[VAL_4]] : memref<16xi32>, memref<16xi32>), ^bb1(%[[VAL_4]], %[[VAL_3]] : memref<16xi32>, memref<16xi32>)
// CHECK:         ^bb1(%[[VAL_5:.*]]: memref<16xi32>, %[[VAL_6:.*]]: memref<16xi32>):
// CHECK:           %[[VAL_7:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_8:.*]] = arith.shli %[[VAL_0]], %[[VAL_7]] : index
// CHECK:           %[[VAL_9:.*]] = arith.addi %[[VAL_8]], %[[VAL_1]] : index
// CHECK:           %[[VAL_10:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_9]]] : memref<16xi32>
// CHECK:           %[[VAL_11:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_12:.*]] = arith.shli %[[VAL_0]], %[[VAL_11]] : index
// CHECK:           %[[VAL_13:.*]] = arith.addi %[[VAL_12]], %[[VAL_1]] : index
// CHECK:           memref.store %[[VAL_10]], %[[VAL_6]]{{\[}}%[[VAL_13]]] : memref<16xi32>
// CHECK:           return
// CHECK:         }
func.func @across_bbs(%i1 : index, %i2 : index, %c : i1) {
  %0 = memref.alloc() : memref<4x4xi32>
  %1 = memref.alloc() : memref<4x4xi32>
  cf.cond_br %c,
    ^bb1(%0, %1 : memref<4x4xi32>, memref<4x4xi32>),
    ^bb1(%1, %0 : memref<4x4xi32>, memref<4x4xi32>)
^bb1(%m1 : memref<4x4xi32>, %m2 : memref<4x4xi32>):
  %2 = memref.load %m1[%i1, %i2] : memref<4x4xi32>
  memref.store %2, %m2[%i1, %i2] : memref<4x4xi32>
  return
}

// -----

func.func @foo(%0 : memref<4x4xi32>) -> memref<4x4xi32> {
  return %0 : memref<4x4xi32>
}

// CHECK-LABEL:   func @calls() {
// CHECK:           %[[VAL_0:.*]] = memref.alloc() : memref<16xi32>
// CHECK:           %[[VAL_1:.*]] = call @foo(%[[VAL_0]]) : (memref<16xi32>) -> memref<16xi32>
// CHECK:           return
// CHECK:         }
func.func @calls() {
  %0 = memref.alloc() : memref<4x4xi32>
  %1 = call @foo(%0) : (memref<4x4xi32>) -> (memref<4x4xi32>)
  return
}

// -----

// CHECK-LABEL:   func.func @as_singleton(
// CHECK-SAME:                            %[[VAL_0:.*]]: memref<1xi32>,
// CHECK-SAME:                            %[[VAL_1:.*]]: index) -> i32 {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_2]]] : memref<1xi32>
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           memref.store %[[VAL_3]], %[[VAL_0]]{{\[}}%[[VAL_4]]] : memref<1xi32>
// CHECK:           return %[[VAL_3]] : i32
// CHECK:         }
func.func @as_singleton(%a : memref<i32>, %i : index) -> i32 {
  %0 = memref.load %a[] : memref<i32>
  memref.store %0, %a[] : memref<i32>
  return %0 : i32
}

// -----

// CHECK-LABEL:   func.func @dealloc_copy(
// CHECK-SAME:                            %[[VAL_0:.*]]: memref<16xi32>) -> memref<16xi32> {
// CHECK:           %[[VAL_1:.*]] = memref.alloc() : memref<16xi32>
// CHECK:           memref.copy %[[VAL_0]], %[[VAL_1]] : memref<16xi32> to memref<16xi32>
// CHECK:           memref.dealloc %[[VAL_1]] : memref<16xi32>
// CHECK:           return %[[VAL_1]] : memref<16xi32>
// CHECK:         }
func.func @dealloc_copy(%arg : memref<4x4xi32>) -> memref<4x4xi32> {
  %0 = memref.alloc() : memref<4x4xi32>
  memref.copy %arg, %0 : memref<4x4xi32> to memref<4x4xi32>
  memref.dealloc %0 : memref<4x4xi32>
  return %0 : memref<4x4xi32>
}

// -----

module {
  // CHECK-LABEL: memref.global "private" constant @constant_10xf32_0 : memref<10xf32> = dense<[0.433561265, 0.0884729773, -0.39487046, -0.190938368, 0.705071926, -0.648731529, -0.00710275536, -0.278010637, -0.573243499, 5.029220e-01]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_5x2xf32 : memref<5x2xf32> = dense<[[0.433561265, 0.0884729773], [-0.39487046, -0.190938368], [0.705071926, -0.648731529], [-0.00710275536, -0.278010637], [-0.573243499, 5.029220e-01]]> {alignment = 64 : i64}

  // CHECK-LABEL:   func.func @forward() -> f32 {
  // CHECK:             %[[VAL_0:.*]] = arith.constant 2 : index
  // CHECK:             %[[VAL_1:.*]] = arith.constant 1 : index
  // CHECK:             %[[VAL_2:.*]] = memref.get_global @constant_10xf32_0 : memref<10xf32>
  // CHECK:             %[[VAL_3:.*]] = arith.constant 1 : index
  // CHECK:             %[[VAL_4:.*]] = arith.shli %[[VAL_0]], %[[VAL_3]] : index
  // CHECK:             %[[VAL_5:.*]] = arith.addi %[[VAL_4]], %[[VAL_1]] : index
  // CHECK:             %[[VAL_6:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_5]]] : memref<10xf32>
  // CHECK:             return %[[VAL_6]] : f32
  // CHECK:           }
  // CHECK:         }
  func.func @forward() -> f32 {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = memref.get_global @__constant_5x2xf32 : memref<5x2xf32>
    %1 = memref.load %0[%c2, %c1] : memref<5x2xf32>
    return %1 :f32
  }
}

// GlobalOp/GetGlobalOp may result in name conflict after flattening

module {
  // CHECK-LABEL:   module {
  // CHECK:           memref.global "private" constant @__constant_1xf32 : memref<1xf32> = dense<-0.344258487> {alignment = 64 : i64}
  // CHECK:           memref.global "private" constant @constant_2xf32_1 : memref<2xf32> = dense<[-0.154929623, 0.142687559]> {alignment = 64 : i64}
  // CHECK:           memref.global "private" constant @__constant_2xf32 : memref<2xf32> = dense<[-0.23427248, 0.918611288]> {alignment = 64 : i64}
  // CHECK:           memref.global "private" constant @constant_2xf32_2 : memref<2xf32> = dense<[0.764538527, 0.83000791]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1xf32 : memref<1xf32> = dense<-0.344258487> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1x2xf32 : memref<1x2xf32> = dense<[[-0.154929623, 0.142687559]]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2xf32 : memref<2xf32> = dense<[-0.23427248, 0.918611288]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2x1xf32 : memref<2x1xf32> = dense<[[0.764538527], [0.83000791]]> {alignment = 64 : i64}

  // CHECK:           func.func @main(%[[VAL_0:.*]]: memref<2xf32>, %[[VAL_1:.*]]: memref<1xf32>) {
  // CHECK:             %[[VAL_2:.*]] = arith.constant 2 : index
  // CHECK:             %[[VAL_3:.*]] = arith.constant 1 : index
  // CHECK:             %[[VAL_4:.*]] = arith.constant 0 : index
  // CHECK:             %[[VAL_5:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:             %[[VAL_6:.*]] = memref.get_global @constant_2xf32_2 : memref<2xf32>
  // CHECK:             %[[VAL_7:.*]] = memref.get_global @__constant_2xf32 : memref<2xf32>
  // CHECK:             %[[VAL_8:.*]] = memref.get_global @constant_2xf32_1 : memref<2xf32>
  // CHECK:             %[[VAL_9:.*]] = memref.get_global @__constant_1xf32 : memref<1xf32>
  // CHECK:             %[[VAL_10:.*]] = arith.constant 0 : index
  // CHECK:             %[[VAL_11:.*]] = arith.shli %[[VAL_3]], %[[VAL_10]] : index
  // CHECK:             %[[VAL_12:.*]] = arith.addi %[[VAL_11]], %[[VAL_4]] : index
  // CHECK:             %[[VAL_13:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_12]]] : memref<2xf32>
  // CHECK:             %[[VAL_14:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_4]]] : memref<2xf32>
  // CHECK:             %[[VAL_15:.*]] = arith.constant 1 : index
  // CHECK:             %[[VAL_16:.*]] = arith.shli %[[VAL_4]], %[[VAL_15]] : index
  // CHECK:             %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_3]] : index
  // CHECK:             %[[VAL_18:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_17]]] : memref<2xf32>
  // CHECK:             %[[VAL_19:.*]] = arith.mulf %[[VAL_13]], %[[VAL_14]] : f32
  // CHECK:             %[[VAL_20:.*]] = arith.addf %[[VAL_18]], %[[VAL_19]] : f32
  // CHECK:             memref.store %[[VAL_20]], %[[VAL_9]]{{\[}}%[[VAL_4]]] : memref<1xf32>
  // CHECK:             memref.copy %[[VAL_9]], %[[VAL_1]] : memref<1xf32> to memref<1xf32>
  // CHECK:             return
  // CHECK:           }
  // CHECK:         }

  func.func @main(%arg0: memref<2x1xf32>, %arg1: memref<1xf32>) {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_2x1xf32 : memref<2x1xf32>
    %1 = memref.get_global @__constant_2xf32 : memref<2xf32>
    %2 = memref.get_global @__constant_1x2xf32 : memref<1x2xf32>
    %3 = memref.get_global @__constant_1xf32 : memref<1xf32>
    %4 = memref.load %0[%c1, %c0] : memref<2x1xf32>
    %5 = memref.load %1[%c0] : memref<2xf32>
    %6 = memref.load %2[%c0, %c1] : memref<1x2xf32>
    %7 = arith.mulf %4, %5 : f32
    %8 = arith.addf %6, %7 : f32
    memref.store %8, %3[%c0] : memref<1xf32>
    memref.copy %3, %arg1 : memref<1xf32> to memref<1xf32>
    return
  }
}

// -----

// CHECK:   func.func @main(%[[VAL_0:arg0]]: memref<30xf32>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 30 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = memref.alloc() : memref<30xf32>
// CHECK:           %[[VAL_6:.*]] = memref.get_global @const_1_30 : memref<2xi64>
// CHECK:           scf.for %[[VAL_7:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_2]] {
// CHECK:             scf.for %[[VAL_8:.*]] = %[[VAL_4]] to %[[VAL_1]] step %[[VAL_2]] {
// CHECK:               %[[VAL_9:.*]] = arith.constant 30 : index
// CHECK:               %[[VAL_10:.*]] = arith.muli %[[VAL_4]], %[[VAL_9]] : index
// CHECK:               %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_8]] : index
// CHECK:               %[[VAL_12:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_11]]] : memref<30xf32>
// CHECK:               %[[VAL_13:.*]] = arith.constant 0 : index
// CHECK:               %[[VAL_14:.*]] = arith.shli %[[VAL_4]], %[[VAL_13]] : index
// CHECK:               %[[VAL_15:.*]] = arith.addi %[[VAL_14]], %[[VAL_7]] : index
// CHECK:               memref.store %[[VAL_12]], %[[VAL_0]]{{\[}}%[[VAL_15]]] : memref<30xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

module {
  memref.global "private" constant @const_1_30 : memref<2xi64> = dense<[1, 30]>
  func.func @main(%arg0: memref<30x1xf32>) {
    %c30 = arith.constant 30 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<2x5x3xf32>
    %0 = memref.get_global @const_1_30 : memref<2xi64>
    %reshape = memref.reshape %alloc(%0) : (memref<2x5x3xf32>, memref<2xi64>) -> memref<1x30xf32>
    scf.for %arg2 = %c0 to %c2 step %c1 {
      scf.for %arg3 = %c0 to %c30 step %c1 {
        %4 = memref.load %reshape[%c0, %arg3] : memref<1x30xf32>
        memref.store %4, %arg0[%c0, %arg2] : memref<30x1xf32>
      }
    }
    return
  }
}

// -----
 
 // CHECK-LABEL:   func @allocas() -> memref<16xi32> {
 // CHECK:           %[[VAL_0:.*]] = memref.alloca() : memref<16xi32>
 // CHECK:           return %[[VAL_0]] : memref<16xi32>
 // CHECK:         }
 func.func @allocas() -> memref<4x4xi32> {
   %0 = memref.alloca() : memref<4x4xi32>
   return %0 : memref<4x4xi32>
 }