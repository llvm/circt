// RUN: circt-opt -convert-affine-to-pipeline --split-input-file %s | FileCheck %s

// CHECK-LABEL: func @matvecmul
func.func @matvecmul(%arg0: memref<1x2xi32>) -> memref<1x2xi32> {
  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  // CHECK: %[[C1:.+]] = arith.constant 1 : index

  // CHECK: %[[MAT:.+]] = memref.alloca() : memref<2x2xi32>
  %3 = memref.alloca() : memref<2x2xi32>
  // CHECK: %[[OUTPUT:.+]] = memref.alloca() : memref<1x2xi32>
  %4 = memref.alloca() : memref<1x2xi32>

  // CHECK: %[[LB:.+]] = arith.constant 0 : [[ITER_TYPE:.+]]
  // CHECK: %[[UB:.+]] = arith.constant [[TRIP_COUNT:.+]] : [[ITER_TYPE]]
  // CHECK: %[[STEP:.+]] = arith.constant 1 : [[ITER_TYPE]]

  // Pipeline header.
  // CHECK: pipeline.while II = 1 trip_count = [[TRIP_COUNT]] iter_args(%[[ITER_ARG:.+]] = %[[LB]]) : ([[ITER_TYPE]]) -> ()

  // Condition block.
  // CHECK: %[[COND_RESULT:.+]] = arith.cmpi ult, %[[ITER_ARG]]
  // CHECK: pipeline.register %[[COND_RESULT]]

  // First stage.
  // CHECK: %[[STAGE0:.+]]:9 = pipeline.while.stage start = 0
  // CHECK-DAG: %[[ITER_INC:.+]] = arith.addi %[[ITER_ARG]], %[[STEP]]

  // Second stage.
  // CHECK: %[[STAGE1:.+]]:4 = pipeline.while.stage start = 1
  // CHECK-DAG: %[[STAGE1_0:.+]] = arith.muli %[[STAGE0]]#0, %[[STAGE0]]#1 : i32
  // CHECK-DAG: %[[STAGE1_1:.+]] = arith.muli %[[STAGE0]]#2, %[[STAGE0]]#3 : i32
  // CHECK-DAG: %[[STAGE1_2:.+]] = arith.muli %[[STAGE0]]#4, %[[STAGE0]]#5 : i32
  // CHECK-DAG: %[[STAGE1_3:.+]] = arith.muli %[[STAGE0]]#6, %[[STAGE0]]#7 : i32
  // CHECK: pipeline.register %[[STAGE1_0]], %[[STAGE1_1]], %[[STAGE1_2]], %[[STAGE1_3]]

  // Third stage.
  // CHECK: %[[STAGE2:.+]]:2 = pipeline.while.stage start = 3
  // CHECK: %[[STAGE2_0:.+]] = memref.load %[[OUTPUT]][%arg1, %[[C0]]] : memref<1x2xi32>
  // CHECK: %[[STAGE2_1:.+]] = memref.load %[[OUTPUT]][%arg1, %[[C1]]] : memref<1x2xi32>
  // CHECK: pipeline.register %[[STAGE2_0]], %[[STAGE2_1]]

  // Fourth stage.
  // CHECK: pipeline.while.stage start = 4
  // CHECK: %[[STAGE3_0:.+]] = arith.addi %[[STAGE2]]#0, %[[STAGE1]]#0 : i32
  // CHECK: %[[STAGE3_1:.+]] = arith.addi %[[STAGE3_0]], %[[STAGE1]]#1 : i32
  // CHECK: memref.store %[[STAGE3_1]], %[[OUTPUT]][%arg1, %[[C0]]] : memref<1x2xi32>
  // CHECK: %[[STAGE3_2:.+]] = arith.addi %[[STAGE2]]#1, %[[STAGE1]]#2 : i32
  // CHECK: %[[STAGE3_3:.+]] = arith.addi %[[STAGE3_2]], %[[STAGE1]]#3 : i32
  // CHECK: memref.store %[[STAGE3_3]], %[[OUTPUT]][%arg1, %[[C1]]] : memref<1x2xi32>
  // CHECK: pipeline.register

  // Pipeline terminator.
  // CHECK: pipeline.terminator iter_args(%[[STAGE0]]#8), results()

  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 2 {
        %5 = affine.load %arg0[%arg1, %arg3] : memref<1x2xi32>
        %6 = affine.load %3[%arg3, %arg2] : memref<2x2xi32>
        %7 = affine.load %4[%arg1, %arg2] : memref<1x2xi32>
        %8 = arith.muli %5, %6 : i32
        %9 = arith.addi %7, %8 : i32
        affine.store %9, %4[%arg1, %arg2] : memref<1x2xi32>
      }
    }
  }
  return %4 : memref<1x2xi32>
}

// -----

#map = affine_map<(d0, d1) -> (d0 + d1)>
module attributes {torch.debug_module_name = "ConvPlusReLU"} {
  memref.global "private" constant @__constant_8x2x3x3xi32 : memref<8x2x3x3xi32> = dense<0>
  memref.global "private" constant @__constant_8xi32 : memref<8xi32> = dense<0>
  memref.global "private" constant @__constant_2x8x3x3xi32 : memref<2x8x3x3xi32> = dense<0>
  memref.global "private" constant @__constant_2xi32 : memref<2xi32> = dense<0>
  // CHECK-LABEL: func @small_cnn
  func.func @small_cnn(%arg0: memref<1x2x5x5xi32>) -> memref<1x2x1x1xi32> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0 : i32
    %1 = memref.get_global @__constant_2x8x3x3xi32 : memref<2x8x3x3xi32>
    %2 = memref.get_global @__constant_8xi32 : memref<8xi32>
    %3 = memref.get_global @__constant_8x2x3x3xi32 : memref<8x2x3x3xi32>
    %5 = memref.alloca() : memref<1x8x3x3xi32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 8 {
        affine.for %arg3 = 0 to 3 {
          affine.for %arg4 = 0 to 3 {
            affine.for %arg5 = 0 to 2 {
              affine.for %arg6 = 0 to 3 {
                affine.for %arg7 = 0 to 3 {
                  %9 = affine.apply #map(%arg3, %arg6)
                  %10 = affine.apply #map(%arg4, %arg7)
                  %11 = affine.load %arg0[%arg1, %arg5, %9, %10] : memref<1x2x5x5xi32>
                  %12 = affine.load %3[%arg2, %arg5, %arg6, %arg7] : memref<8x2x3x3xi32>
                  %13 = affine.load %5[%arg1, %arg2, %arg3, %arg4] : memref<1x8x3x3xi32>
                  %14 = arith.muli %11, %12 : i32
                  %15 = arith.addi %13, %14 : i32
                  affine.store %15, %5[%arg1, %arg2, %arg3, %arg4] : memref<1x8x3x3xi32>
                }
              }
            }
          }
        }
      }
    }
    %7 = memref.alloca() : memref<1x2x1x1xi32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 2 {
        affine.for %arg3 = 0 to 1 {
          affine.for %arg4 = 0 to 1 {
            affine.for %arg5 = 0 to 8 {
              affine.for %arg6 = 0 to 3 {
                affine.for %arg7 = 0 to 3 {
                  %9 = affine.apply #map(%arg3, %arg6)
                  %10 = affine.apply #map(%arg4, %arg7)
                  %11 = affine.load %5[%arg1, %arg5, %9, %10] : memref<1x8x3x3xi32>
                  %12 = affine.load %1[%arg2, %arg5, %arg6, %arg7] : memref<2x8x3x3xi32>
                  %13 = affine.load %7[%arg1, %arg2, %arg3, %arg4] : memref<1x2x1x1xi32>
                  %14 = arith.muli %11, %12 : i32
                  %15 = arith.addi %13, %14 : i32
                  affine.store %15, %7[%arg1, %arg2, %arg3, %arg4] : memref<1x2x1x1xi32>
                }
              }
            }
          }
        }
      }
    }
    %8 = memref.alloc() {alignment = 128 : i64} : memref<1x2x1x1xi32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 2 {
        affine.for %arg3 = 0 to 1 {
          affine.for %arg4 = 0 to 1 {
            %9 = affine.load %7[%c0, %arg2, %c0, %c0] : memref<1x2x1x1xi32>
            %10 = arith.cmpi ugt, %9, %cst : i32
            %11 = arith.select %10, %9, %cst : i32
            affine.store %11, %8[%arg1, %arg2, %arg3, %arg4] : memref<1x2x1x1xi32>
          }
        }
      }
    }
    return %8 : memref<1x2x1x1xi32>
  }
}