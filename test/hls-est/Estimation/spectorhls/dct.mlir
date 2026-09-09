module {
  func.func @_Z9DCT8_autoPiS_ii(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: i32, %arg3: i32)  {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.98078525 : f32
    %cst_1 = arith.constant 0.923879504 : f32
    %cst_2 = arith.constant 0.831469595 : f32
    %cst_3 = arith.constant 0.707106769 : f32
    %cst_4 = arith.constant 0.555570245 : f32
    %cst_5 = arith.constant 0.382683426 : f32
    %cst_6 = arith.constant 0.195090324 : f32
    %cst_7 = arith.constant 0.000000e+00 : f32
    %cst_8 = arith.constant -0.195090324 : f32
    %cst_9 = arith.constant -0.382683426 : f32
    %cst_10 = arith.constant -0.555570245 : f32
    %cst_11 = arith.constant -0.707106769 : f32
    %cst_12 = arith.constant -0.831469595 : f32
    %cst_13 = arith.constant -0.923879504 : f32
    %cst_14 = arith.constant -0.98078525 : f32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c16_i32 = arith.constant 16 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_15 = arith.constant 0.353553385 : f32
    %cst_16 = arith.constant 5.000000e-01 : f32
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg2 : i32 to index
    %alloca = memref.alloca() {hls.preserve, polygeist.varname = "C"} : memref<16xf32>
    %alloca_17 = memref.alloca() {hls.preserve, polygeist.varname = "buf"} : memref<8xf32>
    affine.store %cst, %alloca[0] : memref<16xf32>
    affine.store %cst_0, %alloca[1] : memref<16xf32>
    affine.store %cst_1, %alloca[2] : memref<16xf32>
    affine.store %cst_2, %alloca[3] : memref<16xf32>
    affine.store %cst_3, %alloca[4] : memref<16xf32>
    affine.store %cst_4, %alloca[5] : memref<16xf32>
    affine.store %cst_5, %alloca[6] : memref<16xf32>
    affine.store %cst_6, %alloca[7] : memref<16xf32>
    affine.store %cst_7, %alloca[8] : memref<16xf32>
    affine.store %cst_8, %alloca[9] : memref<16xf32>
    affine.store %cst_9, %alloca[10] : memref<16xf32>
    affine.store %cst_10, %alloca[11] : memref<16xf32>
    affine.store %cst_11, %alloca[12] : memref<16xf32>
    affine.store %cst_12, %alloca[13] : memref<16xf32>
    affine.store %cst_13, %alloca[14] : memref<16xf32>
    affine.store %cst_14, %alloca[15] : memref<16xf32>
    affine.for %arg4 = 0 to 8 {
      %5 = arith.index_cast %arg4 : index to i32
      affine.store %cst_7, %alloca_17[%arg4] : memref<8xf32>
      %6 = affine.for %arg5 = 0 to 8 iter_args(%arg6 = %cst_7) -> (f32) {
        %7 = arith.index_cast %arg5 : index to i32
        %8 = affine.load %arg1[%arg5 * symbol(%0)] : memref<?xi32>
        %9 = arith.sitofp %8 : i32 to f32
        %10 = arith.muli %7, %c2_i32 : i32
        %11 = arith.muli %10, %5 : i32
        %12 = arith.addi %11, %5 : i32
        %13 = arith.remsi %12, %c16_i32 : i32
        %14 = arith.index_cast %13 : i32 to index
        %15 = memref.load %alloca[%14] : memref<16xf32>
        %16 = arith.mulf %9, %15 : f32
        %17 = arith.divsi %12, %c16_i32 : i32
        %18 = arith.remsi %17, %c2_i32 : i32
        %19 = arith.cmpi ne, %18, %c0_i32 : i32
        %20 = arith.select %19, %c-1_i32, %c1_i32 : i32
        %21 = arith.sitofp %20 : i32 to f32
        %22 = arith.mulf %16, %21 : f32
        %23 = arith.addf %arg6, %22 : f32
        affine.store %23, %alloca_17[%arg4] : memref<8xf32>
        affine.yield %23 : f32
      }
    }
    %2 = affine.load %alloca_17[0] : memref<8xf32>
    %3 = arith.mulf %2, %cst_15 : f32
    %4 = arith.fptosi %3 : f32 to i32
    affine.store %4, %arg0[0] : memref<?xi32>
    affine.for %arg4 = 1 to 8 {
      %5 = affine.load %alloca_17[%arg4] : memref<8xf32>
      %6 = arith.mulf %5, %cst_16 : f32
      %7 = arith.fptosi %6 : f32 to i32
      affine.store %7, %arg0[%arg4 * symbol(%1)] : memref<?xi32>
    }
    return
  }
  func.func @_Z3DCTPiS_(%arg0: memref<4194304xi32>, %arg1: memref<4194304xi32>)  {
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant 0.353553385 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant -0.98078525 : f32
    %cst_2 = arith.constant -0.923879504 : f32
    %cst_3 = arith.constant -0.831469595 : f32
    %cst_4 = arith.constant -0.707106769 : f32
    %cst_5 = arith.constant -0.555570245 : f32
    %cst_6 = arith.constant -0.382683426 : f32
    %cst_7 = arith.constant -0.195090324 : f32
    %cst_8 = arith.constant 0.000000e+00 : f32
    %cst_9 = arith.constant 0.195090324 : f32
    %cst_10 = arith.constant 0.382683426 : f32
    %cst_11 = arith.constant 0.555570245 : f32
    %cst_12 = arith.constant 0.707106769 : f32
    %cst_13 = arith.constant 0.831469595 : f32
    %cst_14 = arith.constant 0.923879504 : f32
    %cst_15 = arith.constant 0.98078525 : f32
    %cst_16 = arith.constant 1.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %alloca = memref.alloca() {hls.preserve, polygeist.varname = "C"} : memref<16xf32>
    %alloca_17 = memref.alloca() {hls.preserve, polygeist.varname = "buf"} : memref<8xf32>
    %alloca_18 = memref.alloca() {hls.preserve, polygeist.varname = "C"} : memref<16xf32>
    %alloca_19 = memref.alloca() {hls.preserve, polygeist.varname = "buf"} : memref<8xf32>
    affine.for %arg2 = 7 to 2048 step 8 {
      affine.for %arg3 = 7 to 2048 step 8 {
        affine.for %arg4 = 0 to 8 {
          affine.store %cst_16, %alloca[0] : memref<16xf32>
          affine.store %cst_15, %alloca[1] : memref<16xf32>
          affine.store %cst_14, %alloca[2] : memref<16xf32>
          affine.store %cst_13, %alloca[3] : memref<16xf32>
          affine.store %cst_12, %alloca[4] : memref<16xf32>
          affine.store %cst_11, %alloca[5] : memref<16xf32>
          affine.store %cst_10, %alloca[6] : memref<16xf32>
          affine.store %cst_9, %alloca[7] : memref<16xf32>
          affine.store %cst_8, %alloca[8] : memref<16xf32>
          affine.store %cst_7, %alloca[9] : memref<16xf32>
          affine.store %cst_6, %alloca[10] : memref<16xf32>
          affine.store %cst_5, %alloca[11] : memref<16xf32>
          affine.store %cst_4, %alloca[12] : memref<16xf32>
          affine.store %cst_3, %alloca[13] : memref<16xf32>
          affine.store %cst_2, %alloca[14] : memref<16xf32>
          affine.store %cst_1, %alloca[15] : memref<16xf32>
          affine.for %arg5 = 0 to 8 {
            %3 = arith.index_cast %arg5 : index to i32
            affine.store %cst_8, %alloca_17[%arg5] : memref<8xf32>
            %4 = affine.for %arg6 = 0 to 8 iter_args(%arg7 = %cst_8) -> (f32) {
              %5 = arith.index_cast %arg6 : index to i32
              %6 = affine.load %arg1[%arg6 + %arg3 + %arg4 * 2048 + %arg2 * 2048 - 14343] : memref<4194304xi32>
              %7 = arith.sitofp %6 : i32 to f32
              %8 = arith.muli %5, %c2_i32 : i32
              %9 = arith.muli %8, %3 : i32
              %10 = arith.addi %9, %3 : i32
              %11 = arith.remsi %10, %c16_i32 : i32
              %12 = arith.index_cast %11 : i32 to index
              %13 = memref.load %alloca[%12] : memref<16xf32>
              %14 = arith.mulf %7, %13 : f32
              %15 = arith.divsi %10, %c16_i32 : i32
              %16 = arith.remsi %15, %c2_i32 : i32
              %17 = arith.cmpi ne, %16, %c0_i32 : i32
              %18 = arith.select %17, %c-1_i32, %c1_i32 : i32
              %19 = arith.sitofp %18 : i32 to f32
              %20 = arith.mulf %14, %19 : f32
              %21 = arith.addf %arg7, %20 : f32
              affine.store %21, %alloca_17[%arg5] : memref<8xf32>
              affine.yield %21 : f32
            }
          }
          %0 = affine.load %alloca_17[0] : memref<8xf32>
          %1 = arith.mulf %0, %cst_0 : f32
          %2 = arith.fptosi %1 : f32 to i32
          affine.store %2, %arg0[%arg3 + %arg4 * 2048 + %arg2 * 2048 - 14343] : memref<4194304xi32>
          affine.for %arg5 = 1 to 8 {
            %3 = affine.load %alloca_17[%arg5] : memref<8xf32>
            %4 = arith.mulf %3, %cst : f32
            %5 = arith.fptosi %4 : f32 to i32
            affine.store %5, %arg0[%arg5 + %arg3 + %arg4 * 2048 + %arg2 * 2048 - 14343] : memref<4194304xi32>
          }
        }
        affine.for %arg4 = 0 to 8 {
          affine.store %cst_16, %alloca_18[0] : memref<16xf32>
          affine.store %cst_15, %alloca_18[1] : memref<16xf32>
          affine.store %cst_14, %alloca_18[2] : memref<16xf32>
          affine.store %cst_13, %alloca_18[3] : memref<16xf32>
          affine.store %cst_12, %alloca_18[4] : memref<16xf32>
          affine.store %cst_11, %alloca_18[5] : memref<16xf32>
          affine.store %cst_10, %alloca_18[6] : memref<16xf32>
          affine.store %cst_9, %alloca_18[7] : memref<16xf32>
          affine.store %cst_8, %alloca_18[8] : memref<16xf32>
          affine.store %cst_7, %alloca_18[9] : memref<16xf32>
          affine.store %cst_6, %alloca_18[10] : memref<16xf32>
          affine.store %cst_5, %alloca_18[11] : memref<16xf32>
          affine.store %cst_4, %alloca_18[12] : memref<16xf32>
          affine.store %cst_3, %alloca_18[13] : memref<16xf32>
          affine.store %cst_2, %alloca_18[14] : memref<16xf32>
          affine.store %cst_1, %alloca_18[15] : memref<16xf32>
          affine.for %arg5 = 0 to 8 {
            %3 = arith.index_cast %arg5 : index to i32
            affine.store %cst_8, %alloca_19[%arg5] : memref<8xf32>
            %4 = affine.for %arg6 = 0 to 8 iter_args(%arg7 = %cst_8) -> (f32) {
              %5 = arith.index_cast %arg6 : index to i32
              %6 = affine.load %arg0[%arg6 * 2048 + %arg4 + %arg3 + %arg2 * 2048 - 14343] : memref<4194304xi32>
              %7 = arith.sitofp %6 : i32 to f32
              %8 = arith.muli %5, %c2_i32 : i32
              %9 = arith.muli %8, %3 : i32
              %10 = arith.addi %9, %3 : i32
              %11 = arith.remsi %10, %c16_i32 : i32
              %12 = arith.index_cast %11 : i32 to index
              %13 = memref.load %alloca_18[%12] : memref<16xf32>
              %14 = arith.mulf %7, %13 : f32
              %15 = arith.divsi %10, %c16_i32 : i32
              %16 = arith.remsi %15, %c2_i32 : i32
              %17 = arith.cmpi ne, %16, %c0_i32 : i32
              %18 = arith.select %17, %c-1_i32, %c1_i32 : i32
              %19 = arith.sitofp %18 : i32 to f32
              %20 = arith.mulf %14, %19 : f32
              %21 = arith.addf %arg7, %20 : f32
              affine.store %21, %alloca_19[%arg5] : memref<8xf32>
              affine.yield %21 : f32
            }
          }
          %0 = affine.load %alloca_19[0] : memref<8xf32>
          %1 = arith.mulf %0, %cst_0 : f32
          %2 = arith.fptosi %1 : f32 to i32
          affine.store %2, %arg0[%arg4 + %arg3 + %arg2 * 2048 - 14343] : memref<4194304xi32>
          affine.for %arg5 = 1 to 8 {
            %3 = affine.load %alloca_19[%arg5] : memref<8xf32>
            %4 = arith.mulf %3, %cst : f32
            %5 = arith.fptosi %4 : f32 to i32
            affine.store %5, %arg0[%arg5 * 2048 + %arg4 + %arg3 + %arg2 * 2048 - 14343] : memref<4194304xi32>
          }
        }
      }
    }
    return
  }
}
