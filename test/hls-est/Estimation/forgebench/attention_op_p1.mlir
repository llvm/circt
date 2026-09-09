module {
  memref.global @BRAM_1 : memref<8x256xf32> = uninitialized
  memref.global @BRAM_weights_v : memref<256x256xf32> = uninitialized
  memref.global @BRAM_weights_k : memref<256x256xf32> = uninitialized
  memref.global @BRAM_weights_q : memref<256x256xf32> = uninitialized
  memref.global @BRAM_attn_input : memref<8x256xf32> = uninitialized
  func.func @_Z25load_8_256_ap_fixed_16_5_PA256_fS0_(%arg0: memref<8x256xf32>, %arg1: memref<8x256xf32>) {
    affine.for %arg2 = 0 to 8 {
      affine.for %arg3 = 0 to 256 {
        %0 = affine.load %arg0[%arg2, %arg3] : memref<8x256xf32>
        affine.store %0, %arg1[%arg2, %arg3] : memref<8x256xf32>
      }
    }
    return
  }
  func.func @_Z27load_256_256_ap_fixed_16_5_PA256_fS0_(%arg0: memref<256x256xf32>, %arg1: memref<256x256xf32>) {
    affine.for %arg2 = 0 to 256 {
      affine.for %arg3 = 0 to 256 {
        %0 = affine.load %arg0[%arg2, %arg3] : memref<256x256xf32>
        affine.store %0, %arg1[%arg2, %arg3] : memref<256x256xf32>
      }
    }
    return
  }
  func.func @_Z54grouped_multihead_attention_8_256_16_16_ap_fixed_16_5_PA256_fS0_S0_S0_S0_i(%arg0: memref<8x256xf32>, %arg1: memref<256x256xf32>, %arg2: memref<256x256xf32>, %arg3: memref<256x256xf32>, %arg4: memref<8x256xf32>, %arg5: i32) {
    %cst = arith.constant 2.500000e-01 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c16_i32 = arith.constant 16 : i32
    %alloca = memref.alloca() {hls.preserve} : memref<8x8xf32>
    %alloca_1 = memref.alloca() {hls.preserve} : memref<8x256xf32>
    %alloca_2 = memref.alloca() {hls.preserve} : memref<8x256xf32>
    %alloca_3 = memref.alloca() {hls.preserve} : memref<8x256xf32>
    %0 = arith.divsi %c16_i32, %arg5 : i32
    %1 = arith.index_cast %0 : i32 to index
    affine.for %arg6 = 0 to 8 {
      affine.for %arg7 = 0 to 256 {
        affine.store %cst_0, %alloca_3[%arg6, %arg7] : memref<8x256xf32>
        affine.store %cst_0, %alloca_2[%arg6, %arg7] : memref<8x256xf32>
        affine.store %cst_0, %alloca_1[%arg6, %arg7] : memref<8x256xf32>
        %3:3 = affine.for %arg8 = 0 to 256 iter_args(%arg9 = %cst_0, %arg10 = %cst_0, %arg11 = %cst_0) -> (f32, f32, f32) {
          %4 = affine.load %arg0[%arg6, %arg8] : memref<8x256xf32>
          %5 = affine.load %arg1[%arg7, %arg8] : memref<256x256xf32>
          %6 = arith.mulf %4, %5 : f32
          %7 = arith.addf %arg11, %6 : f32
          affine.store %7, %alloca_3[%arg6, %arg7] : memref<8x256xf32>
          %8 = affine.load %arg0[%arg6, %arg8] : memref<8x256xf32>
          %9 = affine.load %arg2[%arg7, %arg8] : memref<256x256xf32>
          %10 = arith.mulf %8, %9 : f32
          %11 = arith.addf %arg10, %10 : f32
          affine.store %11, %alloca_2[%arg6, %arg7] : memref<8x256xf32>
          %12 = affine.load %arg0[%arg6, %arg8] : memref<8x256xf32>
          %13 = affine.load %arg3[%arg7, %arg8] : memref<256x256xf32>
          %14 = arith.mulf %12, %13 : f32
          %15 = arith.addf %arg9, %14 : f32
          affine.store %15, %alloca_1[%arg6, %arg7] : memref<8x256xf32>
          affine.yield %15, %11, %7 : f32, f32, f32
        }
      }
    }
    %2 = arith.index_cast %arg5 : i32 to index
    affine.for %arg6 = 0 to %2 {
      affine.for %arg7 = 0 to %1 {
        affine.for %arg8 = 0 to 8 {
          affine.for %arg9 = 0 to 8 {
            affine.store %cst_0, %alloca[%arg8, %arg9] : memref<8x8xf32>
            %3 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_0) -> (f32) {
              %6 = affine.load %alloca_3[%arg8, %arg10 + %arg7 * 16 + (%arg6 * symbol(%1)) * 16] : memref<8x256xf32>
              %7 = affine.load %alloca_2[%arg9, %arg10 + %arg7 * 16 + (%arg6 * symbol(%1)) * 16] : memref<8x256xf32>
              %8 = arith.mulf %6, %7 : f32
              %9 = arith.addf %arg11, %8 : f32
              affine.store %9, %alloca[%arg8, %arg9] : memref<8x8xf32>
              affine.yield %9 : f32
            }
            %4 = affine.load %alloca[%arg8, %arg9] : memref<8x8xf32>
            %5 = arith.mulf %4, %cst : f32
            affine.store %5, %alloca[%arg8, %arg9] : memref<8x8xf32>
          }
        }
        affine.for %arg8 = 0 to 8 {
          %3 = affine.load %alloca[%arg8, 0] : memref<8x8xf32>
          %4 = affine.for %arg9 = 1 to 8 iter_args(%arg10 = %3) -> (f32) {
            %6 = affine.load %alloca[%arg8, %arg9] : memref<8x8xf32>
            %7 = arith.cmpf ogt, %6, %arg10 : f32
            %8 = arith.select %7, %6, %arg10 : f32
            affine.yield %8 : f32
          }
          %5 = affine.for %arg9 = 0 to 8 iter_args(%arg10 = %cst_0) -> (f32) {
            %6 = affine.load %alloca[%arg8, %arg9] : memref<8x8xf32>
            %7 = arith.subf %6, %4 : f32
            %8 = math.exp %7 : f32
            affine.store %8, %alloca[%arg8, %arg9] : memref<8x8xf32>
            %9 = arith.addf %arg10, %8 : f32
            affine.yield %9 : f32
          }
          affine.for %arg9 = 0 to 8 {
            %6 = affine.load %alloca[%arg8, %arg9] : memref<8x8xf32>
            %7 = arith.divf %6, %5 : f32
            affine.store %7, %alloca[%arg8, %arg9] : memref<8x8xf32>
          }
        }
        affine.for %arg8 = 0 to 8 {
          affine.for %arg9 = 0 to 16 {
            %3 = affine.for %arg10 = 0 to 8 iter_args(%arg11 = %cst_0) -> (f32) {
              %4 = affine.load %alloca[%arg8, %arg10] : memref<8x8xf32>
              %5 = affine.load %alloca_1[%arg10, %arg9 + %arg7 * 16 + (%arg6 * symbol(%1)) * 16] : memref<8x256xf32>
              %6 = arith.mulf %4, %5 : f32
              %7 = arith.addf %arg11, %6 : f32
              affine.yield %7 : f32
            }
            affine.store %3, %arg4[%arg8, %arg9 + %arg7 * 16 + (%arg6 * symbol(%1)) * 16] : memref<8x256xf32>
          }
        }
      }
    }
    return
  }
  func.func @_ZSt4sqrtf(%arg0: f32) -> f32 {
    %0 = math.sqrt %arg0 : f32
    return %0 : f32
  }
  func.func @_Z26store_8_256_ap_fixed_16_5_PA256_fS0_(%arg0: memref<8x256xf32>, %arg1: memref<8x256xf32>) {
    affine.for %arg2 = 0 to 8 {
      affine.for %arg3 = 0 to 256 {
        %0 = affine.load %arg0[%arg2, %arg3] : memref<8x256xf32>
        affine.store %0, %arg1[%arg2, %arg3] : memref<8x256xf32>
      }
    }
    return
  }
  func.func @_Z3topPA256_fS0_S0_S0_S0_(%arg0: memref<8x256xf32>, %arg1: memref<256x256xf32>, %arg2: memref<256x256xf32>, %arg3: memref<256x256xf32>, %arg4: memref<8x256xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.500000e-01 : f32
    %0 = memref.get_global @BRAM_attn_input : memref<8x256xf32>
    affine.for %arg5 = 0 to 8 {
      affine.for %arg6 = 0 to 256 {
        %5 = affine.load %arg0[%arg5, %arg6] : memref<8x256xf32>
        affine.store %5, %0[%arg5, %arg6] : memref<8x256xf32>
      }
    }
    %1 = memref.get_global @BRAM_weights_q : memref<256x256xf32>
    affine.for %arg5 = 0 to 256 {
      affine.for %arg6 = 0 to 256 {
        %5 = affine.load %arg1[%arg5, %arg6] : memref<256x256xf32>
        affine.store %5, %1[%arg5, %arg6] : memref<256x256xf32>
      }
    }
    %2 = memref.get_global @BRAM_weights_k : memref<256x256xf32>
    affine.for %arg5 = 0 to 256 {
      affine.for %arg6 = 0 to 256 {
        %5 = affine.load %arg2[%arg5, %arg6] : memref<256x256xf32>
        affine.store %5, %2[%arg5, %arg6] : memref<256x256xf32>
      }
    }
    %3 = memref.get_global @BRAM_weights_v : memref<256x256xf32>
    affine.for %arg5 = 0 to 256 {
      affine.for %arg6 = 0 to 256 {
        %5 = affine.load %arg3[%arg5, %arg6] : memref<256x256xf32>
        affine.store %5, %3[%arg5, %arg6] : memref<256x256xf32>
      }
    }
    %4 = memref.get_global @BRAM_1 : memref<8x256xf32>
    %alloca = memref.alloca() {hls.preserve} : memref<8x8xf32>
    %alloca_1 = memref.alloca() {hls.preserve} : memref<8x256xf32>
    %alloca_2 = memref.alloca() {hls.preserve} : memref<8x256xf32>
    %alloca_3 = memref.alloca() {hls.preserve} : memref<8x256xf32>
    affine.for %arg5 = 0 to 8 {
      affine.for %arg6 = 0 to 256 {
        affine.store %cst, %alloca_3[%arg5, %arg6] : memref<8x256xf32>
        affine.store %cst, %alloca_2[%arg5, %arg6] : memref<8x256xf32>
        affine.store %cst, %alloca_1[%arg5, %arg6] : memref<8x256xf32>
        %5:3 = affine.for %arg7 = 0 to 256 iter_args(%arg8 = %cst, %arg9 = %cst, %arg10 = %cst) -> (f32, f32, f32) {
          %6 = affine.load %0[%arg5, %arg7] : memref<8x256xf32>
          %7 = affine.load %1[%arg6, %arg7] : memref<256x256xf32>
          %8 = arith.mulf %6, %7 : f32
          %9 = arith.addf %arg10, %8 : f32
          affine.store %9, %alloca_3[%arg5, %arg6] : memref<8x256xf32>
          %10 = affine.load %0[%arg5, %arg7] : memref<8x256xf32>
          %11 = affine.load %2[%arg6, %arg7] : memref<256x256xf32>
          %12 = arith.mulf %10, %11 : f32
          %13 = arith.addf %arg9, %12 : f32
          affine.store %13, %alloca_2[%arg5, %arg6] : memref<8x256xf32>
          %14 = affine.load %0[%arg5, %arg7] : memref<8x256xf32>
          %15 = affine.load %3[%arg6, %arg7] : memref<256x256xf32>
          %16 = arith.mulf %14, %15 : f32
          %17 = arith.addf %arg8, %16 : f32
          affine.store %17, %alloca_1[%arg5, %arg6] : memref<8x256xf32>
          affine.yield %17, %13, %9 : f32, f32, f32
        }
      }
    }
    affine.for %arg5 = 0 to 16 {
      affine.for %arg6 = 0 to 8 {
        affine.for %arg7 = 0 to 8 {
          affine.store %cst, %alloca[%arg6, %arg7] : memref<8x8xf32>
          %5 = affine.for %arg8 = 0 to 16 iter_args(%arg9 = %cst) -> (f32) {
            %8 = affine.load %alloca_3[%arg6, %arg8 + %arg5 * 16] : memref<8x256xf32>
            %9 = affine.load %alloca_2[%arg7, %arg8 + %arg5 * 16] : memref<8x256xf32>
            %10 = arith.mulf %8, %9 : f32
            %11 = arith.addf %arg9, %10 : f32
            affine.store %11, %alloca[%arg6, %arg7] : memref<8x8xf32>
            affine.yield %11 : f32
          }
          %6 = affine.load %alloca[%arg6, %arg7] : memref<8x8xf32>
          %7 = arith.mulf %6, %cst_0 : f32
          affine.store %7, %alloca[%arg6, %arg7] : memref<8x8xf32>
        }
      }
      affine.for %arg6 = 0 to 8 {
        %5 = affine.load %alloca[%arg6, 0] : memref<8x8xf32>
        %6 = affine.for %arg7 = 1 to 8 iter_args(%arg8 = %5) -> (f32) {
          %8 = affine.load %alloca[%arg6, %arg7] : memref<8x8xf32>
          %9 = arith.cmpf ogt, %8, %arg8 : f32
          %10 = arith.select %9, %8, %arg8 : f32
          affine.yield %10 : f32
        }
        %7 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %cst) -> (f32) {
          %8 = affine.load %alloca[%arg6, %arg7] : memref<8x8xf32>
          %9 = arith.subf %8, %6 : f32
          %10 = math.exp %9 : f32
          affine.store %10, %alloca[%arg6, %arg7] : memref<8x8xf32>
          %11 = arith.addf %arg8, %10 : f32
          affine.yield %11 : f32
        }
        affine.for %arg7 = 0 to 8 {
          %8 = affine.load %alloca[%arg6, %arg7] : memref<8x8xf32>
          %9 = arith.divf %8, %7 : f32
          affine.store %9, %alloca[%arg6, %arg7] : memref<8x8xf32>
        }
      }
      affine.for %arg6 = 0 to 8 {
        affine.for %arg7 = 0 to 16 {
          %5 = affine.for %arg8 = 0 to 8 iter_args(%arg9 = %cst) -> (f32) {
            %6 = affine.load %alloca[%arg6, %arg8] : memref<8x8xf32>
            %7 = affine.load %alloca_1[%arg8, %arg7 + %arg5 * 16] : memref<8x256xf32>
            %8 = arith.mulf %6, %7 : f32
            %9 = arith.addf %arg9, %8 : f32
            affine.yield %9 : f32
          }
          affine.store %5, %4[%arg6, %arg7 + %arg5 * 16] : memref<8x256xf32>
        }
      }
    }
    affine.for %arg5 = 0 to 8 {
      affine.for %arg6 = 0 to 256 {
        %5 = affine.load %4[%arg5, %arg6] : memref<8x256xf32>
        affine.store %5, %arg4[%arg5, %arg6] : memref<8x256xf32>
      }
    }
    return
  }
}
