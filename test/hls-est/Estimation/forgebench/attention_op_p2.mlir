module {
  memref.global @BRAM_1 : memref<8x64xf16> = uninitialized
  memref.global @BRAM_weights_v : memref<64x64xf16> = uninitialized
  memref.global @BRAM_weights_k : memref<64x64xf16> = uninitialized
  memref.global @BRAM_weights_q : memref<64x64xf16> = uninitialized
  memref.global @BRAM_attn_input : memref<8x64xf16> = uninitialized
  func.func @_Z24load_8_64_ap_fixed_16_5_PA64_fS0_(%arg0: memref<8x64xf16>, %arg1: memref<8x64xf16>) {
    affine.for %arg2 = 0 to 8 {
      affine.for %arg3 = 0 to 64 {
        %0 = affine.load %arg0[%arg2, %arg3] : memref<8x64xf16>
        affine.store %0, %arg1[%arg2, %arg3] : memref<8x64xf16>
      }
    }
    return
  }
  func.func @_Z25load_64_64_ap_fixed_16_5_PA64_fS0_(%arg0: memref<64x64xf16>, %arg1: memref<64x64xf16>) {
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 64 {
        %0 = affine.load %arg0[%arg2, %arg3] : memref<64x64xf16>
        affine.store %0, %arg1[%arg2, %arg3] : memref<64x64xf16>
      }
    }
    return
  }
  func.func @_Z52grouped_multihead_attention_8_64_16_4_ap_fixed_16_5_PA64_fS0_S0_S0_S0_i(%arg0: memref<8x64xf16>, %arg1: memref<64x64xf16>, %arg2: memref<64x64xf16>, %arg3: memref<64x64xf16>, %arg4: memref<8x64xf16>, %arg5: i32) {
    %cst = arith.constant 5.000000e-01 : f16
    %cst_0 = arith.constant 0.000000e+00 : f16
    %c16_i32 = arith.constant 16 : i32
    %alloca = memref.alloca() {hls.preserve} : memref<8x8xf16>
    %alloca_1 = memref.alloca() {hls.preserve} : memref<8x64xf16>
    %alloca_2 = memref.alloca() {hls.preserve} : memref<8x64xf16>
    %alloca_3 = memref.alloca() {hls.preserve} : memref<8x64xf16>
    %0 = arith.divsi %c16_i32, %arg5 : i32
    %1 = arith.index_cast %0 : i32 to index
    affine.for %arg6 = 0 to 8 {
      affine.for %arg7 = 0 to 64 {
        affine.store %cst_0, %alloca_3[%arg6, %arg7] : memref<8x64xf16>
        affine.store %cst_0, %alloca_2[%arg6, %arg7] : memref<8x64xf16>
        affine.store %cst_0, %alloca_1[%arg6, %arg7] : memref<8x64xf16>
        %3:3 = affine.for %arg8 = 0 to 64 iter_args(%arg9 = %cst_0, %arg10 = %cst_0, %arg11 = %cst_0) -> (f16, f16, f16) {
          %4 = affine.load %arg0[%arg6, %arg8] : memref<8x64xf16>
          %5 = affine.load %arg1[%arg7, %arg8] : memref<64x64xf16>
          %6 = arith.mulf %4, %5 : f16
          %7 = arith.addf %arg11, %6 : f16
          affine.store %7, %alloca_3[%arg6, %arg7] : memref<8x64xf16>
          %8 = affine.load %arg0[%arg6, %arg8] : memref<8x64xf16>
          %9 = affine.load %arg2[%arg7, %arg8] : memref<64x64xf16>
          %10 = arith.mulf %8, %9 : f16
          %11 = arith.addf %arg10, %10 : f16
          affine.store %11, %alloca_2[%arg6, %arg7] : memref<8x64xf16>
          %12 = affine.load %arg0[%arg6, %arg8] : memref<8x64xf16>
          %13 = affine.load %arg3[%arg7, %arg8] : memref<64x64xf16>
          %14 = arith.mulf %12, %13 : f16
          %15 = arith.addf %arg9, %14 : f16
          affine.store %15, %alloca_1[%arg6, %arg7] : memref<8x64xf16>
          affine.yield %15, %11, %7 : f16, f16, f16
        }
      }
    }
    %2 = arith.index_cast %arg5 : i32 to index
    affine.for %arg6 = 0 to %2 {
      affine.for %arg7 = 0 to %1 {
        affine.for %arg8 = 0 to 8 {
          affine.for %arg9 = 0 to 8 {
            affine.store %cst_0, %alloca[%arg8, %arg9] : memref<8x8xf16>
            %3 = affine.for %arg10 = 0 to 4 iter_args(%arg11 = %cst_0) -> (f16) {
              %6 = affine.load %alloca_3[%arg8, %arg10 + %arg7 * 4 + (%arg6 * symbol(%1)) * 4] : memref<8x64xf16>
              %7 = affine.load %alloca_2[%arg9, %arg10 + %arg7 * 4 + (%arg6 * symbol(%1)) * 4] : memref<8x64xf16>
              %8 = arith.mulf %6, %7 : f16
              %9 = arith.addf %arg11, %8 : f16
              affine.store %9, %alloca[%arg8, %arg9] : memref<8x8xf16>
              affine.yield %9 : f16
            }
            %4 = affine.load %alloca[%arg8, %arg9] : memref<8x8xf16>
            %5 = arith.mulf %4, %cst : f16
            affine.store %5, %alloca[%arg8, %arg9] : memref<8x8xf16>
          }
        }
        affine.for %arg8 = 0 to 8 {
          %3 = affine.load %alloca[%arg8, 0] : memref<8x8xf16>
          %4 = affine.for %arg9 = 1 to 8 iter_args(%arg10 = %3) -> (f16) {
            %6 = affine.load %alloca[%arg8, %arg9] : memref<8x8xf16>
            %7 = arith.cmpf ogt, %6, %arg10 : f16
            %8 = arith.select %7, %6, %arg10 : f16
            affine.yield %8 : f16
          }
          %5 = affine.for %arg9 = 0 to 8 iter_args(%arg10 = %cst_0) -> (f16) {
            %6 = affine.load %alloca[%arg8, %arg9] : memref<8x8xf16>
            %7 = arith.subf %6, %4 : f16
            %8 = math.exp %7 : f16
            affine.store %8, %alloca[%arg8, %arg9] : memref<8x8xf16>
            %9 = arith.addf %arg10, %8 : f16
            affine.yield %9 : f16
          }
          affine.for %arg9 = 0 to 8 {
            %6 = affine.load %alloca[%arg8, %arg9] : memref<8x8xf16>
            %7 = arith.divf %6, %5 : f16
            affine.store %7, %alloca[%arg8, %arg9] : memref<8x8xf16>
          }
        }
        affine.for %arg8 = 0 to 8 {
          affine.for %arg9 = 0 to 4 {
            %3 = affine.for %arg10 = 0 to 8 iter_args(%arg11 = %cst_0) -> (f16) {
              %4 = affine.load %alloca[%arg8, %arg10] : memref<8x8xf16>
              %5 = affine.load %alloca_1[%arg10, %arg9 + %arg7 * 4 + (%arg6 * symbol(%1)) * 4] : memref<8x64xf16>
              %6 = arith.mulf %4, %5 : f16
              %7 = arith.addf %arg11, %6 : f16
              affine.yield %7 : f16
            }
            affine.store %3, %arg4[%arg8, %arg9 + %arg7 * 4 + (%arg6 * symbol(%1)) * 4] : memref<8x64xf16>
          }
        }
      }
    }
    return
  }
  func.func @_ZSt4sqrtf(%arg0: f16) -> f16 {
    %0 = math.sqrt %arg0 : f16
    return %0 : f16
  }
  func.func @_Z25store_8_64_ap_fixed_16_5_PA64_fS0_(%arg0: memref<8x64xf16>, %arg1: memref<8x64xf16>) {
    affine.for %arg2 = 0 to 8 {
      affine.for %arg3 = 0 to 64 {
        %0 = affine.load %arg0[%arg2, %arg3] : memref<8x64xf16>
        affine.store %0, %arg1[%arg2, %arg3] : memref<8x64xf16>
      }
    }
    return
  }
  func.func @_Z3topPA64_fS0_S0_S0_S0_(%arg0: memref<8x64xf16>, %arg1: memref<64x64xf16>, %arg2: memref<64x64xf16>, %arg3: memref<64x64xf16>, %arg4: memref<8x64xf16>) {
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 5.000000e-01 : f16
    %0 = memref.get_global @BRAM_attn_input : memref<8x64xf16>
    affine.for %arg5 = 0 to 8 {
      affine.for %arg6 = 0 to 64 {
        %5 = affine.load %arg0[%arg5, %arg6] : memref<8x64xf16>
        affine.store %5, %0[%arg5, %arg6] : memref<8x64xf16>
      }
    }
    %1 = memref.get_global @BRAM_weights_q : memref<64x64xf16>
    affine.for %arg5 = 0 to 64 {
      affine.for %arg6 = 0 to 64 {
        %5 = affine.load %arg1[%arg5, %arg6] : memref<64x64xf16>
        affine.store %5, %1[%arg5, %arg6] : memref<64x64xf16>
      }
    }
    %2 = memref.get_global @BRAM_weights_k : memref<64x64xf16>
    affine.for %arg5 = 0 to 64 {
      affine.for %arg6 = 0 to 64 {
        %5 = affine.load %arg2[%arg5, %arg6] : memref<64x64xf16>
        affine.store %5, %2[%arg5, %arg6] : memref<64x64xf16>
      }
    }
    %3 = memref.get_global @BRAM_weights_v : memref<64x64xf16>
    affine.for %arg5 = 0 to 64 {
      affine.for %arg6 = 0 to 64 {
        %5 = affine.load %arg3[%arg5, %arg6] : memref<64x64xf16>
        affine.store %5, %3[%arg5, %arg6] : memref<64x64xf16>
      }
    }
    %4 = memref.get_global @BRAM_1 : memref<8x64xf16>
    %alloca = memref.alloca() {hls.preserve} : memref<8x8xf16>
    %alloca_1 = memref.alloca() {hls.preserve} : memref<8x64xf16>
    %alloca_2 = memref.alloca() {hls.preserve} : memref<8x64xf16>
    %alloca_3 = memref.alloca() {hls.preserve} : memref<8x64xf16>
    affine.for %arg5 = 0 to 8 {
      affine.for %arg6 = 0 to 64 {
        affine.store %cst, %alloca_3[%arg5, %arg6] : memref<8x64xf16>
        affine.store %cst, %alloca_2[%arg5, %arg6] : memref<8x64xf16>
        affine.store %cst, %alloca_1[%arg5, %arg6] : memref<8x64xf16>
        %5:3 = affine.for %arg7 = 0 to 64 iter_args(%arg8 = %cst, %arg9 = %cst, %arg10 = %cst) -> (f16, f16, f16) {
          %6 = affine.load %0[%arg5, %arg7] : memref<8x64xf16>
          %7 = affine.load %1[%arg6, %arg7] : memref<64x64xf16>
          %8 = arith.mulf %6, %7 : f16
          %9 = arith.addf %arg10, %8 : f16
          affine.store %9, %alloca_3[%arg5, %arg6] : memref<8x64xf16>
          %10 = affine.load %0[%arg5, %arg7] : memref<8x64xf16>
          %11 = affine.load %2[%arg6, %arg7] : memref<64x64xf16>
          %12 = arith.mulf %10, %11 : f16
          %13 = arith.addf %arg9, %12 : f16
          affine.store %13, %alloca_2[%arg5, %arg6] : memref<8x64xf16>
          %14 = affine.load %0[%arg5, %arg7] : memref<8x64xf16>
          %15 = affine.load %3[%arg6, %arg7] : memref<64x64xf16>
          %16 = arith.mulf %14, %15 : f16
          %17 = arith.addf %arg8, %16 : f16
          affine.store %17, %alloca_1[%arg5, %arg6] : memref<8x64xf16>
          affine.yield %17, %13, %9 : f16, f16, f16
        }
      }
    }
    affine.for %arg5 = 0 to 8 {
      affine.for %arg6 = 0 to 2 {
        affine.for %arg7 = 0 to 8 {
          affine.for %arg8 = 0 to 8 {
            affine.store %cst, %alloca[%arg7, %arg8] : memref<8x8xf16>
            %5 = affine.for %arg9 = 0 to 4 iter_args(%arg10 = %cst) -> (f16) {
              %8 = affine.load %alloca_3[%arg7, %arg9 + %arg5 * 8 + %arg6 * 4] : memref<8x64xf16>
              %9 = affine.load %alloca_2[%arg8, %arg9 + %arg5 * 8 + %arg6 * 4] : memref<8x64xf16>
              %10 = arith.mulf %8, %9 : f16
              %11 = arith.addf %arg10, %10 : f16
              affine.store %11, %alloca[%arg7, %arg8] : memref<8x8xf16>
              affine.yield %11 : f16
            }
            %6 = affine.load %alloca[%arg7, %arg8] : memref<8x8xf16>
            %7 = arith.mulf %6, %cst_0 : f16
            affine.store %7, %alloca[%arg7, %arg8] : memref<8x8xf16>
          }
        }
        affine.for %arg7 = 0 to 8 {
          %5 = affine.load %alloca[%arg7, 0] : memref<8x8xf16>
          %6 = affine.for %arg8 = 1 to 8 iter_args(%arg9 = %5) -> (f16) {
            %8 = affine.load %alloca[%arg7, %arg8] : memref<8x8xf16>
            %9 = arith.cmpf ogt, %8, %arg9 : f16
            %10 = arith.select %9, %8, %arg9 : f16
            affine.yield %10 : f16
          }
          %7 = affine.for %arg8 = 0 to 8 iter_args(%arg9 = %cst) -> (f16) {
            %8 = affine.load %alloca[%arg7, %arg8] : memref<8x8xf16>
            %9 = arith.subf %8, %6 : f16
            %10 = math.exp %9 : f16
            affine.store %10, %alloca[%arg7, %arg8] : memref<8x8xf16>
            %11 = arith.addf %arg9, %10 : f16
            affine.yield %11 : f16
          }
          affine.for %arg8 = 0 to 8 {
            %8 = affine.load %alloca[%arg7, %arg8] : memref<8x8xf16>
            %9 = arith.divf %8, %7 : f16
            affine.store %9, %alloca[%arg7, %arg8] : memref<8x8xf16>
          }
        }
        affine.for %arg7 = 0 to 8 {
          affine.for %arg8 = 0 to 4 {
            %5 = affine.for %arg9 = 0 to 8 iter_args(%arg10 = %cst) -> (f16) {
              %6 = affine.load %alloca[%arg7, %arg9] : memref<8x8xf16>
              %7 = affine.load %alloca_1[%arg9, %arg8 + %arg5 * 8 + %arg6 * 4] : memref<8x64xf16>
              %8 = arith.mulf %6, %7 : f16
              %9 = arith.addf %arg10, %8 : f16
              affine.yield %9 : f16
            }
            affine.store %5, %4[%arg7, %arg8 + %arg5 * 8 + %arg6 * 4] : memref<8x64xf16>
          }
        }
      }
    }
    affine.for %arg5 = 0 to 8 {
      affine.for %arg6 = 0 to 64 {
        %5 = affine.load %4[%arg5, %arg6] : memref<8x64xf16>
        affine.store %5, %arg4[%arg5, %arg6] : memref<8x64xf16>
      }
    }
    return
  }
}
