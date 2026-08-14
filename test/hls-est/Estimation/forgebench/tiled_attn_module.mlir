// RUN: hls-est %s --affine-loop-normalize --memory-banking-bram --convert-affine-to-loopschedule --bram-analysis | FileCheck %s -check-prefix=CHECK-BRAM

module {
  func.func @_Z25load_8_256_ap_fixed_16_5_PA256_fS0_(%arg0: memref<8x256xf32>, %arg1: memref<8x256xf32>){
    affine.for %arg2 = 0 to 8 {
      affine.for %arg3 = 0 to 256 {
        %0 = affine.load %arg0[%arg2, %arg3] : memref<8x256xf32>
        affine.store %0, %arg1[%arg2, %arg3] : memref<8x256xf32>
      }
    }
    return
  }
  func.func @_Z26load_64_256_ap_fixed_16_5_PA256_fS0_(%arg0: memref<64x256xf32>, %arg1: memref<64x256xf32>) {
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 256 {
        %0 = affine.load %arg0[%arg2, %arg3] : memref<64x256xf32>
        affine.store %0, %arg1[%arg2, %arg3] : memref<64x256xf32>
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
  func.func @_Z12transpose_64PA64_f(%arg0: memref<64x64xf32>) {
    affine.for %arg1 = 0 to 64 {
      affine.for %arg2 = 0 to 64 {
        %0 = affine.load %arg0[%arg2, %arg1] : memref<64x64xf32>
        affine.store %0, %arg0[%arg1, %arg2] : memref<64x64xf32>
      }
    }
    return
  }
  func.func @_Z12transpoe_256PA256_f(%arg0: memref<256x256xf32>) {
    affine.for %arg1 = 0 to 256 {
      affine.for %arg2 = 0 to 256 {
        %0 = affine.load %arg0[%arg2, %arg1] : memref<256x256xf32>
        affine.store %0, %arg0[%arg1, %arg2] : memref<256x256xf32>
      }
    }
    return
  }
  func.func @_Z53grouped_multihead_attention_8_256_4_16_ap_fixed_16_5_PA256_fS0_S0_S0_PA64_f(%arg0: memref<8x256xf32>, %arg1: memref<64x256xf32>, %arg2: memref<64x256xf32>, %arg3: memref<64x256xf32>, %arg4: memref<8x64xf32>) {
    %cst = arith.constant 2.500000e-01 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca() {hls.preserve, polygeist.varname = "scores"} : memref<8x8xf32>
    %alloca_1 = memref.alloca() {hls.preserve, polygeist.varname = "V"} : memref<8x64xf32>
    %alloca_2 = memref.alloca() {hls.preserve, polygeist.varname = "K"} : memref<8x64xf32>
    %alloca_3 = memref.alloca() {hls.preserve, polygeist.varname = "Q"} : memref<8x64xf32>
    affine.for %arg5 = 0 to 8 {
      affine.for %arg6 = 0 to 64 {
        affine.store %cst_0, %alloca_3[%arg5, %arg6] : memref<8x64xf32>
        affine.store %cst_0, %alloca_2[%arg5, %arg6] : memref<8x64xf32>
        affine.store %cst_0, %alloca_1[%arg5, %arg6] : memref<8x64xf32>
        %0:3 = affine.for %arg7 = 0 to 256 iter_args(%arg8 = %cst_0, %arg9 = %cst_0, %arg10 = %cst_0) -> (f32, f32, f32) {
          %1 = affine.load %arg0[%arg5, %arg7] : memref<8x256xf32>
          %2 = affine.load %arg1[%arg6, %arg7] : memref<64x256xf32>
          %3 = arith.mulf %1, %2 : f32
          %4 = arith.addf %arg10, %3 : f32
          affine.store %4, %alloca_3[%arg5, %arg6] : memref<8x64xf32>
          %5 = affine.load %arg0[%arg5, %arg7] : memref<8x256xf32>
          %6 = affine.load %arg2[%arg6, %arg7] : memref<64x256xf32>
          %7 = arith.mulf %5, %6 : f32
          %8 = arith.addf %arg9, %7 : f32
          affine.store %8, %alloca_2[%arg5, %arg6] : memref<8x64xf32>
          %9 = affine.load %arg0[%arg5, %arg7] : memref<8x256xf32>
          %10 = affine.load %arg3[%arg6, %arg7] : memref<64x256xf32>
          %11 = arith.mulf %9, %10 : f32
          %12 = arith.addf %arg8, %11 : f32
          affine.store %12, %alloca_1[%arg5, %arg6] : memref<8x64xf32>
          affine.yield %12, %8, %4 : f32, f32, f32
        }
      }
    }
    affine.for %arg5 = 0 to 4 {
      affine.for %arg6 = 0 to 8 {
        affine.for %arg7 = 0 to 8 {
          affine.store %cst_0, %alloca[%arg6, %arg7] : memref<8x8xf32>
          %0 = affine.for %arg8 = 0 to 16 iter_args(%arg9 = %cst_0) -> (f32) {
            %3 = affine.load %alloca_3[%arg6, %arg8 + %arg5 * 16] : memref<8x64xf32>
            %4 = affine.load %alloca_2[%arg7, %arg8 + %arg5 * 16] : memref<8x64xf32>
            %5 = arith.mulf %3, %4 : f32
            %6 = arith.addf %arg9, %5 : f32
            affine.store %6, %alloca[%arg6, %arg7] : memref<8x8xf32>
            affine.yield %6 : f32
          }
          %1 = affine.load %alloca[%arg6, %arg7] : memref<8x8xf32>
          %2 = arith.mulf %1, %cst : f32
          affine.store %2, %alloca[%arg6, %arg7] : memref<8x8xf32>
        }
      }
      affine.for %arg6 = 0 to 8 {
        %0 = affine.load %alloca[%arg6, 0] : memref<8x8xf32>
        %1 = affine.for %arg7 = 1 to 8 iter_args(%arg8 = %0) -> (f32) {
          %3 = affine.load %alloca[%arg6, %arg7] : memref<8x8xf32>
          %4 = arith.cmpf ogt, %3, %arg8 : f32
          %5 = arith.select %4, %3, %arg8 : f32
          affine.yield %5 : f32
        }
        %2 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %cst_0) -> (f32) {
          %3 = affine.load %alloca[%arg6, %arg7] : memref<8x8xf32>
          %4 = arith.subf %3, %1 : f32
          %5 = math.exp %4 : f32
          affine.store %5, %alloca[%arg6, %arg7] : memref<8x8xf32>
          %6 = arith.addf %arg8, %5 : f32
          affine.yield %6 : f32
        }
        affine.for %arg7 = 0 to 8 {
          %3 = affine.load %alloca[%arg6, %arg7] : memref<8x8xf32>
          %4 = arith.divf %3, %2 : f32
          affine.store %4, %alloca[%arg6, %arg7] : memref<8x8xf32>
        }
      }
      affine.for %arg6 = 0 to 8 {
        affine.for %arg7 = 0 to 16 {
          %0 = affine.for %arg8 = 0 to 8 iter_args(%arg9 = %cst_0) -> (f32) {
            %1 = affine.load %alloca[%arg6, %arg8] : memref<8x8xf32>
            %2 = affine.load %alloca_1[%arg8, %arg7 + %arg5 * 16] : memref<8x64xf32>
            %3 = arith.mulf %1, %2 : f32
            %4 = arith.addf %arg9, %3 : f32
            affine.yield %4 : f32
          }
          affine.store %0, %arg4[%arg6, %arg7 + %arg5 * 16] : memref<8x64xf32>
        }
      }
    }
    return
  }
  func.func @_ZSt4sqrtf(%arg0: f32) -> f32 {
    %0 = math.sqrt %arg0 : f32
    return %0 : f32
  }
  func.func @_Z54grouped_multihead_attention_8_256_16_16_ap_fixed_16_5_PA256_fS0_S0_S0_S0_(%arg0: memref<8x256xf32>, %arg1: memref<256x256xf32>, %arg2: memref<256x256xf32>, %arg3: memref<256x256xf32>, %arg4: memref<8x256xf32>) {
    %cst = arith.constant 2.500000e-01 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca() {hls.preserve, polygeist.varname = "scores"} : memref<8x8xf32>
    %alloca_1 = memref.alloca() {hls.preserve, polygeist.varname = "V"} : memref<8x256xf32>
    %alloca_2 = memref.alloca() {hls.preserve, polygeist.varname = "K"} : memref<8x256xf32>
    %alloca_3 = memref.alloca() {hls.preserve, polygeist.varname = "Q"} : memref<8x256xf32>
    affine.for %arg5 = 0 to 8 {
      affine.for %arg6 = 0 to 256 {
        affine.store %cst_0, %alloca_3[%arg5, %arg6] : memref<8x256xf32>
        affine.store %cst_0, %alloca_2[%arg5, %arg6] : memref<8x256xf32>
        affine.store %cst_0, %alloca_1[%arg5, %arg6] : memref<8x256xf32>
        %0:3 = affine.for %arg7 = 0 to 256 iter_args(%arg8 = %cst_0, %arg9 = %cst_0, %arg10 = %cst_0) -> (f32, f32, f32) {
          %1 = affine.load %arg0[%arg5, %arg7] : memref<8x256xf32>
          %2 = affine.load %arg1[%arg6, %arg7] : memref<256x256xf32>
          %3 = arith.mulf %1, %2 : f32
          %4 = arith.addf %arg10, %3 : f32
          affine.store %4, %alloca_3[%arg5, %arg6] : memref<8x256xf32>
          %5 = affine.load %arg0[%arg5, %arg7] : memref<8x256xf32>
          %6 = affine.load %arg2[%arg6, %arg7] : memref<256x256xf32>
          %7 = arith.mulf %5, %6 : f32
          %8 = arith.addf %arg9, %7 : f32
          affine.store %8, %alloca_2[%arg5, %arg6] : memref<8x256xf32>
          %9 = affine.load %arg0[%arg5, %arg7] : memref<8x256xf32>
          %10 = affine.load %arg3[%arg6, %arg7] : memref<256x256xf32>
          %11 = arith.mulf %9, %10 : f32
          %12 = arith.addf %arg8, %11 : f32
          affine.store %12, %alloca_1[%arg5, %arg6] : memref<8x256xf32>
          affine.yield %12, %8, %4 : f32, f32, f32
        }
      }
    }
    affine.for %arg5 = 0 to 16 {
      affine.for %arg6 = 0 to 8 {
        affine.for %arg7 = 0 to 8 {
          affine.store %cst_0, %alloca[%arg6, %arg7] : memref<8x8xf32>
          %0 = affine.for %arg8 = 0 to 16 iter_args(%arg9 = %cst_0) -> (f32) {
            %3 = affine.load %alloca_3[%arg6, %arg8 + %arg5 * 16] : memref<8x256xf32>
            %4 = affine.load %alloca_2[%arg7, %arg8 + %arg5 * 16] : memref<8x256xf32>
            %5 = arith.mulf %3, %4 : f32
            %6 = arith.addf %arg9, %5 : f32
            affine.store %6, %alloca[%arg6, %arg7] : memref<8x8xf32>
            affine.yield %6 : f32
          }
          %1 = affine.load %alloca[%arg6, %arg7] : memref<8x8xf32>
          %2 = arith.mulf %1, %cst : f32
          affine.store %2, %alloca[%arg6, %arg7] : memref<8x8xf32>
        }
      }
      affine.for %arg6 = 0 to 8 {
        %0 = affine.load %alloca[%arg6, 0] : memref<8x8xf32>
        %1 = affine.for %arg7 = 1 to 8 iter_args(%arg8 = %0) -> (f32) {
          %3 = affine.load %alloca[%arg6, %arg7] : memref<8x8xf32>
          %4 = arith.cmpf ogt, %3, %arg8 : f32
          %5 = arith.select %4, %3, %arg8 : f32
          affine.yield %5 : f32
        }
        %2 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %cst_0) -> (f32) {
          %3 = affine.load %alloca[%arg6, %arg7] : memref<8x8xf32>
          %4 = arith.subf %3, %1 : f32
          %5 = math.exp %4 : f32
          affine.store %5, %alloca[%arg6, %arg7] : memref<8x8xf32>
          %6 = arith.addf %arg8, %5 : f32
          affine.yield %6 : f32
        }
        affine.for %arg7 = 0 to 8 {
          %3 = affine.load %alloca[%arg6, %arg7] : memref<8x8xf32>
          %4 = arith.divf %3, %2 : f32
          affine.store %4, %alloca[%arg6, %arg7] : memref<8x8xf32>
        }
      }
      affine.for %arg6 = 0 to 8 {
        affine.for %arg7 = 0 to 16 {
          %0 = affine.for %arg8 = 0 to 8 iter_args(%arg9 = %cst_0) -> (f32) {
            %1 = affine.load %alloca[%arg6, %arg8] : memref<8x8xf32>
            %2 = affine.load %alloca_1[%arg8, %arg7 + %arg5 * 16] : memref<8x256xf32>
            %3 = arith.mulf %1, %2 : f32
            %4 = arith.addf %arg9, %3 : f32
            affine.yield %4 : f32
          }
          affine.store %0, %arg4[%arg6, %arg7 + %arg5 * 16] : memref<8x256xf32>
        }
      }
    }
    return
  }
  func.func @_Z34grouped_mha_8_256_16_16_with_mha_4PA256_fS0_S0_S0_S0_(%arg0: memref<8x256xf32>, %arg1: memref<256x256xf32>, %arg2: memref<256x256xf32>, %arg3: memref<256x256xf32>, %arg4: memref<8x256xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.500000e-01 : f32
    %alloca = memref.alloca() {hls.preserve, polygeist.varname = "split_out"} : memref<8x64xf32>
    %alloca_1 = memref.alloca() {hls.preserve, polygeist.varname = "split_wv"} : memref<64x256xf32>
    %alloca_2 = memref.alloca() {hls.preserve, polygeist.varname = "split_wk"} : memref<64x256xf32>
    %alloca_3 = memref.alloca() {hls.preserve, polygeist.varname = "split_wq"} : memref<64x256xf32>
    %alloca_4 = memref.alloca() {hls.preserve, polygeist.varname = "scores"} : memref<8x8xf32>
    %alloca_5 = memref.alloca() {hls.preserve, polygeist.varname = "V"} : memref<8x64xf32>
    %alloca_6 = memref.alloca() {hls.preserve, polygeist.varname = "K"} : memref<8x64xf32>
    %alloca_7 = memref.alloca() {hls.preserve, polygeist.varname = "Q"} : memref<8x64xf32>
    affine.for %arg5 = 0 to 4 {
      affine.for %arg6 = 0 to 64 {
        affine.for %arg7 = 0 to 256 {
          %0 = affine.load %arg1[%arg6 + %arg5 * 64, %arg7] : memref<256x256xf32>
          affine.store %0, %alloca_3[%arg6, %arg7] : memref<64x256xf32>
          %1 = affine.load %arg2[%arg6 + %arg5 * 64, %arg7] : memref<256x256xf32>
          affine.store %1, %alloca_2[%arg6, %arg7] : memref<64x256xf32>
          %2 = affine.load %arg3[%arg6 + %arg5 * 64, %arg7] : memref<256x256xf32>
          affine.store %2, %alloca_1[%arg6, %arg7] : memref<64x256xf32>
        }
      }
      affine.for %arg6 = 0 to 8 {
        affine.for %arg7 = 0 to 64 {
          affine.store %cst, %alloca_7[%arg6, %arg7] : memref<8x64xf32>
          affine.store %cst, %alloca_6[%arg6, %arg7] : memref<8x64xf32>
          affine.store %cst, %alloca_5[%arg6, %arg7] : memref<8x64xf32>
          %0:3 = affine.for %arg8 = 0 to 256 iter_args(%arg9 = %cst, %arg10 = %cst, %arg11 = %cst) -> (f32, f32, f32) {
            %1 = affine.load %arg0[%arg6, %arg8] : memref<8x256xf32>
            %2 = affine.load %alloca_3[%arg7, %arg8] : memref<64x256xf32>
            %3 = arith.mulf %1, %2 : f32
            %4 = arith.addf %arg11, %3 : f32
            affine.store %4, %alloca_7[%arg6, %arg7] : memref<8x64xf32>
            %5 = affine.load %arg0[%arg6, %arg8] : memref<8x256xf32>
            %6 = affine.load %alloca_2[%arg7, %arg8] : memref<64x256xf32>
            %7 = arith.mulf %5, %6 : f32
            %8 = arith.addf %arg10, %7 : f32
            affine.store %8, %alloca_6[%arg6, %arg7] : memref<8x64xf32>
            %9 = affine.load %arg0[%arg6, %arg8] : memref<8x256xf32>
            %10 = affine.load %alloca_1[%arg7, %arg8] : memref<64x256xf32>
            %11 = arith.mulf %9, %10 : f32
            %12 = arith.addf %arg9, %11 : f32
            affine.store %12, %alloca_5[%arg6, %arg7] : memref<8x64xf32>
            affine.yield %12, %8, %4 : f32, f32, f32
          }
        }
      }
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 8 {
          affine.for %arg8 = 0 to 8 {
            affine.store %cst, %alloca_4[%arg7, %arg8] : memref<8x8xf32>
            %0 = affine.for %arg9 = 0 to 16 iter_args(%arg10 = %cst) -> (f32) {
              %3 = affine.load %alloca_7[%arg7, %arg9 + %arg6 * 16] : memref<8x64xf32>
              %4 = affine.load %alloca_6[%arg8, %arg9 + %arg6 * 16] : memref<8x64xf32>
              %5 = arith.mulf %3, %4 : f32
              %6 = arith.addf %arg10, %5 : f32
              affine.store %6, %alloca_4[%arg7, %arg8] : memref<8x8xf32>
              affine.yield %6 : f32
            }
            %1 = affine.load %alloca_4[%arg7, %arg8] : memref<8x8xf32>
            %2 = arith.mulf %1, %cst_0 : f32
            affine.store %2, %alloca_4[%arg7, %arg8] : memref<8x8xf32>
          }
        }
        affine.for %arg7 = 0 to 8 {
          %0 = affine.load %alloca_4[%arg7, 0] : memref<8x8xf32>
          %1 = affine.for %arg8 = 1 to 8 iter_args(%arg9 = %0) -> (f32) {
            %3 = affine.load %alloca_4[%arg7, %arg8] : memref<8x8xf32>
            %4 = arith.cmpf ogt, %3, %arg9 : f32
            %5 = arith.select %4, %3, %arg9 : f32
            affine.yield %5 : f32
          }
          %2 = affine.for %arg8 = 0 to 8 iter_args(%arg9 = %cst) -> (f32) {
            %3 = affine.load %alloca_4[%arg7, %arg8] : memref<8x8xf32>
            %4 = arith.subf %3, %1 : f32
            %5 = math.exp %4 : f32
            affine.store %5, %alloca_4[%arg7, %arg8] : memref<8x8xf32>
            %6 = arith.addf %arg9, %5 : f32
            affine.yield %6 : f32
          }
          affine.for %arg8 = 0 to 8 {
            %3 = affine.load %alloca_4[%arg7, %arg8] : memref<8x8xf32>
            %4 = arith.divf %3, %2 : f32
            affine.store %4, %alloca_4[%arg7, %arg8] : memref<8x8xf32>
          }
        }
        affine.for %arg7 = 0 to 8 {
          affine.for %arg8 = 0 to 16 {
            %0 = affine.for %arg9 = 0 to 8 iter_args(%arg10 = %cst) -> (f32) {
              %1 = affine.load %alloca_4[%arg7, %arg9] : memref<8x8xf32>
              %2 = affine.load %alloca_5[%arg9, %arg8 + %arg6 * 16] : memref<8x64xf32>
              %3 = arith.mulf %1, %2 : f32
              %4 = arith.addf %arg10, %3 : f32
              affine.yield %4 : f32
            }
            affine.store %0, %alloca[%arg7, %arg8 + %arg6 * 16] : memref<8x64xf32>
          }
        }
      }
      affine.for %arg6 = 0 to 8 {
        affine.for %arg7 = 0 to 64 {
          %0 = affine.load %alloca[%arg7, %arg6] : memref<8x64xf32>
          affine.store %0, %arg4[%arg7, %arg6 + %arg5 * 64] : memref<8x256xf32>
        }
      }
    }
    return
  }
  func.func @_Z25store_8_64_ap_fixed_16_5_PA64_fS0_(%arg0: memref<8x64xf32>, %arg1: memref<8x64xf32>) {
    affine.for %arg2 = 0 to 8 {
      affine.for %arg3 = 0 to 64 {
        %0 = affine.load %arg0[%arg2, %arg3] : memref<8x64xf32>
        affine.store %0, %arg1[%arg2, %arg3] : memref<8x64xf32>
      }
    }
    return
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
  func.func @_Z5top_APA256_fS0_S0_S0_S0_(%arg0: memref<8x256xf32>, %arg1: memref<256x256xf32>, %arg2: memref<256x256xf32>, %arg3: memref<256x256xf32>, %arg4: memref<8x256xf32>) {
    %cst = arith.constant 2.500000e-01 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca() {hls.preserve, polygeist.varname = "BRAM_1"} : memref<8x256xf32>
    %alloca_1 = memref.alloca() {hls.preserve, polygeist.varname = "BRAM_weights_v"} : memref<256x256xf32>
    %alloca_2 = memref.alloca() {hls.preserve, polygeist.varname = "BRAM_weights_k"} : memref<256x256xf32>
    %alloca_3 = memref.alloca() {hls.preserve, polygeist.varname = "BRAM_weights_q"} : memref<256x256xf32>
    %alloca_4 = memref.alloca() {hls.preserve, polygeist.varname = "BRAM_attn_input"} : memref<8x256xf32>
    affine.for %arg5 = 0 to 8 {
      affine.for %arg6 = 0 to 256 {
        %0 = affine.load %arg0[%arg5, %arg6] : memref<8x256xf32>
        affine.store %0, %alloca_4[%arg5, %arg6] : memref<8x256xf32>
      }
    }
    affine.for %arg5 = 0 to 256 {
      affine.for %arg6 = 0 to 256 {
        %0 = affine.load %arg1[%arg5, %arg6] : memref<256x256xf32>
        affine.store %0, %alloca_3[%arg5, %arg6] : memref<256x256xf32>
      }
    }
    affine.for %arg5 = 0 to 256 {
      affine.for %arg6 = 0 to 256 {
        %0 = affine.load %arg2[%arg5, %arg6] : memref<256x256xf32>
        affine.store %0, %alloca_2[%arg5, %arg6] : memref<256x256xf32>
      }
    }
    affine.for %arg5 = 0 to 256 {
      affine.for %arg6 = 0 to 256 {
        %0 = affine.load %arg3[%arg5, %arg6] : memref<256x256xf32>
        affine.store %0, %alloca_1[%arg5, %arg6] : memref<256x256xf32>
      }
    }
    %alloca_5 = memref.alloca() {hls.preserve, polygeist.varname = "split_out"} : memref<8x64xf32>
    %alloca_6 = memref.alloca() {hls.preserve, polygeist.varname = "split_wv"} : memref<64x256xf32>
    %alloca_7 = memref.alloca() {hls.preserve, polygeist.varname = "split_wk"} : memref<64x256xf32>
    %alloca_8 = memref.alloca() {hls.preserve, polygeist.varname = "split_wq"} : memref<64x256xf32>
    %alloca_9 = memref.alloca() {hls.preserve, polygeist.varname = "scores"} : memref<8x8xf32>
    %alloca_10 = memref.alloca() {hls.preserve, polygeist.varname = "V"} : memref<8x64xf32>
    %alloca_11 = memref.alloca() {hls.preserve, polygeist.varname = "K"} : memref<8x64xf32>
    %alloca_12 = memref.alloca() {hls.preserve, polygeist.varname = "Q"} : memref<8x64xf32>
    affine.for %arg5 = 0 to 4 {
      affine.for %arg6 = 0 to 64 {
        affine.for %arg7 = 0 to 256 {
          %0 = affine.load %alloca_3[%arg6 + %arg5 * 64, %arg7] : memref<256x256xf32>
          affine.store %0, %alloca_8[%arg6, %arg7] : memref<64x256xf32>
          %1 = affine.load %alloca_2[%arg6 + %arg5 * 64, %arg7] : memref<256x256xf32>
          affine.store %1, %alloca_7[%arg6, %arg7] : memref<64x256xf32>
          %2 = affine.load %alloca_1[%arg6 + %arg5 * 64, %arg7] : memref<256x256xf32>
          affine.store %2, %alloca_6[%arg6, %arg7] : memref<64x256xf32>
        }
      }
      affine.for %arg6 = 0 to 8 {
        affine.for %arg7 = 0 to 64 {
          affine.store %cst_0, %alloca_12[%arg6, %arg7] : memref<8x64xf32>
          affine.store %cst_0, %alloca_11[%arg6, %arg7] : memref<8x64xf32>
          affine.store %cst_0, %alloca_10[%arg6, %arg7] : memref<8x64xf32>
          %0:3 = affine.for %arg8 = 0 to 256 iter_args(%arg9 = %cst_0, %arg10 = %cst_0, %arg11 = %cst_0) -> (f32, f32, f32) {
            %1 = affine.load %alloca_4[%arg6, %arg8] : memref<8x256xf32>
            %2 = affine.load %alloca_8[%arg7, %arg8] : memref<64x256xf32>
            %3 = arith.mulf %1, %2 : f32
            %4 = arith.addf %arg11, %3 : f32
            affine.store %4, %alloca_12[%arg6, %arg7] : memref<8x64xf32>
            %5 = affine.load %alloca_7[%arg7, %arg8] : memref<64x256xf32>
            %6 = arith.mulf %1, %5 : f32
            %7 = arith.addf %arg10, %6 : f32
            affine.store %7, %alloca_11[%arg6, %arg7] : memref<8x64xf32>
            %8 = affine.load %alloca_6[%arg7, %arg8] : memref<64x256xf32>
            %9 = arith.mulf %1, %8 : f32
            %10 = arith.addf %arg9, %9 : f32
            affine.store %10, %alloca_10[%arg6, %arg7] : memref<8x64xf32>
            affine.yield %10, %7, %4 : f32, f32, f32
          }
        }
      }
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 8 {
          affine.for %arg8 = 0 to 8 {
            affine.store %cst_0, %alloca_9[%arg7, %arg8] : memref<8x8xf32>
            %0 = affine.for %arg9 = 0 to 16 iter_args(%arg10 = %cst_0) -> (f32) {
              %3 = affine.load %alloca_12[%arg7, %arg9 + %arg6 * 16] : memref<8x64xf32>
              %4 = affine.load %alloca_11[%arg8, %arg9 + %arg6 * 16] : memref<8x64xf32>
              %5 = arith.mulf %3, %4 : f32
              %6 = arith.addf %arg10, %5 : f32
              affine.store %6, %alloca_9[%arg7, %arg8] : memref<8x8xf32>
              affine.yield %6 : f32
            }
            %1 = affine.load %alloca_9[%arg7, %arg8] : memref<8x8xf32>
            %2 = arith.mulf %1, %cst : f32
            affine.store %2, %alloca_9[%arg7, %arg8] : memref<8x8xf32>
          }
        }
        affine.for %arg7 = 0 to 8 {
          %0 = affine.load %alloca_9[%arg7, 0] : memref<8x8xf32>
          %1 = affine.for %arg8 = 1 to 8 iter_args(%arg9 = %0) -> (f32) {
            %3 = affine.load %alloca_9[%arg7, %arg8] : memref<8x8xf32>
            %4 = arith.cmpf ogt, %3, %arg9 : f32
            %5 = arith.select %4, %3, %arg9 : f32
            affine.yield %5 : f32
          }
          %2 = affine.for %arg8 = 0 to 8 iter_args(%arg9 = %cst_0) -> (f32) {
            %3 = affine.load %alloca_9[%arg7, %arg8] : memref<8x8xf32>
            %4 = arith.subf %3, %1 : f32
            %5 = math.exp %4 : f32
            affine.store %5, %alloca_9[%arg7, %arg8] : memref<8x8xf32>
            %6 = arith.addf %arg9, %5 : f32
            affine.yield %6 : f32
          }
          affine.for %arg8 = 0 to 8 {
            %3 = affine.load %alloca_9[%arg7, %arg8] : memref<8x8xf32>
            %4 = arith.divf %3, %2 : f32
            affine.store %4, %alloca_9[%arg7, %arg8] : memref<8x8xf32>
          }
        }
        affine.for %arg7 = 0 to 8 {
          affine.for %arg8 = 0 to 16 {
            %0 = affine.for %arg9 = 0 to 8 iter_args(%arg10 = %cst_0) -> (f32) {
              %1 = affine.load %alloca_9[%arg7, %arg9] : memref<8x8xf32>
              %2 = affine.load %alloca_10[%arg9, %arg8 + %arg6 * 16] : memref<8x64xf32>
              %3 = arith.mulf %1, %2 : f32
              %4 = arith.addf %arg10, %3 : f32
              affine.yield %4 : f32
            }
            affine.store %0, %alloca_5[%arg7, %arg8 + %arg6 * 16] : memref<8x64xf32>
          }
        }
      }
      affine.for %arg6 = 0 to 8 {
        affine.for %arg7 = 0 to 64 {
          %0 = affine.load %alloca_5[%arg7, %arg6] : memref<8x64xf32>
          affine.store %0, %alloca[%arg7, %arg6 + %arg5 * 64] : memref<8x256xf32>
        }
      }
    }
    affine.for %arg5 = 0 to 8 {
      affine.for %arg6 = 0 to 256 {
        %0 = affine.load %alloca[%arg5, %arg6] : memref<8x256xf32>
        affine.store %0, %arg4[%arg5, %arg6] : memref<8x256xf32>
      }
    }
    return
  }
  func.func @_Z5top_BPA256_fS0_S0_S0_PA64_f(%arg0: memref<8x256xf32>, %arg1: memref<64x256xf32>, %arg2: memref<64x256xf32>, %arg3: memref<64x256xf32>, %arg4: memref<8x64xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.500000e-01 : f32
    %alloca = memref.alloca() {hls.preserve, polygeist.varname = "BRAM_1"} : memref<8x64xf32>
    %alloca_1 = memref.alloca() {hls.preserve, polygeist.varname = "BRAM_weights_v"} : memref<64x256xf32>
    %alloca_2 = memref.alloca() {hls.preserve, polygeist.varname = "BRAM_weights_k"} : memref<64x256xf32>
    %alloca_3 = memref.alloca() {hls.preserve, polygeist.varname = "BRAM_weights_q"} : memref<64x256xf32>
    %alloca_4 = memref.alloca() {hls.preserve, polygeist.varname = "BRAM_attn_input"} : memref<8x256xf32>
    affine.for %arg5 = 0 to 8 {
      affine.for %arg6 = 0 to 256 {
        %0 = affine.load %arg0[%arg5, %arg6] : memref<8x256xf32>
        affine.store %0, %alloca_4[%arg5, %arg6] : memref<8x256xf32>
      }
    }
    affine.for %arg5 = 0 to 64 {
      affine.for %arg6 = 0 to 256 {
        %0 = affine.load %arg1[%arg5, %arg6] : memref<64x256xf32>
        affine.store %0, %alloca_3[%arg5, %arg6] : memref<64x256xf32>
      }
    }
    affine.for %arg5 = 0 to 64 {
      affine.for %arg6 = 0 to 256 {
        %0 = affine.load %arg2[%arg5, %arg6] : memref<64x256xf32>
        affine.store %0, %alloca_2[%arg5, %arg6] : memref<64x256xf32>
      }
    }
    affine.for %arg5 = 0 to 64 {
      affine.for %arg6 = 0 to 256 {
        %0 = affine.load %arg3[%arg5, %arg6] : memref<64x256xf32>
        affine.store %0, %alloca_1[%arg5, %arg6] : memref<64x256xf32>
      }
    }
    %alloca_5 = memref.alloca() {hls.preserve, polygeist.varname = "scores"} : memref<8x8xf32>
    %alloca_6 = memref.alloca() {hls.preserve, polygeist.varname = "V"} : memref<8x64xf32>
    %alloca_7 = memref.alloca() {hls.preserve, polygeist.varname = "K"} : memref<8x64xf32>
    %alloca_8 = memref.alloca() {hls.preserve, polygeist.varname = "Q"} : memref<8x64xf32>
    affine.for %arg5 = 0 to 8 {
      affine.for %arg6 = 0 to 64 {
        affine.store %cst, %alloca_8[%arg5, %arg6] : memref<8x64xf32>
        affine.store %cst, %alloca_7[%arg5, %arg6] : memref<8x64xf32>
        affine.store %cst, %alloca_6[%arg5, %arg6] : memref<8x64xf32>
        %0:3 = affine.for %arg7 = 0 to 256 iter_args(%arg8 = %cst, %arg9 = %cst, %arg10 = %cst) -> (f32, f32, f32) {
          %1 = affine.load %alloca_4[%arg5, %arg7] : memref<8x256xf32>
          %2 = affine.load %alloca_3[%arg6, %arg7] : memref<64x256xf32>
          %3 = arith.mulf %1, %2 : f32
          %4 = arith.addf %arg10, %3 : f32
          affine.store %4, %alloca_8[%arg5, %arg6] : memref<8x64xf32>
          %5 = affine.load %alloca_2[%arg6, %arg7] : memref<64x256xf32>
          %6 = arith.mulf %1, %5 : f32
          %7 = arith.addf %arg9, %6 : f32
          affine.store %7, %alloca_7[%arg5, %arg6] : memref<8x64xf32>
          %8 = affine.load %alloca_1[%arg6, %arg7] : memref<64x256xf32>
          %9 = arith.mulf %1, %8 : f32
          %10 = arith.addf %arg8, %9 : f32
          affine.store %10, %alloca_6[%arg5, %arg6] : memref<8x64xf32>
          affine.yield %10, %7, %4 : f32, f32, f32
        }
      }
    }
    affine.for %arg5 = 0 to 4 {
      affine.for %arg6 = 0 to 8 {
        affine.for %arg7 = 0 to 8 {
          affine.store %cst, %alloca_5[%arg6, %arg7] : memref<8x8xf32>
          %0 = affine.for %arg8 = 0 to 16 iter_args(%arg9 = %cst) -> (f32) {
            %3 = affine.load %alloca_8[%arg6, %arg8 + %arg5 * 16] : memref<8x64xf32>
            %4 = affine.load %alloca_7[%arg7, %arg8 + %arg5 * 16] : memref<8x64xf32>
            %5 = arith.mulf %3, %4 : f32
            %6 = arith.addf %arg9, %5 : f32
            affine.store %6, %alloca_5[%arg6, %arg7] : memref<8x8xf32>
            affine.yield %6 : f32
          }
          %1 = affine.load %alloca_5[%arg6, %arg7] : memref<8x8xf32>
          %2 = arith.mulf %1, %cst_0 : f32
          affine.store %2, %alloca_5[%arg6, %arg7] : memref<8x8xf32>
        }
      }
      affine.for %arg6 = 0 to 8 {
        %0 = affine.load %alloca_5[%arg6, 0] : memref<8x8xf32>
        %1 = affine.for %arg7 = 1 to 8 iter_args(%arg8 = %0) -> (f32) {
          %3 = affine.load %alloca_5[%arg6, %arg7] : memref<8x8xf32>
          %4 = arith.cmpf ogt, %3, %arg8 : f32
          %5 = arith.select %4, %3, %arg8 : f32
          affine.yield %5 : f32
        }
        %2 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %cst) -> (f32) {
          %3 = affine.load %alloca_5[%arg6, %arg7] : memref<8x8xf32>
          %4 = arith.subf %3, %1 : f32
          %5 = math.exp %4 : f32
          affine.store %5, %alloca_5[%arg6, %arg7] : memref<8x8xf32>
          %6 = arith.addf %arg8, %5 : f32
          affine.yield %6 : f32
        }
        affine.for %arg7 = 0 to 8 {
          %3 = affine.load %alloca_5[%arg6, %arg7] : memref<8x8xf32>
          %4 = arith.divf %3, %2 : f32
          affine.store %4, %alloca_5[%arg6, %arg7] : memref<8x8xf32>
        }
      }
      affine.for %arg6 = 0 to 8 {
        affine.for %arg7 = 0 to 16 {
          %0 = affine.for %arg8 = 0 to 8 iter_args(%arg9 = %cst) -> (f32) {
            %1 = affine.load %alloca_5[%arg6, %arg8] : memref<8x8xf32>
            %2 = affine.load %alloca_6[%arg8, %arg7 + %arg5 * 16] : memref<8x64xf32>
            %3 = arith.mulf %1, %2 : f32
            %4 = arith.addf %arg9, %3 : f32
            affine.yield %4 : f32
          }
          affine.store %0, %alloca[%arg6, %arg7 + %arg5 * 16] : memref<8x64xf32>
        }
      }
    }
    affine.for %arg5 = 0 to 8 {
      affine.for %arg6 = 0 to 64 {
        %0 = affine.load %alloca[%arg5, %arg6] : memref<8x64xf32>
        affine.store %0, %arg4[%arg5, %arg6] : memref<8x64xf32>
      }
    }
    return
  }
  func.func @_Z3topPA256_fS0_S0_S0_S0_S0_S0_S0_S0_PA64_f(%arg0: memref<8x256xf32>, %arg1: memref<256x256xf32>, %arg2: memref<256x256xf32>, %arg3: memref<256x256xf32>, %arg4: memref<8x256xf32>, %arg5: memref<8x256xf32>, %arg6: memref<64x256xf32>, %arg7: memref<64x256xf32>, %arg8: memref<64x256xf32>, %arg9: memref<8x64xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.500000e-01 : f32
    %alloca = memref.alloca() {hls.preserve, polygeist.varname = "BRAM_B5"} : memref<8x64xf32>
    %alloca_1 = memref.alloca() {hls.preserve, polygeist.varname = "BRAM_B4"} : memref<64x256xf32>
    %alloca_2 = memref.alloca() {hls.preserve, polygeist.varname = "BRAM_B3"} : memref<64x256xf32>
    %alloca_3 = memref.alloca() {hls.preserve, polygeist.varname = "BRAM_B2"} : memref<64x256xf32>
    %alloca_4 = memref.alloca() {hls.preserve, polygeist.varname = "BRAM_B1"} : memref<8x256xf32>
    %alloca_5 = memref.alloca() {hls.preserve, polygeist.varname = "split_out"} : memref<8x64xf32>
    %alloca_6 = memref.alloca() {hls.preserve, polygeist.varname = "split_wv"} : memref<64x256xf32>
    %alloca_7 = memref.alloca() {hls.preserve, polygeist.varname = "split_wk"} : memref<64x256xf32>
    %alloca_8 = memref.alloca() {hls.preserve, polygeist.varname = "split_wq"} : memref<64x256xf32>
    %alloca_9 = memref.alloca() {hls.preserve, polygeist.varname = "BRAM_A5"} : memref<8x256xf32>
    %alloca_10 = memref.alloca() {hls.preserve, polygeist.varname = "BRAM_A4"} : memref<256x256xf32>
    %alloca_11 = memref.alloca() {hls.preserve, polygeist.varname = "BRAM_A3"} : memref<256x256xf32>
    %alloca_12 = memref.alloca() {hls.preserve, polygeist.varname = "BRAM_A2"} : memref<256x256xf32>
    %alloca_13 = memref.alloca() {hls.preserve, polygeist.varname = "BRAM_A1"} : memref<8x256xf32>
    affine.for %arg10 = 0 to 8 {
      affine.for %arg11 = 0 to 256 {
        %0 = affine.load %arg0[%arg10, %arg11] : memref<8x256xf32>
        affine.store %0, %alloca_13[%arg10, %arg11] : memref<8x256xf32>
      }
    }
    affine.for %arg10 = 0 to 256 {
      affine.for %arg11 = 0 to 256 {
        %0 = affine.load %arg1[%arg10, %arg11] : memref<256x256xf32>
        affine.store %0, %alloca_12[%arg10, %arg11] : memref<256x256xf32>
      }
    }
    affine.for %arg10 = 0 to 256 {
      affine.for %arg11 = 0 to 256 {
        %0 = affine.load %arg2[%arg10, %arg11] : memref<256x256xf32>
        affine.store %0, %alloca_11[%arg10, %arg11] : memref<256x256xf32>
      }
    }
    affine.for %arg10 = 0 to 256 {
      affine.for %arg11 = 0 to 256 {
        %0 = affine.load %arg3[%arg10, %arg11] : memref<256x256xf32>
        affine.store %0, %alloca_10[%arg10, %arg11] : memref<256x256xf32>
      }
    }
    %alloca_14 = memref.alloca() {hls.preserve, polygeist.varname = "scores"} : memref<8x8xf32>
    %alloca_15 = memref.alloca() {hls.preserve, polygeist.varname = "V"} : memref<8x64xf32>
    %alloca_16 = memref.alloca() {hls.preserve, polygeist.varname = "K"} : memref<8x64xf32>
    %alloca_17 = memref.alloca() {hls.preserve, polygeist.varname = "Q"} : memref<8x64xf32>
    affine.for %arg10 = 0 to 4 {
      affine.for %arg11 = 0 to 64 {
        affine.for %arg12 = 0 to 256 {
          %0 = affine.load %alloca_12[%arg11 + %arg10 * 64, %arg12] : memref<256x256xf32>
          affine.store %0, %alloca_8[%arg11, %arg12] : memref<64x256xf32>
          %1 = affine.load %alloca_11[%arg11 + %arg10 * 64, %arg12] : memref<256x256xf32>
          affine.store %1, %alloca_7[%arg11, %arg12] : memref<64x256xf32>
          %2 = affine.load %alloca_10[%arg11 + %arg10 * 64, %arg12] : memref<256x256xf32>
          affine.store %2, %alloca_6[%arg11, %arg12] : memref<64x256xf32>
        }
      }
      affine.for %arg11 = 0 to 8 {
        affine.for %arg12 = 0 to 64 {
          affine.store %cst, %alloca_17[%arg11, %arg12] : memref<8x64xf32>
          affine.store %cst, %alloca_16[%arg11, %arg12] : memref<8x64xf32>
          affine.store %cst, %alloca_15[%arg11, %arg12] : memref<8x64xf32>
          %0:3 = affine.for %arg13 = 0 to 256 iter_args(%arg14 = %cst, %arg15 = %cst, %arg16 = %cst) -> (f32, f32, f32) {
            %1 = affine.load %alloca_13[%arg11, %arg13] : memref<8x256xf32>
            %2 = affine.load %alloca_8[%arg12, %arg13] : memref<64x256xf32>
            %3 = arith.mulf %1, %2 : f32
            %4 = arith.addf %arg16, %3 : f32
            affine.store %4, %alloca_17[%arg11, %arg12] : memref<8x64xf32>
            %5 = affine.load %alloca_7[%arg12, %arg13] : memref<64x256xf32>
            %6 = arith.mulf %1, %5 : f32
            %7 = arith.addf %arg15, %6 : f32
            affine.store %7, %alloca_16[%arg11, %arg12] : memref<8x64xf32>
            %8 = affine.load %alloca_6[%arg12, %arg13] : memref<64x256xf32>
            %9 = arith.mulf %1, %8 : f32
            %10 = arith.addf %arg14, %9 : f32
            affine.store %10, %alloca_15[%arg11, %arg12] : memref<8x64xf32>
            affine.yield %10, %7, %4 : f32, f32, f32
          }
        }
      }
      affine.for %arg11 = 0 to 4 {
        affine.for %arg12 = 0 to 8 {
          affine.for %arg13 = 0 to 8 {
            affine.store %cst, %alloca_14[%arg12, %arg13] : memref<8x8xf32>
            %0 = affine.for %arg14 = 0 to 16 iter_args(%arg15 = %cst) -> (f32) {
              %3 = affine.load %alloca_17[%arg12, %arg14 + %arg11 * 16] : memref<8x64xf32>
              %4 = affine.load %alloca_16[%arg13, %arg14 + %arg11 * 16] : memref<8x64xf32>
              %5 = arith.mulf %3, %4 : f32
              %6 = arith.addf %arg15, %5 : f32
              affine.store %6, %alloca_14[%arg12, %arg13] : memref<8x8xf32>
              affine.yield %6 : f32
            }
            %1 = affine.load %alloca_14[%arg12, %arg13] : memref<8x8xf32>
            %2 = arith.mulf %1, %cst_0 : f32
            affine.store %2, %alloca_14[%arg12, %arg13] : memref<8x8xf32>
          }
        }
        affine.for %arg12 = 0 to 8 {
          %0 = affine.load %alloca_14[%arg12, 0] : memref<8x8xf32>
          %1 = affine.for %arg13 = 1 to 8 iter_args(%arg14 = %0) -> (f32) {
            %3 = affine.load %alloca_14[%arg12, %arg13] : memref<8x8xf32>
            %4 = arith.cmpf ogt, %3, %arg14 : f32
            %5 = arith.select %4, %3, %arg14 : f32
            affine.yield %5 : f32
          }
          %2 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst) -> (f32) {
            %3 = affine.load %alloca_14[%arg12, %arg13] : memref<8x8xf32>
            %4 = arith.subf %3, %1 : f32
            %5 = math.exp %4 : f32
            affine.store %5, %alloca_14[%arg12, %arg13] : memref<8x8xf32>
            %6 = arith.addf %arg14, %5 : f32
            affine.yield %6 : f32
          }
          affine.for %arg13 = 0 to 8 {
            %3 = affine.load %alloca_14[%arg12, %arg13] : memref<8x8xf32>
            %4 = arith.divf %3, %2 : f32
            affine.store %4, %alloca_14[%arg12, %arg13] : memref<8x8xf32>
          }
        }
        affine.for %arg12 = 0 to 8 {
          affine.for %arg13 = 0 to 16 {
            %0 = affine.for %arg14 = 0 to 8 iter_args(%arg15 = %cst) -> (f32) {
              %1 = affine.load %alloca_14[%arg12, %arg14] : memref<8x8xf32>
              %2 = affine.load %alloca_15[%arg14, %arg13 + %arg11 * 16] : memref<8x64xf32>
              %3 = arith.mulf %1, %2 : f32
              %4 = arith.addf %arg15, %3 : f32
              affine.yield %4 : f32
            }
            affine.store %0, %alloca_5[%arg12, %arg13 + %arg11 * 16] : memref<8x64xf32>
          }
        }
      }
      affine.for %arg11 = 0 to 8 {
        affine.for %arg12 = 0 to 64 {
          %0 = affine.load %alloca_5[%arg12, %arg11] : memref<8x64xf32>
          affine.store %0, %alloca_9[%arg12, %arg11 + %arg10 * 64] : memref<8x256xf32>
        }
      }
    }
    affine.for %arg10 = 0 to 8 {
      affine.for %arg11 = 0 to 256 {
        %0 = affine.load %alloca_9[%arg10, %arg11] : memref<8x256xf32>
        affine.store %0, %arg4[%arg10, %arg11] : memref<8x256xf32>
      }
    }
    affine.for %arg10 = 0 to 8 {
      affine.for %arg11 = 0 to 256 {
        %0 = affine.load %arg5[%arg10, %arg11] : memref<8x256xf32>
        affine.store %0, %alloca_4[%arg10, %arg11] : memref<8x256xf32>
      }
    }
    affine.for %arg10 = 0 to 64 {
      affine.for %arg11 = 0 to 256 {
        %0 = affine.load %arg6[%arg10, %arg11] : memref<64x256xf32>
        affine.store %0, %alloca_3[%arg10, %arg11] : memref<64x256xf32>
      }
    }
    affine.for %arg10 = 0 to 64 {
      affine.for %arg11 = 0 to 256 {
        %0 = affine.load %arg7[%arg10, %arg11] : memref<64x256xf32>
        affine.store %0, %alloca_2[%arg10, %arg11] : memref<64x256xf32>
      }
    }
    affine.for %arg10 = 0 to 64 {
      affine.for %arg11 = 0 to 256 {
        %0 = affine.load %arg8[%arg10, %arg11] : memref<64x256xf32>
        affine.store %0, %alloca_1[%arg10, %arg11] : memref<64x256xf32>
      }
    }
    %alloca_18 = memref.alloca() {hls.preserve, polygeist.varname = "scores"} : memref<8x8xf32>
    %alloca_19 = memref.alloca() {hls.preserve, polygeist.varname = "V"} : memref<8x64xf32>
    %alloca_20 = memref.alloca() {hls.preserve, polygeist.varname = "K"} : memref<8x64xf32>
    %alloca_21 = memref.alloca() {hls.preserve, polygeist.varname = "Q"} : memref<8x64xf32>
    affine.for %arg10 = 0 to 8 {
      affine.for %arg11 = 0 to 64 {
        affine.store %cst, %alloca_21[%arg10, %arg11] : memref<8x64xf32>
        affine.store %cst, %alloca_20[%arg10, %arg11] : memref<8x64xf32>
        affine.store %cst, %alloca_19[%arg10, %arg11] : memref<8x64xf32>
        %0:3 = affine.for %arg12 = 0 to 256 iter_args(%arg13 = %cst, %arg14 = %cst, %arg15 = %cst) -> (f32, f32, f32) {
          %1 = affine.load %alloca_4[%arg10, %arg12] : memref<8x256xf32>
          %2 = affine.load %alloca_3[%arg11, %arg12] : memref<64x256xf32>
          %3 = arith.mulf %1, %2 : f32
          %4 = arith.addf %arg15, %3 : f32
          affine.store %4, %alloca_21[%arg10, %arg11] : memref<8x64xf32>
          %5 = affine.load %alloca_2[%arg11, %arg12] : memref<64x256xf32>
          %6 = arith.mulf %1, %5 : f32
          %7 = arith.addf %arg14, %6 : f32
          affine.store %7, %alloca_20[%arg10, %arg11] : memref<8x64xf32>
          %8 = affine.load %alloca_1[%arg11, %arg12] : memref<64x256xf32>
          %9 = arith.mulf %1, %8 : f32
          %10 = arith.addf %arg13, %9 : f32
          affine.store %10, %alloca_19[%arg10, %arg11] : memref<8x64xf32>
          affine.yield %10, %7, %4 : f32, f32, f32
        }
      }
    }
    affine.for %arg10 = 0 to 4 {
      affine.for %arg11 = 0 to 8 {
        affine.for %arg12 = 0 to 8 {
          affine.store %cst, %alloca_18[%arg11, %arg12] : memref<8x8xf32>
          %0 = affine.for %arg13 = 0 to 16 iter_args(%arg14 = %cst) -> (f32) {
            %3 = affine.load %alloca_21[%arg11, %arg13 + %arg10 * 16] : memref<8x64xf32>
            %4 = affine.load %alloca_20[%arg12, %arg13 + %arg10 * 16] : memref<8x64xf32>
            %5 = arith.mulf %3, %4 : f32
            %6 = arith.addf %arg14, %5 : f32
            affine.store %6, %alloca_18[%arg11, %arg12] : memref<8x8xf32>
            affine.yield %6 : f32
          }
          %1 = affine.load %alloca_18[%arg11, %arg12] : memref<8x8xf32>
          %2 = arith.mulf %1, %cst_0 : f32
          affine.store %2, %alloca_18[%arg11, %arg12] : memref<8x8xf32>
        }
      }
      affine.for %arg11 = 0 to 8 {
        %0 = affine.load %alloca_18[%arg11, 0] : memref<8x8xf32>
        %1 = affine.for %arg12 = 1 to 8 iter_args(%arg13 = %0) -> (f32) {
          %3 = affine.load %alloca_18[%arg11, %arg12] : memref<8x8xf32>
          %4 = arith.cmpf ogt, %3, %arg13 : f32
          %5 = arith.select %4, %3, %arg13 : f32
          affine.yield %5 : f32
        }
        %2 = affine.for %arg12 = 0 to 8 iter_args(%arg13 = %cst) -> (f32) {
          %3 = affine.load %alloca_18[%arg11, %arg12] : memref<8x8xf32>
          %4 = arith.subf %3, %1 : f32
          %5 = math.exp %4 : f32
          affine.store %5, %alloca_18[%arg11, %arg12] : memref<8x8xf32>
          %6 = arith.addf %arg13, %5 : f32
          affine.yield %6 : f32
        }
        affine.for %arg12 = 0 to 8 {
          %3 = affine.load %alloca_18[%arg11, %arg12] : memref<8x8xf32>
          %4 = arith.divf %3, %2 : f32
          affine.store %4, %alloca_18[%arg11, %arg12] : memref<8x8xf32>
        }
      }
      affine.for %arg11 = 0 to 8 {
        affine.for %arg12 = 0 to 16 {
          %0 = affine.for %arg13 = 0 to 8 iter_args(%arg14 = %cst) -> (f32) {
            %1 = affine.load %alloca_18[%arg11, %arg13] : memref<8x8xf32>
            %2 = affine.load %alloca_19[%arg13, %arg12 + %arg10 * 16] : memref<8x64xf32>
            %3 = arith.mulf %1, %2 : f32
            %4 = arith.addf %arg14, %3 : f32
            affine.yield %4 : f32
          }
          affine.store %0, %alloca[%arg11, %arg12 + %arg10 * 16] : memref<8x64xf32>
        }
      }
    }
    affine.for %arg10 = 0 to 8 {
      affine.for %arg11 = 0 to 64 {
        %0 = affine.load %alloca[%arg10, %arg11] : memref<8x64xf32>
        affine.store %0, %arg9[%arg10, %arg11] : memref<8x64xf32>
      }
    }
    return
  }
}


// Kernel Source:
// #include <stdio.h>
// #include <iostream>
// #include <fstream>
// #include <cstdlib>
// #include <ap_fixed.h>
// #include <hls_math.h>
// #include <stdlib.h>
// #include <cstdint>
// #include <hls_math.h>
// using namespace std;
// 
// typedef ap_fixed<16, 5> data_t;
// 
// data_t BRAM_attn_input[8][256];
// data_t BRAM_weights_q[64][256];
// data_t BRAM_weights_k[64][256];
// data_t BRAM_weights_v[64][256];
// data_t BRAM_1[8][64];
// 
// void load_8_256_ap_fixed_16_5_(data_t input[8][256], data_t output[8][256])
// {
//     #pragma HLS inline off
//     for (int idx0 = 0; idx0 < 8; idx0++) {
//         for (int idx1 = 0; idx1 < 256; idx1++) {
//             output[idx0][idx1] = input[idx0][idx1];
//         }
//     }
// }
// 
// void load_64_256_ap_fixed_16_5_(data_t input[64][256], data_t output[64][256])
// {
//     #pragma HLS inline off
//     for (int idx0 = 0; idx0 < 64; idx0++) {
//         for (int idx1 = 0; idx1 < 256; idx1++) {
//             output[idx0][idx1] = input[idx0][idx1];
//         }
//     }
// }
// 
// void load_256_256_ap_fixed_16_5_(data_t input[256][256], data_t output[256][256])
// {
//     #pragma HLS inline off
//     for (int idx0 = 0; idx0 < 256; idx0++) {
//         for (int idx1 = 0; idx1 < 256; idx1++) {
//             output[idx0][idx1] = input[idx0][idx1];
//         }
//     }
// }
// 
// /*
//  * Auto-generated Grouped Multi-head Attention (with optional inline RoPE)
//  *
//  * Input     : [8][256]
//  * W_q/k/v   : [64][256], DIM_OUT = NUM_HEADS * HEAD_DIM
//  * Output    : [8][64]
//  *
//  * Data type : ap_fixed<16, 5>
//  * Num Heads : 4
//  * Head Dim  : 16
//  */
// 
//  void transpose_64(
//     ap_fixed<16, 5> input[64][64]
// )
// {
//     #pragma HLS inline off
// for (int i = 0; i < 64; i++) {
// for (int j = 0; j < 64; j++) {
//     input[i][j] = input[j][i];
// }
// }
// }
// 
// void transpoe_256(
//     ap_fixed<16, 5> input[256][256]
// )
// {
//     #pragma HLS inline off
// for (int i = 0; i < 256; i++) {
// for (int j = 0; j < 256; j++) {
//     input[i][j] = input[j][i];
// }
// }   
// }
// 
// 
//  void grouped_multihead_attention_8_256_4_16_ap_fixed_16_5_(
//     data_t input[8][256],
//     data_t W_q[64][256],
//     data_t W_k[64][256],
//     data_t W_v[64][256],
//     data_t output[8][64]  // for trasose purposes
// )
// {
//     #pragma HLS inline off
//     const int groups = 4;
//     const int num_heads = 4;   // total number of heads (must equal DIM_OUT / HEAD_DIM)
//     const int head_dim = 16;       // dimension per head
//     const int heads_per_group = num_heads / groups;
//     const data_t scale = (data_t)1.0 / hls::sqrt((data_t)head_dim);
// 
//     data_t Q[8][64];
//     data_t K[8][64];
//     data_t V[8][64];
// 
//     // Compute Q, K, V
//     for (int seq = 0; seq < 8; seq++) {
//         for (int dout = 0; dout < 64; dout++) {
//             Q[seq][dout] = 0;
//             K[seq][dout] = 0;
//             V[seq][dout] = 0;
//             for (int din = 0; din < 256; din++) {
//                 Q[seq][dout] += input[seq][din] * W_q[dout][din];
//                 K[seq][dout] += input[seq][din] * W_k[dout][din];
//                 V[seq][dout] += input[seq][din] * W_v[dout][din];
//             }
//         }
//     }
// 
//     /*==== BEGIN OPTIONAL ROPE LOGIC ====*/
//     
//     /*==== END OPTIONAL ROPE LOGIC ====*/
// 
//     // Compute Attention per head in groups.
//     for (int g = 0; g < groups; g++) {
//         for (int h = 0; h < heads_per_group; h++) {
//             int head_index = g * heads_per_group + h;
//             data_t scores[8][8];
// 
//             // Scaled Dot-product: Q x K^T for head head_index
//             for (int i = 0; i < 8; i++) {
//                 for (int j = 0; j < 8; j++) {
//                     scores[i][j] = 0;
//                     for (int d = 0; d < head_dim; d++) {
//                         int idx = head_index * head_dim + d;
//                         scores[i][j] += Q[i][idx] * K[j][idx];
//                     }
//                     scores[i][j] *= scale;
//                 }
//             }
// 
//             // Softmax for this head.
//             for (int i = 0; i < 8; i++) {
//                 data_t sum_exp = 0;
//                 data_t max_score = scores[i][0];
//                 for (int j = 1; j < 8; j++) {
//                     if (scores[i][j] > max_score)
//                         max_score = scores[i][j];
//                 }
//                 for (int j = 0; j < 8; j++) {
//                     scores[i][j] = hls::exp(scores[i][j] - max_score);
//                     sum_exp += scores[i][j];
//                 }
//                 for (int j = 0; j < 8; j++) {
//                     scores[i][j] /= sum_exp;
//                 }
//             }
// 
//             // Compute context: scores x V for head head_index.
//             for (int i = 0; i < 8; i++) {
//                 for (int d = 0; d < head_dim; d++) {
//                     data_t context = 0;
//                     for (int j = 0; j < 8; j++) {
//                         context += scores[i][j] * V[j][head_index * head_dim + d];
//                     }
//                     output[i][head_index * head_dim + d] = context;
//                 }
//             }
//         }
//     }
// }
// 
// void grouped_multihead_attention_8_256_16_16_ap_fixed_16_5_(
//     data_t input[8][256],
//     data_t W_q[256][256],
//     data_t W_k[256][256],
//     data_t W_v[256][256],
//     data_t output[8][256]
// )
// {
//     #pragma HLS inline off
//     const int groups = 16; // number of groups (must divide num_heads evenly)
//     const int num_heads = 16;   // total number of heads (must equal DIM_OUT / HEAD_DIM)
//     const int head_dim = 16;       // dimension per head
//     const int heads_per_group = num_heads / groups;
//     const data_t scale = (data_t)1.0 / hls::sqrt((data_t)head_dim);
// 
//     data_t Q[8][256];
//     data_t K[8][256];
//     data_t V[8][256];
// 
//     // Compute Q, K, V
//     for (int seq = 0; seq < 8; seq++) {
//         for (int dout = 0; dout < 256; dout++) {
//             Q[seq][dout] = 0;
//             K[seq][dout] = 0;
//             V[seq][dout] = 0;
//             for (int din = 0; din < 256; din++) {
//                 Q[seq][dout] += input[seq][din] * W_q[dout][din];
//                 K[seq][dout] += input[seq][din] * W_k[dout][din];
//                 V[seq][dout] += input[seq][din] * W_v[dout][din];
//             }
//         }
//     }
// 
//     /*==== BEGIN OPTIONAL ROPE LOGIC ====*/
//     
//     /*==== END OPTIONAL ROPE LOGIC ====*/
// 
//     // Compute Attention per head in groups.
//     for (int g = 0; g < groups; g++) {
//         for (int h = 0; h < heads_per_group; h++) {
//             int head_index = g * heads_per_group + h;
//             data_t scores[8][8];
// 
//             // Scaled Dot-product: Q x K^T for head head_index
//             for (int i = 0; i < 8; i++) {
//                 for (int j = 0; j < 8; j++) {
//                     scores[i][j] = 0;
//                     for (int d = 0; d < head_dim; d++) {
//                         int idx = head_index * head_dim + d;
//                         scores[i][j] += Q[i][idx] * K[j][idx];
//                     }
//                     scores[i][j] *= scale;
//                 }
//             }
// 
//             // Softmax for this head.
//             for (int i = 0; i < 8; i++) {
//                 data_t sum_exp = 0;
//                 data_t max_score = scores[i][0];
//                 for (int j = 1; j < 8; j++) {
//                     if (scores[i][j] > max_score)
//                         max_score = scores[i][j];
//                 }
//                 for (int j = 0; j < 8; j++) {
//                     scores[i][j] = hls::exp(scores[i][j] - max_score);
//                     sum_exp += scores[i][j];
//                 }
//                 for (int j = 0; j < 8; j++) {
//                     scores[i][j] /= sum_exp;
//                 }
//             }
// 
//             // Compute context: scores x V for head head_index.
//             for (int i = 0; i < 8; i++) {
//                 for (int d = 0; d < head_dim; d++) {
//                     data_t context = 0;
//                     for (int j = 0; j < 8; j++) {
//                         context += scores[i][j] * V[j][head_index * head_dim + d];
//                     }
//                     output[i][head_index * head_dim + d] = context;
//                 }
//             }
//         }
//     }
// }
// 
// void grouped_mha_8_256_16_16_with_mha_4(
//     data_t input[8][256],
//     data_t W_q[256][256],
//     data_t W_k[256][256],
//     data_t W_v[256][256],
//     data_t output[8][256]
// )
// {
// 
// 
//     #pragma HLS inline off
//     #pragma HLS allocation instances=grouped_multihead_attention_8_256_4_16_ap_fixed_16_5_ limit=1 function
//     for (int i=0; i<4; i++){
//         data_t split_wq[64][256];
//         data_t split_wk[64][256];
//         data_t split_wv[64][256];
// 
//         // Copy rows directly
//         for (int r = 0; r < 64; r++) {
//             for (int c = 0; c < 256; c++) {
//             split_wq[r][c] = W_q[i * 64 + r][c];
//             split_wk[r][c] = W_k[i * 64 + r][c];
//             split_wv[r][c] = W_v[i * 64 + r][c];
//             }
//         }
// 
//         data_t split_out[8][64]; // temporary output for each group
// 
//         grouped_multihead_attention_8_256_4_16_ap_fixed_16_5_(input, split_wq, split_wk, split_wv, split_out); 
//         // Transpose the output for concatenation
//         
//         for (int r=0; r<8; r++){
//             for (int c=0; c<64; c++){
//                 output[c][i * 64 + r] = split_out[c][r];
//             }
//         }
//     }
// }
// 
// void store_8_64_ap_fixed_16_5_(data_t input[8][64], data_t output[8][64])
// {
//     #pragma HLS inline off
//     for (int idx0 = 0; idx0 < 8; idx0++) {
//         for (int idx1 = 0; idx1 < 64; idx1++) {
//             output[idx0][idx1] = input[idx0][idx1];
//         }
//     }
// }
// 
// void store_8_256_ap_fixed_16_5_(data_t input[8][256], data_t output[8][256])
// {
//     #pragma HLS inline off
//     for (int idx0 = 0; idx0 < 8; idx0++) {
//         for (int idx1 = 0; idx1 < 256; idx1++) {
//             output[idx0][idx1] = input[idx0][idx1];
//         }
//     }
// }
// 
// void top_A(data_t DRAM_attn_input[8][256], data_t DRAM_weights_q[256][256], 
//     data_t DRAM_weights_k[256][256], data_t DRAM_weights_v[256][256], data_t DRAM_output[8][256])
// {
//     data_t BRAM_attn_input[8][256];
//     data_t BRAM_weights_q[256][256];
//     data_t BRAM_weights_k[256][256];
//     data_t BRAM_weights_v[256][256];
//     data_t BRAM_1[8][256];
// 
//     load_8_256_ap_fixed_16_5_(DRAM_attn_input, BRAM_attn_input);
//     load_256_256_ap_fixed_16_5_(DRAM_weights_q, BRAM_weights_q);
//     load_256_256_ap_fixed_16_5_(DRAM_weights_k, BRAM_weights_k);
//     load_256_256_ap_fixed_16_5_(DRAM_weights_v, BRAM_weights_v);
//     
//     grouped_mha_8_256_16_16_with_mha_4(BRAM_attn_input, BRAM_weights_q, BRAM_weights_k, BRAM_weights_v, BRAM_1);
//         
//     store_8_256_ap_fixed_16_5_(BRAM_1, DRAM_output);
// }
// 
// void top_B(data_t DRAM_attn_input[8][256], data_t DRAM_weights_q[64][256], data_t DRAM_weights_k[64][256], data_t DRAM_weights_v[64][256], data_t DRAM_output[8][64])
// {
//     data_t BRAM_attn_input[8][256];
//     data_t BRAM_weights_q[64][256];
//     data_t BRAM_weights_k[64][256];
//     data_t BRAM_weights_v[64][256];
//     data_t BRAM_1[8][64];
// 
//     load_8_256_ap_fixed_16_5_(DRAM_attn_input, BRAM_attn_input);
//     load_64_256_ap_fixed_16_5_(DRAM_weights_q, BRAM_weights_q);
//     load_64_256_ap_fixed_16_5_(DRAM_weights_k, BRAM_weights_k);
//     load_64_256_ap_fixed_16_5_(DRAM_weights_v, BRAM_weights_v);
// 
//     grouped_multihead_attention_8_256_4_16_ap_fixed_16_5_(BRAM_attn_input, BRAM_weights_q, BRAM_weights_k, BRAM_weights_v, BRAM_1);
//     
//     store_8_64_ap_fixed_16_5_(BRAM_1, DRAM_output);
// }
// 
// 
// 
// void top( data_t DRAM_A1[8][256], data_t DRAM_A2[256][256], data_t DRAM_A3[256][256], data_t DRAM_A4[256][256], data_t DRAM_A5[8][256],
//           data_t DRAM_B1[8][256], data_t DRAM_B2[64][256], data_t DRAM_B3[64][256], data_t DRAM_B4[64][256], data_t DRAM_B5[8][64])
// {
//     
//     #pragma HLS allocation function instances=load_8_256_ap_fixed_16_5_ limit=1
//     #pragma HLS allocation function instances=load_256_256_ap_fixed_16_5_ limit=1
//     #pragma HLS allocation function instances=grouped_mha_8_256_16_16_with_mha_4 limit=1
//     #pragma HLS allocation function instances=grouped_multihead_attention_8_256_4_16_ap_fixed_16_5_ limit=1
//     #pragma HLS allocation function instances=load_64_256_ap_fixed_16_5_ limit=1
//     #pragma HLS allocation function instances=grouped_multihead_attention_8_256_4_16_ap_fixed_16_5_ limit=1
//     #pragma HLS allocation function instances=store_8_256_ap_fixed_16_5_ limit=1
//     #pragma HLS allocation function instances=store_8_64_ap_fixed_16_5_ limit=1
// 
//     #pragma HLS interface m_axi port=DRAM_A1 offset=slave bundle=mem1
//     #pragma HLS interface m_axi port=DRAM_A2 offset=slave bundle=mem1
//     #pragma HLS interface m_axi port=DRAM_A3 offset=slave bundle=mem1
//     #pragma HLS interface m_axi port=DRAM_A4 offset=slave bundle=mem1
//     #pragma HLS interface m_axi port=DRAM_A5 offset=slave bundle=mem2
//     
//     #pragma HLS interface m_axi port=DRAM_B1 offset=slave bundle=mem1
//     #pragma HLS interface m_axi port=DRAM_B2 offset=slave bundle=mem1
//     #pragma HLS interface m_axi port=DRAM_B3 offset=slave bundle=mem1
//     #pragma HLS interface m_axi port=DRAM_B4 offset=slave bundle=mem1
//     #pragma HLS interface m_axi port=DRAM_B5 offset=slave bundle=mem2
// 
//     //TOP_A
//     data_t BRAM_A1[8][256];
//     data_t BRAM_A2[256][256];
//     data_t BRAM_A3[256][256];
//     data_t BRAM_A4[256][256];
//     data_t BRAM_A5[8][256];
// 
//     load_8_256_ap_fixed_16_5_(DRAM_A1, BRAM_A1);
//     load_256_256_ap_fixed_16_5_(DRAM_A2, BRAM_A2);
//     load_256_256_ap_fixed_16_5_(DRAM_A3, BRAM_A3);
//     load_256_256_ap_fixed_16_5_(DRAM_A4, BRAM_A4);
//     // grouped_mha_8_256_16_16_with_mha_4(BRAM_A1, BRAM_A2, BRAM_A3, BRAM_A4, BRAM_A5);
// 
//     for (int i=0; i<4; i++){
//         data_t split_wq[64][256];
//         data_t split_wk[64][256];
//         data_t split_wv[64][256];
// 
//         // Copy rows directly
//         for (int r = 0; r < 64; r++) {
//             for (int c = 0; c < 256; c++) {
//             split_wq[r][c] = BRAM_A2[i * 64 + r][c];
//             split_wk[r][c] = BRAM_A3[i * 64 + r][c];
//             split_wv[r][c] = BRAM_A4[i * 64 + r][c];
//             }
//         }
// 
//         data_t split_out[8][64]; // temporary output for each group
// 
//         grouped_multihead_attention_8_256_4_16_ap_fixed_16_5_(BRAM_A1, split_wq, split_wk, split_wv, split_out); 
//         // Transpose the output for concatenation
//         
//         for (int r=0; r<8; r++){
//             for (int c=0; c<64; c++){
//                 BRAM_A5[c][i * 64 + r] = split_out[c][r];
//             }
//         }
//     }
// 
//     store_8_256_ap_fixed_16_5_(BRAM_A5, DRAM_A5);
// 
//     //TOP_B
//     data_t BRAM_B1[8][256];
//     data_t BRAM_B2[64][256];
//     data_t BRAM_B3[64][256];
//     data_t BRAM_B4[64][256];
//     data_t BRAM_B5[8][64];
// 
//     load_8_256_ap_fixed_16_5_(DRAM_B1, BRAM_B1);
//     load_64_256_ap_fixed_16_5_(DRAM_B2, BRAM_B2);
//     load_64_256_ap_fixed_16_5_(DRAM_B3, BRAM_B3);
//     load_64_256_ap_fixed_16_5_(DRAM_B4, BRAM_B4);
//     grouped_multihead_attention_8_256_4_16_ap_fixed_16_5_(BRAM_B1, BRAM_B2, BRAM_B3, BRAM_B4, BRAM_B5);
//     store_8_64_ap_fixed_16_5_(BRAM_B5, DRAM_B5);
// }