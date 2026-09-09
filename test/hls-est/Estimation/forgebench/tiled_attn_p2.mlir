// RUN: hls-est %s --affine-loop-normalize --memory-banking-bram --convert-affine-to-loopschedule --bram-analysis | FileCheck %s -check-prefix=CHECK-BRAM

module {
  memref.global @BRAM_1 : memref<8x64xf16> = uninitialized
  memref.global @BRAM_weights_v : memref<64x256xf16> = uninitialized
  memref.global @BRAM_weights_k : memref<64x256xf16> = uninitialized
  memref.global @BRAM_weights_q : memref<64x256xf16> = uninitialized
  memref.global @BRAM_attn_input : memref<8x256xf16> = uninitialized
  func.func @_Z25load_8_256_ap_fixed_16_5_PA256_fS0_(%arg0: memref<8x256xf16>, %arg1: memref<8x256xf16>) {
    affine.for %arg2 = 0 to 8 {
      affine.for %arg3 = 0 to 256 {
        %0 = affine.load %arg0[%arg2, %arg3] : memref<8x256xf16>
        affine.store %0, %arg1[%arg2, %arg3] : memref<8x256xf16>
      }
    }
    return
  }
  func.func @_Z26load_64_256_ap_fixed_16_5_PA256_fS0_(%arg0: memref<64x256xf16>, %arg1: memref<64x256xf16>) {
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 256 {
        %0 = affine.load %arg0[%arg2, %arg3] : memref<64x256xf16>
        affine.store %0, %arg1[%arg2, %arg3] : memref<64x256xf16>
      }
    }
    return
  }
  func.func @_Z53grouped_multihead_attention_8_256_4_16_ap_fixed_16_5_PA256_fS0_S0_S0_PA64_fi(%arg0: memref<8x256xf16>, %arg1: memref<64x256xf16>, %arg2: memref<64x256xf16>, %arg3: memref<64x256xf16>, %arg4: memref<8x64xf16>, %arg5: i32) {
    %cst = arith.constant 2.500000e-01 : f16
    %cst_0 = arith.constant 0.000000e+00 : f16
    %c4_i32 = arith.constant 4 : i32
    %alloca = memref.alloca() {hls.preserve, polygeist.varname = "scores"} : memref<8x8xf16>
    %alloca_1 = memref.alloca() {hls.preserve, polygeist.varname = "V"} : memref<8x64xf16>
    %alloca_2 = memref.alloca() {hls.preserve, polygeist.varname = "K"} : memref<8x64xf16>
    %alloca_3 = memref.alloca() {hls.preserve, polygeist.varname = "Q"} : memref<8x64xf16>
    %0 = arith.divsi %c4_i32, %arg5 : i32
    %1 = arith.index_cast %0 : i32 to index
    affine.for %arg6 = 0 to 8 {
      affine.for %arg7 = 0 to 64 {
        affine.store %cst_0, %alloca_3[%arg6, %arg7] : memref<8x64xf16>
        affine.store %cst_0, %alloca_2[%arg6, %arg7] : memref<8x64xf16>
        affine.store %cst_0, %alloca_1[%arg6, %arg7] : memref<8x64xf16>
        %3:3 = affine.for %arg8 = 0 to 256 iter_args(%arg9 = %cst_0, %arg10 = %cst_0, %arg11 = %cst_0) -> (f16, f16, f16) {
          %4 = affine.load %arg0[%arg6, %arg8] : memref<8x256xf16>
          %5 = affine.load %arg1[%arg7, %arg8] : memref<64x256xf16>
          %6 = arith.mulf %4, %5 : f16
          %7 = arith.addf %arg11, %6 : f16
          affine.store %7, %alloca_3[%arg6, %arg7] : memref<8x64xf16>
          %8 = affine.load %arg0[%arg6, %arg8] : memref<8x256xf16>
          %9 = affine.load %arg2[%arg7, %arg8] : memref<64x256xf16>
          %10 = arith.mulf %8, %9 : f16
          %11 = arith.addf %arg10, %10 : f16
          affine.store %11, %alloca_2[%arg6, %arg7] : memref<8x64xf16>
          %12 = affine.load %arg0[%arg6, %arg8] : memref<8x256xf16>
          %13 = affine.load %arg3[%arg7, %arg8] : memref<64x256xf16>
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
            %3 = affine.for %arg10 = 0 to 16 iter_args(%arg11 = %cst_0) -> (f16) {
              %6 = affine.load %alloca_3[%arg8, %arg10 + %arg7 * 16 + (%arg6 * symbol(%1)) * 16] : memref<8x64xf16>
              %7 = affine.load %alloca_2[%arg9, %arg10 + %arg7 * 16 + (%arg6 * symbol(%1)) * 16] : memref<8x64xf16>
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
          affine.for %arg9 = 0 to 16 {
            %3 = affine.for %arg10 = 0 to 8 iter_args(%arg11 = %cst_0) -> (f16) {
              %4 = affine.load %alloca[%arg8, %arg10] : memref<8x8xf16>
              %5 = affine.load %alloca_1[%arg10, %arg9 + %arg7 * 16 + (%arg6 * symbol(%1)) * 16] : memref<8x64xf16>
              %6 = arith.mulf %4, %5 : f16
              %7 = arith.addf %arg11, %6 : f16
              affine.yield %7 : f16
            }
            affine.store %3, %arg4[%arg8, %arg9 + %arg7 * 16 + (%arg6 * symbol(%1)) * 16] : memref<8x64xf16>
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
  func.func @_Z3topPA256_fS0_S0_S0_PA64_f(%arg0: memref<8x256xf16>, %arg1: memref<64x256xf16>, %arg2: memref<64x256xf16>, %arg3: memref<64x256xf16>, %arg4: memref<8x64xf16>) {
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 2.500000e-01 : f16
    %0 = memref.get_global @BRAM_attn_input : memref<8x256xf16>
    affine.for %arg5 = 0 to 8 {
      affine.for %arg6 = 0 to 256 {
        %5 = affine.load %arg0[%arg5, %arg6] : memref<8x256xf16>
        affine.store %5, %0[%arg5, %arg6] : memref<8x256xf16>
      }
    }
    %1 = memref.get_global @BRAM_weights_q : memref<64x256xf16>
    affine.for %arg5 = 0 to 64 {
      affine.for %arg6 = 0 to 256 {
        %5 = affine.load %arg1[%arg5, %arg6] : memref<64x256xf16>
        affine.store %5, %1[%arg5, %arg6] : memref<64x256xf16>
      }
    }
    %2 = memref.get_global @BRAM_weights_k : memref<64x256xf16>
    affine.for %arg5 = 0 to 64 {
      affine.for %arg6 = 0 to 256 {
        %5 = affine.load %arg2[%arg5, %arg6] : memref<64x256xf16>
        affine.store %5, %2[%arg5, %arg6] : memref<64x256xf16>
      }
    }
    %3 = memref.get_global @BRAM_weights_v : memref<64x256xf16>
    affine.for %arg5 = 0 to 64 {
      affine.for %arg6 = 0 to 256 {
        %5 = affine.load %arg3[%arg5, %arg6] : memref<64x256xf16>
        affine.store %5, %3[%arg5, %arg6] : memref<64x256xf16>
      }
    }
    %4 = memref.get_global @BRAM_1 : memref<8x64xf16>
    %alloca = memref.alloca() {hls.preserve, polygeist.varname = "scores"} : memref<8x8xf16>
    %alloca_1 = memref.alloca() {hls.preserve, polygeist.varname = "V"} : memref<8x64xf16>
    %alloca_2 = memref.alloca() {hls.preserve, polygeist.varname = "K"} : memref<8x64xf16>
    %alloca_3 = memref.alloca() {hls.preserve, polygeist.varname = "Q"} : memref<8x64xf16>
    affine.for %arg5 = 0 to 8 {
      affine.for %arg6 = 0 to 64 {
        affine.store %cst, %alloca_3[%arg5, %arg6] : memref<8x64xf16>
        affine.store %cst, %alloca_2[%arg5, %arg6] : memref<8x64xf16>
        affine.store %cst, %alloca_1[%arg5, %arg6] : memref<8x64xf16>
        %5:3 = affine.for %arg7 = 0 to 256 iter_args(%arg8 = %cst, %arg9 = %cst, %arg10 = %cst) -> (f16, f16, f16) {
          %6 = affine.load %0[%arg5, %arg7] : memref<8x256xf16>
          %7 = affine.load %1[%arg6, %arg7] : memref<64x256xf16>
          %8 = arith.mulf %6, %7 : f16
          %9 = arith.addf %arg10, %8 : f16
          affine.store %9, %alloca_3[%arg5, %arg6] : memref<8x64xf16>
          %10 = affine.load %0[%arg5, %arg7] : memref<8x256xf16>
          %11 = affine.load %2[%arg6, %arg7] : memref<64x256xf16>
          %12 = arith.mulf %10, %11 : f16
          %13 = arith.addf %arg9, %12 : f16
          affine.store %13, %alloca_2[%arg5, %arg6] : memref<8x64xf16>
          %14 = affine.load %0[%arg5, %arg7] : memref<8x256xf16>
          %15 = affine.load %3[%arg6, %arg7] : memref<64x256xf16>
          %16 = arith.mulf %14, %15 : f16
          %17 = arith.addf %arg8, %16 : f16
          affine.store %17, %alloca_1[%arg5, %arg6] : memref<8x64xf16>
          affine.yield %17, %13, %9 : f16, f16, f16
        }
      }
    }
    affine.for %arg5 = 0 to 4 {
      affine.for %arg6 = 0 to 8 {
        affine.for %arg7 = 0 to 8 {
          affine.store %cst, %alloca[%arg6, %arg7] : memref<8x8xf16>
          %5 = affine.for %arg8 = 0 to 16 iter_args(%arg9 = %cst) -> (f16) {
            %8 = affine.load %alloca_3[%arg6, %arg8 + %arg5 * 16] : memref<8x64xf16>
            %9 = affine.load %alloca_2[%arg7, %arg8 + %arg5 * 16] : memref<8x64xf16>
            %10 = arith.mulf %8, %9 : f16
            %11 = arith.addf %arg9, %10 : f16
            affine.store %11, %alloca[%arg6, %arg7] : memref<8x8xf16>
            affine.yield %11 : f16
          }
          %6 = affine.load %alloca[%arg6, %arg7] : memref<8x8xf16>
          %7 = arith.mulf %6, %cst_0 : f16
          affine.store %7, %alloca[%arg6, %arg7] : memref<8x8xf16>
        }
      }
      affine.for %arg6 = 0 to 8 {
        %5 = affine.load %alloca[%arg6, 0] : memref<8x8xf16>
        %6 = affine.for %arg7 = 1 to 8 iter_args(%arg8 = %5) -> (f16) {
          %8 = affine.load %alloca[%arg6, %arg7] : memref<8x8xf16>
          %9 = arith.cmpf ogt, %8, %arg8 : f16
          %10 = arith.select %9, %8, %arg8 : f16
          affine.yield %10 : f16
        }
        %7 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %cst) -> (f16) {
          %8 = affine.load %alloca[%arg6, %arg7] : memref<8x8xf16>
          %9 = arith.subf %8, %6 : f16
          %10 = math.exp %9 : f16
          affine.store %10, %alloca[%arg6, %arg7] : memref<8x8xf16>
          %11 = arith.addf %arg8, %10 : f16
          affine.yield %11 : f16
        }
        affine.for %arg7 = 0 to 8 {
          %8 = affine.load %alloca[%arg6, %arg7] : memref<8x8xf16>
          %9 = arith.divf %8, %7 : f16
          affine.store %9, %alloca[%arg6, %arg7] : memref<8x8xf16>
        }
      }
      affine.for %arg6 = 0 to 8 {
        affine.for %arg7 = 0 to 16 {
          %5 = affine.for %arg8 = 0 to 8 iter_args(%arg9 = %cst) -> (f16) {
            %6 = affine.load %alloca[%arg6, %arg8] : memref<8x8xf16>
            %7 = affine.load %alloca_1[%arg8, %arg7 + %arg5 * 16] : memref<8x64xf16>
            %8 = arith.mulf %6, %7 : f16
            %9 = arith.addf %arg9, %8 : f16
            affine.yield %9 : f16
          }
          affine.store %5, %4[%arg6, %arg7 + %arg5 * 16] : memref<8x64xf16>
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
//     for (int idx0 = 0; idx0 < 8; idx0++) {
//         for (int idx1 = 0; idx1 < 256; idx1++) {
//             output[idx0][idx1] = input[idx0][idx1];
//         }
//     }
// }
// 
// void load_64_256_ap_fixed_16_5_(data_t input[64][256], data_t output[64][256])
// {
//     for (int idx0 = 0; idx0 < 64; idx0++) {
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
//  void grouped_multihead_attention_8_256_4_16_ap_fixed_16_5_(
//     data_t input[8][256],
//     data_t W_q[64][256],
//     data_t W_k[64][256],
//     data_t W_v[64][256],
//     data_t output[8][64],
//     int groups
// )
// {
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
// void store_8_64_ap_fixed_16_5_(data_t input[8][64], data_t output[8][64])
// {
//     for (int idx0 = 0; idx0 < 8; idx0++) {
//         for (int idx1 = 0; idx1 < 64; idx1++) {
//             output[idx0][idx1] = input[idx0][idx1];
//         }
//     }
// }
// 
// void top(data_t DRAM_attn_input[8][256], data_t DRAM_weights_q[64][256], data_t DRAM_weights_k[64][256], data_t DRAM_weights_v[64][256], data_t DRAM_output[8][64])
// {
//     #pragma HLS interface m_axi port=DRAM_attn_input offset=slave bundle=mem1
//     #pragma HLS interface m_axi port=DRAM_weights_q offset=slave bundle=mem1
//     #pragma HLS interface m_axi port=DRAM_weights_k offset=slave bundle=mem1
//     #pragma HLS interface m_axi port=DRAM_weights_v offset=slave bundle=mem1
//     #pragma HLS interface m_axi port=DRAM_output offset=slave bundle=mem2
// 
//     load_8_256_ap_fixed_16_5_(DRAM_attn_input, BRAM_attn_input);
//     load_64_256_ap_fixed_16_5_(DRAM_weights_q, BRAM_weights_q);
//     load_64_256_ap_fixed_16_5_(DRAM_weights_k, BRAM_weights_k);
//     load_64_256_ap_fixed_16_5_(DRAM_weights_v, BRAM_weights_v);
//     grouped_multihead_attention_8_256_4_16_ap_fixed_16_5_(BRAM_attn_input, BRAM_weights_q, BRAM_weights_k, BRAM_weights_v, BRAM_1, 4);
//     store_8_64_ap_fixed_16_5_(BRAM_1, DRAM_output);
// }