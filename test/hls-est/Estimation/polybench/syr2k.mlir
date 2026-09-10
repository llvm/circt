// RUN: hls-est %s --affine-loop-normalize --memory-banking-bram --merge-conditional-stores --convert-affine-to-loopschedule --bram-analysis

// Total BRAM: 56

#set = affine_set<(d0, d1) : (d0 - d1 - 1 >= 0)>
module {
  func.func @syr2k(%arg0: f32, %arg1: f32, %arg2: memref<64x64xf32>, %arg3: memref<64x64xf32>, %arg4: memref<64x64xf32>, %arg5: memref<64x64xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca() {hls.preserve, polygeist.varname = "tmp2"} : memref<64x64xf32>
    %alloca_0 = memref.alloca() {hls.preserve, polygeist.varname = "tmp1"} : memref<64x64xf32>
    %alloca_1 = memref.alloca() {hls.preserve, polygeist.varname = "buff_D_out"} : memref<64x64xf32>
    %alloca_2 = memref.alloca() {hls.preserve, polygeist.varname = "buff_C"} : memref<64x64xf32>
    %alloca_3 = memref.alloca() {hls.preserve, polygeist.varname = "buff_B0"} : memref<64x64xf32>
    %alloca_4 = memref.alloca() {hls.preserve, polygeist.varname = "buff_A1"} : memref<64x64xf32>
    %alloca_5 = memref.alloca() {hls.preserve, polygeist.varname = "buff_A0"} : memref<64x64xf32>
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 64 {
        %0 = affine.load %arg2[%arg6, %arg7] : memref<64x64xf32>
        affine.store %0, %alloca_5[%arg6, %arg7] : memref<64x64xf32>
        %1 = affine.load %arg2[%arg6, %arg7] : memref<64x64xf32>
        affine.store %1, %alloca_4[%arg6, %arg7] : memref<64x64xf32>
        %2 = affine.load %arg3[%arg6, %arg7] : memref<64x64xf32>
        affine.store %2, %alloca_3[%arg6, %arg7] : memref<64x64xf32>
        %3 = affine.load %arg4[%arg6, %arg7] : memref<64x64xf32>
        affine.store %3, %alloca_2[%arg6, %arg7] : memref<64x64xf32>
        affine.store %cst, %alloca_1[%arg6, %arg7] : memref<64x64xf32>
        affine.store %cst, %alloca_0[%arg6, %arg7] : memref<64x64xf32>
        affine.store %cst, %alloca[%arg6, %arg7] : memref<64x64xf32>
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 64 {
        %0 = affine.load %alloca_0[%arg6, %arg7] : memref<64x64xf32>
        %1 = affine.for %arg8 = 0 to 64 iter_args(%acc = %0) -> (f32) {
          %2 = affine.load %alloca_5[%arg6, %arg8] : memref<64x64xf32>
          %3 = arith.mulf %arg0, %2 : f32
          %4 = affine.load %alloca_3[%arg7, %arg8] : memref<64x64xf32>
          %5 = arith.mulf %3, %4 : f32
          %6 = arith.addf %acc, %5 : f32
          affine.yield %6 : f32
        }
        affine.store %1, %alloca_0[%arg6, %arg7] : memref<64x64xf32>
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 64 {
        %0 = affine.load %alloca[%arg6, %arg7] : memref<64x64xf32>
        %1 = affine.for %arg8 = 0 to 64 iter_args(%acc = %0) -> (f32) {
          %2 = affine.load %alloca_3[%arg6, %arg8] : memref<64x64xf32>
          %3 = arith.mulf %arg0, %2 : f32
          %4 = affine.load %alloca_4[%arg7, %arg8] : memref<64x64xf32>
          %5 = arith.mulf %3, %4 : f32
          %6 = arith.addf %acc, %5 : f32
          affine.yield %6 : f32
        }
        affine.store %1, %alloca[%arg6, %arg7] : memref<64x64xf32>
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 64 {
        %0 = affine.load %alloca_0[%arg6, %arg7] : memref<64x64xf32>
        %1 = affine.load %alloca[%arg6, %arg7] : memref<64x64xf32>
        %2 = arith.addf %0, %1 : f32
        %3 = affine.load %alloca_2[%arg6, %arg7] : memref<64x64xf32>
        %4 = arith.mulf %arg1, %3 : f32
        %5 = arith.addf %2, %4 : f32
        affine.store %5, %alloca_1[%arg6, %arg7] : memref<64x64xf32>
      }
    }
    affine.for %arg6 = 0 to 64 {
      affine.for %arg7 = 0 to 64 {
        affine.if #set(%arg7, %arg6) {
          affine.store %cst, %arg5[%arg6, %arg7] : memref<64x64xf32>
        } else {
          %0 = affine.load %alloca_1[%arg6, %arg7] : memref<64x64xf32>
          affine.store %0, %arg5[%arg6, %arg7] : memref<64x64xf32>
        }
      }
    }
    return
  }
}

// Kernel: syr2k
// 
// #define N 64
// #define DATA_TYPE float
// #define SCALAR_VAL(x) x##f
// #define SQRT_FUN(x) sqrtf(x)
// #define EXP_FUN(x) expf(x)
// #define POW_FUN(x,y) powf(x,y)
// 
// void syr2k(DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE C[N][N], DATA_TYPE D_out[N][N]){
// 	int i, j, k;
// 	DATA_TYPE buff_A0[N][N], buff_A1[N][N];
// 	DATA_TYPE buff_B0[N][N], buff_B1[N][N];
// 	DATA_TYPE buff_C[N][N];
// 	DATA_TYPE buff_D_out[N][N];
// 	DATA_TYPE tmp1[N][N];
// 	DATA_TYPE tmp2[N][N];
// 
// 
// 	lprd_1: for (i = 0; i < N; i++){
// 		lprd_2: for (j = 0; j < N; j++){
//#pragma HLS unroll factor=1
// 			buff_A0[i][j] = A[i][j];
// 			buff_A1[i][j] = A[i][j];
// 			buff_B0[i][j] = B[i][j];
// 			buff_B1[i][j] = B[i][j];
// 			buff_C[i][j] = C[i][j];
// 			buff_D_out[i][j] = 0;
// 			tmp1[i][j] = 0;
// 			tmp2[i][j] = 0;
// 		}
// 	}
// 
// 	lp1: for (i = 0; i < N; i++){
// 		lp2: for (j = 0; j < N; j++){
// 			// if (j > i) continue;
// 			lp3: for (k = 0; k < N; k++){
//#pragma HLS unroll factor=1
// 				tmp1[i][j] += alpha * buff_A0[i][k] * buff_B0[j][k];
// 			}
// 		}
// 	}
// 
// 	lp4: for (i = 0; i < N; i++){
// 		lp5: for (j = 0; j < N; j++){
// 			// if (j > i) continue;
// 			lp6: for (k = 0; k < N; k++){
//#pragma HLS unroll factor=1
// 				tmp2[i][j] += alpha * buff_B0[i][k] * buff_A1[j][k];
// 			}
// 		}
// 	}
// 
// 	lp7: for (i = 0; i < N; i++){
// 		lp8: for (j = 0; j < N; j++){
//#pragma HLS unroll factor=1
// 			//if (j > i) continue;
// 			buff_D_out[i][j] = tmp1[i][j] + tmp2[i][j] + beta * buff_C[i][j];
// 		}
// 	}
// 
// 	lpwr_1: for(i = 0; i < N; i++){
// 		lpwr_2: for(j = 0; j < N; j++){
//#pragma HLS unroll factor=1
// 			if (j > i) D_out[i][j] = 0;
// 			else D_out[i][j] = buff_D_out[i][j];
// 		}
// 	}
// }