// RUN: hls-est %s --affine-loop-normalize --merge-conditional-stores --convert-affine-to-loopschedule --bram-analysis

// Total BRAM: 32

#set = affine_set<(d0, d1) : (d0 - d1 - 1 >= 0)>
module {
  func.func @syrk(%arg0: f32, %arg1: f32, %arg2: memref<64x64xf32>, %arg3: memref<64x64xf32>, %arg4: memref<64x64xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca() {hls.preserve, polygeist.varname = "buff_C_out"} : memref<64x64xf32>
    %alloca_0 = memref.alloca() {hls.preserve, polygeist.varname = "buff_B"} : memref<64x64xf32>
    %alloca_1 = memref.alloca() {hls.preserve, polygeist.varname = "buff_A1"} : memref<64x64xf32>
    %alloca_2 = memref.alloca() {hls.preserve, polygeist.varname = "buff_A0"} : memref<64x64xf32>
    affine.for %arg5 = 0 to 64 {
      affine.for %arg6 = 0 to 64 {
        %0 = affine.load %arg2[%arg5, %arg6] : memref<64x64xf32>
        affine.store %0, %alloca_2[%arg5, %arg6] : memref<64x64xf32>
        %1 = affine.load %arg2[%arg5, %arg6] : memref<64x64xf32>
        affine.store %1, %alloca_1[%arg5, %arg6] : memref<64x64xf32>
        %2 = affine.load %arg3[%arg5, %arg6] : memref<64x64xf32>
        affine.store %2, %alloca_0[%arg5, %arg6] : memref<64x64xf32>
        affine.store %cst, %alloca[%arg5, %arg6] : memref<64x64xf32>
      }
    }
    affine.for %arg5 = 0 to 64 {
      affine.for %arg6 = 0 to 64 {
        %0 = affine.load %alloca[%arg5, %arg6] : memref<64x64xf32>
        %1 = affine.for %arg7 = 0 to 64 iter_args(%acc = %0) -> (f32) {
          %2 = affine.load %alloca_2[%arg5, %arg7] : memref<64x64xf32>
          %3 = arith.mulf %arg0, %2 : f32
          %4 = affine.load %alloca_1[%arg6, %arg7] : memref<64x64xf32>
          %5 = arith.mulf %3, %4 : f32
          %6 = arith.addf %acc, %5 : f32
          affine.yield %6 : f32
        }
        affine.store %1, %alloca[%arg5, %arg6] : memref<64x64xf32>
      }
    }
    affine.for %arg5 = 0 to 64 {
      affine.for %arg6 = 0 to 64 {
        %0 = affine.load %alloca_0[%arg5, %arg6] : memref<64x64xf32>
        %1 = arith.mulf %arg1, %0 : f32
        %2 = affine.load %alloca[%arg5, %arg6] : memref<64x64xf32>
        %3 = arith.addf %2, %1 : f32
        affine.store %3, %alloca[%arg5, %arg6] : memref<64x64xf32>
      }
    }
    affine.for %arg5 = 0 to 64 {
      affine.for %arg6 = 0 to 64 {
        affine.if #set(%arg6, %arg5) {
          affine.store %cst, %arg4[%arg5, %arg6] : memref<64x64xf32>
        } else {
          %0 = affine.load %alloca[%arg5, %arg6] : memref<64x64xf32>
          affine.store %0, %arg4[%arg5, %arg6] : memref<64x64xf32>
        }
      }
    }
    return
  }
}

// Kernel: syrk
// 
// #define DATA_TYPE float
// #define SCALAR_VAL(x) x##f
// #define SQRT_FUN(x) sqrtf(x)
// #define EXP_FUN(x) expf(x)
// #define POW_FUN(x,y) powf(x,y)
// 
// void syrk(DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE C_out[N][N]) {
// 	int i, j, k;
// 
// 	DATA_TYPE buff_A0[N][N], buff_A1[N][N];
// 	DATA_TYPE buff_B[N][N];
// 	DATA_TYPE buff_C_out[N][N];
// 
// 	lprd_1: for (i = 0; i < N; i++) {
// 		lprd_2: for (j = 0; j < N; j++) {
//#pragma HLS unroll factor=1
// 			buff_A0[i][j] = A[i][j];
// 			buff_A1[i][j] = A[i][j];
// 			buff_B[i][j] = B[i][j];
// 			buff_C_out[i][j] = 0;
// 		}
// 	}
// 
// 	lp1: for (i = 0; i < N; i++) {
// 		lp2: for (j = 0; j < N; j++) {
//#pragma HLS unroll factor=1
// 			// if (j > i) continue;
// 			lp3: for (k = 0; k < N; k++) {
// 				buff_C_out[i][j] += alpha * buff_A0[i][k] * buff_A1[j][k];
// 			}
// 		}
// 	}
// 
// 	lp4: for (i = 0; i < N; i++) {
// 		lp5: for (j = 0; j < N; j++) {
//#pragma HLS unroll factor=1
// 			// if (j > i) continue;
// 			buff_C_out[i][j] += beta * buff_B[i][j];
// 		}
// 	}
// 
// 	
// 
// 	lpwr_1: for(i = 0; i < N; i++) {
// 		lpwr_2: for(j = 0; j < N; j++) {
//#pragma HLS unroll factor=1
// 			if (j > i) C_out[i][j] = 0;
// 			else C_out[i][j] = buff_C_out[i][j];
// 		}
// 	}
// }