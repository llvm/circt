// RUN: hls-est %s --affine-loop-normalize --memory-banking-bram --convert-affine-to-loopschedule --bram-analysis

// Total BRAM: 12

module {
  func.func @mvt(%arg0: memref<64x64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>, %arg3: memref<64xf32>, %arg4: memref<64xf32>, %arg5: memref<64xf32>, %arg6: memref<64xf32>) {
    %alloca = memref.alloca() {hls.preserve, polygeist.varname = "buff_y2"} : memref<64xf32>
    %alloca_0 = memref.alloca() {hls.preserve, polygeist.varname = "buff_y1"} : memref<64xf32>
    %alloca_1 = memref.alloca() {hls.preserve, polygeist.varname = "buff_x2"} : memref<64xf32>
    %alloca_2 = memref.alloca() {hls.preserve, polygeist.varname = "buff_x1"} : memref<64xf32>
    %alloca_3 = memref.alloca() {hls.preserve, polygeist.varname = "buff_A"} : memref<64x64xf32>
    affine.for %arg7 = 0 to 64 {
      %0 = affine.load %arg1[%arg7] : memref<64xf32>
      affine.store %0, %alloca_2[%arg7] : memref<64xf32>
      %1 = affine.load %arg2[%arg7] : memref<64xf32>
      affine.store %1, %alloca_1[%arg7] : memref<64xf32>
      %2 = affine.load %arg3[%arg7] : memref<64xf32>
      affine.store %2, %alloca_0[%arg7] : memref<64xf32>
      %3 = affine.load %arg4[%arg7] : memref<64xf32>
      affine.store %3, %alloca[%arg7] : memref<64xf32>
      affine.for %arg8 = 0 to 64 {
        %4 = affine.load %arg0[%arg7, %arg8] : memref<64x64xf32>
        affine.store %4, %alloca_3[%arg7, %arg8] : memref<64x64xf32>
      }
    }
    affine.for %arg7 = 0 to 64 {
      %0 = affine.load %alloca_2[%arg7] : memref<64xf32>
      %1 = affine.for %arg8 = 0 to 64 iter_args(%acc = %0) -> (f32) {
        %2 = affine.load %alloca_3[%arg7, %arg8] : memref<64x64xf32>
        %3 = affine.load %alloca_0[%arg8] : memref<64xf32>
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %acc, %4 : f32
        affine.yield %5 : f32
      }
      affine.store %1, %alloca_2[%arg7] : memref<64xf32>
    }
    affine.for %arg7 = 0 to 64 {
      %0 = affine.load %alloca_1[%arg7] : memref<64xf32>
      %1 = affine.for %arg8 = 0 to 64 iter_args(%acc = %0) -> (f32) {
        %2 = affine.load %alloca_3[%arg8, %arg7] : memref<64x64xf32>
        %3 = affine.load %alloca[%arg8] : memref<64xf32>
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.addf %acc, %4 : f32
        affine.yield %5 : f32
      }
      affine.store %1, %alloca_1[%arg7] : memref<64xf32>
    }
    affine.for %arg7 = 0 to 64 {
      %0 = affine.load %alloca_2[%arg7] : memref<64xf32>
      affine.store %0, %arg5[%arg7] : memref<64xf32>
      %1 = affine.load %alloca_1[%arg7] : memref<64xf32>
      affine.store %1, %arg6[%arg7] : memref<64xf32>
    }
    return
  }
}

// Kernel: mvt
// 
// #define N 64
// 
// #define DATA_TYPE float
// #define SCALAR_VAL(x) x##f
// #define SQRT_FUN(x) sqrtf(x)
// #define EXP_FUN(x) expf(x)
// #define POW_FUN(x,y) powf(x,y)
// 
// void mvt(DATA_TYPE A[N][N], DATA_TYPE x1[N], DATA_TYPE x2[N], DATA_TYPE y1[N], DATA_TYPE y2[N], DATA_TYPE x1_out[N], DATA_TYPE x2_out[N])
// {
// 	int i, j;
// 	DATA_TYPE buff_A[N][N];
// 	DATA_TYPE buff_x1[N];
// 	DATA_TYPE buff_x2[N];
// 	DATA_TYPE buff_y1[N];
// 	DATA_TYPE buff_y2[N];
// 
// 	lprd_1: for (i = 0; i < N; i++) {
// 		buff_x1[i] = x1[i];
// 		buff_x2[i] = x2[i];
// 		buff_y1[i] = y1[i];
// 		buff_y2[i] = y2[i];
//     	lprd_2: for (j = 0; j < N; j++) {
//#pragma HLS unroll factor=1
// 			buff_A[i][j] = A[i][j];
// 		}
// 	}
// 
// 	lp1: for (i = 0; i < N; i++) {
//     	lp2: for (j = 0; j < N; j++) {
//#pragma HLS unroll factor=1
//       		buff_x1[i] = buff_x1[i] + buff_A[i][j] * buff_y1[j];
// 		}
// 	}
// 
// 	lp3: for (i = 0; i < N; i++) {
// 		lp4: for (j = 0; j < N; j++) {
//#pragma HLS unroll factor=1
//     		buff_x2[i] = buff_x2[i] + buff_A[j][i] * buff_y2[j];
// 		}
// 	}
// 	
// 	lpwr: for (i = 0; i < N; i++) {
// 		x1_out[i] = buff_x1[i];
// 		x2_out[i] = buff_x2[i];
// 	}
// }
