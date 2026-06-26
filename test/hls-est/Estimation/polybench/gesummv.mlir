// RUN: hls-est %s --affine-loop-normalize --memory-banking-bram --convert-affine-to-loopschedule --bram-analysis 

// CHECK-BRAM: Total BRAM: 20

module {
  func.func @gesummv(%arg0: f32, %arg1: f32, %arg2: memref<64x64xf32>, %arg3: memref<64x64xf32>, %arg4: memref<64xf32>, %arg5: memref<64xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca() {hls.preserve, polygeist.varname = "tmp2"} : memref<64xf32>
    %alloca_0 = memref.alloca() {hls.preserve, polygeist.varname = "tmp1"} : memref<64xf32>
    %alloca_1 = memref.alloca() {hls.preserve, polygeist.varname = "buff_y_out"} : memref<64xf32>
    %alloca_2 = memref.alloca() {hls.preserve, polygeist.varname = "buff_x"} : memref<64xf32>
    %alloca_3 = memref.alloca() {hls.preserve, polygeist.varname = "buff_B"} : memref<64x64xf32>
    %alloca_4 = memref.alloca() {hls.preserve, polygeist.varname = "buff_A"} : memref<64x64xf32>
    affine.for %arg6 = 0 to 64 {
      %0 = affine.load %arg4[%arg6] : memref<64xf32>
      affine.store %0, %alloca_2[%arg6] : memref<64xf32>
      affine.store %cst, %alloca_0[%arg6] : memref<64xf32>
      affine.store %cst, %alloca[%arg6] : memref<64xf32>
      affine.store %cst, %alloca_1[%arg6] : memref<64xf32>
      affine.for %arg7 = 0 to 64 {
        %1 = affine.load %arg2[%arg6, %arg7] : memref<64x64xf32>
        affine.store %1, %alloca_4[%arg6, %arg7] : memref<64x64xf32>
        %2 = affine.load %arg3[%arg6, %arg7] : memref<64x64xf32>
        affine.store %2, %alloca_3[%arg6, %arg7] : memref<64x64xf32>
      }
    }
    affine.for %arg6 = 0 to 64 {
      %0 = affine.load %alloca_0[%arg6] : memref<64xf32>
      %1 = affine.for %arg7 = 0 to 64 iter_args(%acc = %0) -> (f32) {
        %2 = affine.load %alloca_4[%arg6, %arg7] : memref<64x64xf32>
        %3 = arith.mulf %arg0, %2 : f32
        %4 = affine.load %alloca_2[%arg7] : memref<64xf32>
        %5 = arith.mulf %3, %4 : f32
        %6 = arith.addf %acc, %5 : f32
        affine.yield %6 : f32
      }
      affine.store %1, %alloca_0[%arg6] : memref<64xf32>
    }
    affine.for %arg6 = 0 to 64 {
      %0 = affine.load %alloca[%arg6] : memref<64xf32>
      %1 = affine.for %arg7 = 0 to 64 iter_args(%acc = %0) -> (f32) {
        %2 = affine.load %alloca_3[%arg6, %arg7] : memref<64x64xf32>
        %3 = arith.mulf %arg1, %2 : f32
        %4 = affine.load %alloca_2[%arg7] : memref<64xf32>
        %5 = arith.mulf %3, %4 : f32
        %6 = arith.addf %acc, %5 : f32
        affine.yield %6 : f32
      }
      affine.store %1, %alloca[%arg6] : memref<64xf32>
    }
    affine.for %arg6 = 0 to 64 {
      %0 = affine.load %alloca_0[%arg6] : memref<64xf32>
      %1 = affine.load %alloca[%arg6] : memref<64xf32>
      %2 = arith.addf %0, %1 : f32
      affine.store %2, %alloca_1[%arg6] : memref<64xf32>
    }
    affine.for %arg6 = 0 to 64 {
      %0 = affine.load %alloca_1[%arg6] : memref<64xf32>
      affine.store %0, %arg5[%arg6] : memref<64xf32>
    }
    return
  }
}

// Kernel: gesummv
// 
// #define N 64
// 
// #define DATA_TYPE float
// #define SCALAR_VAL(x) x##f
// #define SQRT_FUN(x) sqrtf(x)
// #define EXP_FUN(x) expf(x)
// #define POW_FUN(x,y) powf(x,y)
// 
// void gesummv(DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE x[N], DATA_TYPE y_out[N])
// {
//     int i, j;
// 	DATA_TYPE buff_A[N][N];
// 	DATA_TYPE buff_B[N][N];
// 	DATA_TYPE buff_x[N];
// 	DATA_TYPE buff_y_out[N];
// 	DATA_TYPE tmp1[N];
// 	DATA_TYPE tmp2[N];
// 
// 	lprd_1: for(i = 0; i < N; i++) {
// 		buff_x[i] = x[i];
// 		tmp1[i] = 0;
// 		tmp2[i] = 0;
// 		buff_y_out[i] = 0;
// 		lprd_2: for(j = 0; j < N; j++) {
//#pragma HLS unroll factor=1
// 			buff_A[i][j] = A[i][j];
// 			buff_B[i][j] = B[i][j];
// 		}
// 	}
// 
//     lp1: for(i = 0; i < N; i++) {
//         lp2: for(j = 0; j < N; j++) {
//#pragma HLS unroll factor=1
// 	        tmp1[i] += alpha * buff_A[i][j] * buff_x[j];
//         }
//     }
// 
// 	lp3: for(i = 0; i < N; i++) {
//         lp4: for(j = 0; j < N; j++) {
//#pragma HLS unroll factor=1
// 	        tmp2[i] += beta * buff_B[i][j] * buff_x[j];
//         }
//     }
// 
// 	lp5: for(i = 0; i < N; i++) {
// 		buff_y_out[i] = tmp1[i] + tmp2[i];
// 	}
// 
// 	lpwr: for(i = 0; i < N; i++) {
// 		y_out[i] = buff_y_out[i];
// 	}
// }
