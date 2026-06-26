// RUN: hls-est %s --affine-loop-normalize --memory-banking-bram --convert-affine-to-loopschedule --bram-analysis | FileCheck %s -check-prefix=CHECK-BRAM

// CHECK-BRAM: Total BRAM: 1

module {
  func.func @matmul_partition(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: i32, %arg4: i32) {
    %c0 = arith.constant 0 : index
    %c16_i32 = arith.constant 16 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() {hls.array_partition = [{dim = 1 : i32, kind = "cyclic", factor = 16 : i32, variable = "C"}], polygeist.varname = "C"} : memref<256xi32>
    %alloca_0 = memref.alloca() {hls.array_partition = [{dim = 1 : i32, kind = "block", factor = 16 : i32, variable = "B"}], polygeist.varname = "B"} : memref<256xi32>
    %alloca_1 = memref.alloca() : memref<256xi32>
    %0 = arith.muli %arg3, %arg3 : i32
    %1 = arith.index_cast %0 : i32 to index
    %2:2 = affine.for %arg5 = 0 to %1 iter_args(%arg6 = %c0_i32, %arg7 = %c0_i32) -> (i32, i32) {
      %11 = arith.cmpi eq, %arg6, %arg3 : i32
      %12 = arith.select %11, %c0_i32, %arg6 : i32
      %13 = scf.if %11 -> (i32) {
        %19 = arith.addi %arg7, %c1_i32 : i32
        scf.yield %19 : i32
      } else {
        scf.yield %arg7 : i32
      }
      %14 = arith.muli %13, %c16_i32 : i32
      %15 = arith.addi %14, %12 : i32
      %16 = arith.index_cast %15 : i32 to index
      %17 = affine.load %arg0[%arg5] : memref<?xi32>
      memref.store %17, %alloca_1[%16] : memref<256xi32>
      %18 = arith.addi %12, %c1_i32 : i32
      affine.yield %18, %13 : i32, i32
    }
    %3 = arith.index_cast %0 : i32 to index
    %4:2 = affine.for %arg5 = 0 to %3 iter_args(%arg6 = %c0_i32, %arg7 = %c0_i32) -> (i32, i32) {
      %11 = arith.cmpi eq, %arg6, %arg3 : i32
      %12 = arith.select %11, %c0_i32, %arg6 : i32
      %13 = scf.if %11 -> (i32) {
        %19 = arith.addi %arg7, %c1_i32 : i32
        scf.yield %19 : i32
      } else {
        scf.yield %arg7 : i32
      }
      %14 = arith.muli %13, %c16_i32 : i32
      %15 = arith.addi %14, %12 : i32
      %16 = arith.index_cast %15 : i32 to index
      %17 = affine.load %arg1[%arg5] : memref<?xi32>
      memref.store %17, %alloca_0[%16] : memref<256xi32>
      %18 = arith.addi %12, %c1_i32 : i32
      affine.yield %18, %13 : i32, i32
    }
    %5 = arith.index_cast %arg4 : i32 to index
    %6 = arith.index_cast %arg3 : i32 to index
    %7 = arith.index_cast %arg3 : i32 to index
    %8 = arith.cmpi sgt, %5, %c0 : index
    scf.if %8 {
      affine.for %arg5 = 0 to %6 {
        affine.for %arg6 = 0 to %7 {
          %11 = affine.for %arg7 = 0 to 16 iter_args(%arg8 = %c0_i32) -> (i32) {
            %12 = affine.load %alloca_1[%arg7 + %arg5 * 16] : memref<256xi32>
            %13 = affine.load %alloca_0[%arg6 + %arg7 * 16] : memref<256xi32>
            %14 = arith.muli %12, %13 : i32
            %15 = arith.addi %arg8, %14 : i32
            affine.yield %15 : i32
          }
          affine.store %11, %alloca[%arg6 + %arg5 * 16] : memref<256xi32>
        }
      }
    }
    %9 = arith.index_cast %0 : i32 to index
    %10:2 = affine.for %arg5 = 0 to %9 iter_args(%arg6 = %c0_i32, %arg7 = %c0_i32) -> (i32, i32) {
      %11 = arith.cmpi eq, %arg6, %arg3 : i32
      %12 = arith.select %11, %c0_i32, %arg6 : i32
      %13 = scf.if %11 -> (i32) {
        %19 = arith.addi %arg7, %c1_i32 : i32
        scf.yield %19 : i32
      } else {
        scf.yield %arg7 : i32
      }
      %14 = arith.muli %13, %c16_i32 : i32
      %15 = arith.addi %14, %12 : i32
      %16 = arith.index_cast %15 : i32 to index
      %17 = memref.load %alloca[%16] : memref<256xi32>
      affine.store %17, %arg2[%arg5] : memref<?xi32>
      %18 = arith.addi %12, %c1_i32 : i32
      affine.yield %18, %13 : i32, i32
    }
    return
  }
}


// Kernel:
// #define MAX_DIM 16
// // TRIPCOUNT identifier
// const unsigned int c_dim = MAX_DIM;
// 
// void matmul_partition(int* in1, int* in2, int* out_r, int dim, int rep_count) { // Matrix Dimension. Assuming Square Matrix
// 
//     int A[MAX_DIM * MAX_DIM];
//     int B[MAX_DIM * MAX_DIM];
//     int C[MAX_DIM * MAX_DIM];
// // Cyclic Partition for A as matrix multiplication needs row-wise parallel
// // access
// #pragma HLS ARRAY_PARTITION variable = A dim = 1 cyclic factor = 16
// // Block Partition for B as matrix multiplication needs column-wise parallel
// // access
// #pragma HLS ARRAY_PARTITION variable = B dim = 1 block factor = 16
// 
// // As A and B Matrix are partitioned with the factor of MAX_DIM, so to get
// // parallel row/column access, input square matrix[dimXdim] should be written
// // into local Array in MATRIX[MAX_DIM * MAX_DIM] format
// 
// // Burst read for matrix A
// // Auto-pipeline is going to apply pipeline to these loops
// readA:
//     for (int itr = 0, i = 0, j = 0; itr < dim * dim; itr++, j++) {
// #pragma HLS LOOP_TRIPCOUNT min = c_dim* c_dim max = c_dim * c_dim
//         if (j == dim) {
//             j = 0;
//             i++;
//         }
//         A[i * MAX_DIM + j] = in1[itr];
//     }
// 
// // Burst read for matrix B
// readB:
//     for (int itr = 0, i = 0, j = 0; itr < dim * dim; itr++, j++) {
// #pragma HLS LOOP_TRIPCOUNT min = c_dim* c_dim max = c_dim * c_dim
//         if (j == dim) {
//             j = 0;
//             i++;
//         }
//         B[i * MAX_DIM + j] = in2[itr];
//     }
// 
// loop2:
//     for (int x = 0; x < rep_count; x++) {
// #pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
//     lreorder1:
//         for (int i = 0; i < dim; i++) {
// #pragma HLS LOOP_TRIPCOUNT min = c_dim max = c_dim
//         // As A and B are partition correctly so loop pipelining is applied
//         // at 2nd level loop and which will eventually unroll the lower loop
//         lreorder2:
//             for (int j = 0; j < dim; j++) {
// #pragma HLS LOOP_TRIPCOUNT min = c_dim max = c_dim
//                 int result = 0;
//             lreorder3:
//                 for (int k = 0; k < MAX_DIM; k++) {
//                     //#pragma HLS LOOP_TRIPCOUNT min = c_dim max = c_dim
//                     result += A[i * MAX_DIM + k] * B[k * MAX_DIM + j];
//                 }
//                 C[i * MAX_DIM + j] = result;
//             }
//         }
//     }
// 
// // Burst write from output matrices to global memory
// // Burst write from matrix C
// writeC:
//     for (int itr = 0, i = 0, j = 0; itr < dim * dim; itr++, j++) {
// #pragma HLS LOOP_TRIPCOUNT min = c_dim* c_dim max = c_dim * c_dim
//         if (j == dim) {
//             j = 0;
//             i++;
//         }
//         out_r[itr] = C[i * MAX_DIM + j];
//     }
// }