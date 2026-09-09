module {
  func.func @_Z13histogram_hlsPhPjm(%arg0: memref<8192xi8>, %arg1: memref<256xi32>, %arg2: i64)  {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %arg2 : i64 to index
    %alloca = memref.alloca() {hls.preserve, polygeist.varname = "hist8"} : memref<256xi32>
    %alloca_0 = memref.alloca() {hls.preserve, polygeist.varname = "hist7"} : memref<256xi32>
    %alloca_1 = memref.alloca() {hls.preserve, polygeist.varname = "hist6"} : memref<256xi32>
    %alloca_2 = memref.alloca() {hls.preserve, polygeist.varname = "hist5"} : memref<256xi32>
    %alloca_3 = memref.alloca() {hls.preserve, polygeist.varname = "hist4"} : memref<256xi32>
    %alloca_4 = memref.alloca() {hls.preserve, polygeist.varname = "hist3"} : memref<256xi32>
    %alloca_5 = memref.alloca() {hls.preserve, polygeist.varname = "hist2"} : memref<256xi32>
    %alloca_6 = memref.alloca() {hls.preserve, polygeist.varname = "hist1"} : memref<256xi32>
    affine.for %arg3 = 0 to 256 {
      affine.store %c0_i32, %alloca_6[%arg3] : memref<256xi32>
      affine.store %c0_i32, %alloca_5[%arg3] : memref<256xi32>
      affine.store %c0_i32, %alloca_4[%arg3] : memref<256xi32>
      affine.store %c0_i32, %alloca_3[%arg3] : memref<256xi32>
      affine.store %c0_i32, %alloca_2[%arg3] : memref<256xi32>
      affine.store %c0_i32, %alloca_1[%arg3] : memref<256xi32>
      affine.store %c0_i32, %alloca_0[%arg3] : memref<256xi32>
      affine.store %c0_i32, %alloca[%arg3] : memref<256xi32>
    }
    affine.for %arg3 = 0 to 1017 step 8 {
      %1 = affine.load %arg0[%arg3 + symbol(%0)] : memref<8192xi8>
      %2 = arith.index_cast %1 : i8 to index
      %3 = memref.load %alloca_6[%2] : memref<256xi32>
      %4 = arith.addi %3, %c1_i32 : i32
      memref.store %4, %alloca_6[%2] : memref<256xi32>
      %5 = affine.load %arg0[%arg3 + symbol(%0) + 1] : memref<8192xi8>
      %6 = arith.index_cast %5 : i8 to index
      %7 = memref.load %alloca_5[%6] : memref<256xi32>
      %8 = arith.addi %7, %c1_i32 : i32
      memref.store %8, %alloca_5[%6] : memref<256xi32>
      %9 = affine.load %arg0[%arg3 + symbol(%0) + 2] : memref<8192xi8>
      %10 = arith.index_cast %9 : i8 to index
      %11 = memref.load %alloca_4[%10] : memref<256xi32>
      %12 = arith.addi %11, %c1_i32 : i32
      memref.store %12, %alloca_4[%10] : memref<256xi32>
      %13 = affine.load %arg0[%arg3 + symbol(%0) + 3] : memref<8192xi8>
      %14 = arith.index_cast %13 : i8 to index
      %15 = memref.load %alloca_3[%14] : memref<256xi32>
      %16 = arith.addi %15, %c1_i32 : i32
      memref.store %16, %alloca_3[%14] : memref<256xi32>
      %17 = affine.load %arg0[%arg3 + symbol(%0) + 4] : memref<8192xi8>
      %18 = arith.index_cast %17 : i8 to index
      %19 = memref.load %alloca_2[%18] : memref<256xi32>
      %20 = arith.addi %19, %c1_i32 : i32
      memref.store %20, %alloca_2[%18] : memref<256xi32>
      %21 = affine.load %arg0[%arg3 + symbol(%0) + 5] : memref<8192xi8>
      %22 = arith.index_cast %21 : i8 to index
      %23 = memref.load %alloca_1[%22] : memref<256xi32>
      %24 = arith.addi %23, %c1_i32 : i32
      memref.store %24, %alloca_1[%22] : memref<256xi32>
      %25 = affine.load %arg0[%arg3 + symbol(%0) + 6] : memref<8192xi8>
      %26 = arith.index_cast %25 : i8 to index
      %27 = memref.load %alloca_0[%26] : memref<256xi32>
      %28 = arith.addi %27, %c1_i32 : i32
      memref.store %28, %alloca_0[%26] : memref<256xi32>
      %29 = affine.load %arg0[%arg3 + symbol(%0) + 7] : memref<8192xi8>
      %30 = arith.index_cast %29 : i8 to index
      %31 = memref.load %alloca[%30] : memref<256xi32>
      %32 = arith.addi %31, %c1_i32 : i32
      memref.store %32, %alloca[%30] : memref<256xi32>
    }
    affine.for %arg3 = 0 to 256 {
      %1 = affine.load %alloca_6[%arg3] : memref<256xi32>
      %2 = affine.load %alloca_5[%arg3] : memref<256xi32>
      %3 = arith.addi %1, %2 : i32
      %4 = affine.load %alloca_4[%arg3] : memref<256xi32>
      %5 = arith.addi %3, %4 : i32
      %6 = affine.load %alloca_3[%arg3] : memref<256xi32>
      %7 = arith.addi %5, %6 : i32
      %8 = affine.load %alloca_2[%arg3] : memref<256xi32>
      %9 = arith.addi %7, %8 : i32
      %10 = affine.load %alloca_1[%arg3] : memref<256xi32>
      %11 = arith.addi %9, %10 : i32
      %12 = affine.load %alloca_0[%arg3] : memref<256xi32>
      %13 = arith.addi %11, %12 : i32
      %14 = affine.load %alloca[%arg3] : memref<256xi32>
      %15 = arith.addi %13, %14 : i32
      %16 = affine.load %arg1[%arg3] : memref<256xi32>
      %17 = arith.addi %16, %15 : i32
      affine.store %17, %arg1[%arg3] : memref<256xi32>
    }
    return
  }
}
