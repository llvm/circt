module {
  func.func @twiddles8(%arg0: memref<8xf64>, %arg1: memref<8xf64>, %arg2: i32, %arg3: i32)  {
    %cst = arith.constant -6.2831853070000001 : f64
    %c7_i32 = arith.constant 7 : i32
    %c3_i32 = arith.constant 3 : i32
    %c5_i32 = arith.constant 5 : i32
    %c1_i32 = arith.constant 1 : i32
    %c6_i32 = arith.constant 6 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() {hls.preserve, polygeist.varname = "reversed8"} : memref<8xi32>
    affine.store %c0_i32, %alloca[0] : memref<8xi32>
    affine.store %c4_i32, %alloca[1] : memref<8xi32>
    affine.store %c2_i32, %alloca[2] : memref<8xi32>
    affine.store %c6_i32, %alloca[3] : memref<8xi32>
    affine.store %c1_i32, %alloca[4] : memref<8xi32>
    affine.store %c5_i32, %alloca[5] : memref<8xi32>
    affine.store %c3_i32, %alloca[6] : memref<8xi32>
    affine.store %c7_i32, %alloca[7] : memref<8xi32>
    %0 = arith.sitofp %arg3 : i32 to f64
    %1 = arith.sitofp %arg2 : i32 to f64
    affine.for %arg4 = 1 to 8 {
      %2 = affine.load %alloca[%arg4] : memref<8xi32>
      %3 = arith.sitofp %2 : i32 to f64
      %4 = arith.mulf %3, %cst : f64
      %5 = arith.divf %4, %0 : f64
      %6 = arith.mulf %5, %1 {polygeist.ssa_names = ["phi"]} : f64
      %7 = math.cos %6 {polygeist.ssa_names = ["phi_x"]} : f64
      %8 = math.sin %6 {polygeist.ssa_names = ["phi_y"]} : f64
      %9 = affine.load %arg0[%arg4] : memref<8xf64>
      %10 = arith.mulf %9, %7 : f64
      %11 = affine.load %arg1[%arg4] : memref<8xf64>
      %12 = arith.mulf %11, %8 : f64
      %13 = arith.subf %10, %12 : f64
      affine.store %13, %arg0[%arg4] : memref<8xf64>
      %14 = arith.mulf %9, %8 : f64
      %15 = affine.load %arg1[%arg4] : memref<8xf64>
      %16 = arith.mulf %15, %7 : f64
      %17 = arith.addf %14, %16 : f64
      affine.store %17, %arg1[%arg4] : memref<8xf64>
    }
    return
  }
  func.func @loadx8(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: i32, %arg3: i32)  {
    %c7_i32 = arith.constant 7 : i32
    %c6_i32 = arith.constant 6 : i32
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = affine.load %arg1[symbol(%0)] : memref<?xf64>
    affine.store %1, %arg0[0] : memref<?xf64>
    %2 = arith.addi %arg3, %arg2 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = affine.load %arg1[symbol(%3)] : memref<?xf64>
    affine.store %4, %arg0[1] : memref<?xf64>
    %5 = arith.muli %arg3, %c2_i32 : i32
    %6 = arith.addi %5, %arg2 : i32
    %7 = arith.index_cast %6 : i32 to index
    %8 = affine.load %arg1[symbol(%7)] : memref<?xf64>
    affine.store %8, %arg0[2] : memref<?xf64>
    %9 = arith.muli %arg3, %c3_i32 : i32
    %10 = arith.addi %9, %arg2 : i32
    %11 = arith.index_cast %10 : i32 to index
    %12 = affine.load %arg1[symbol(%11)] : memref<?xf64>
    affine.store %12, %arg0[3] : memref<?xf64>
    %13 = arith.muli %arg3, %c4_i32 : i32
    %14 = arith.addi %13, %arg2 : i32
    %15 = arith.index_cast %14 : i32 to index
    %16 = affine.load %arg1[symbol(%15)] : memref<?xf64>
    affine.store %16, %arg0[4] : memref<?xf64>
    %17 = arith.muli %arg3, %c5_i32 : i32
    %18 = arith.addi %17, %arg2 : i32
    %19 = arith.index_cast %18 : i32 to index
    %20 = affine.load %arg1[symbol(%19)] : memref<?xf64>
    affine.store %20, %arg0[5] : memref<?xf64>
    %21 = arith.muli %arg3, %c6_i32 : i32
    %22 = arith.addi %21, %arg2 : i32
    %23 = arith.index_cast %22 : i32 to index
    %24 = affine.load %arg1[symbol(%23)] : memref<?xf64>
    affine.store %24, %arg0[6] : memref<?xf64>
    %25 = arith.muli %arg3, %c7_i32 : i32
    %26 = arith.addi %25, %arg2 : i32
    %27 = arith.index_cast %26 : i32 to index
    %28 = affine.load %arg1[symbol(%27)] : memref<?xf64>
    affine.store %28, %arg0[7] : memref<?xf64>
    return
  }
  func.func @loady8(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: i32, %arg3: i32)  {
    %c7_i32 = arith.constant 7 : i32
    %c6_i32 = arith.constant 6 : i32
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = affine.load %arg1[symbol(%0)] : memref<?xf64>
    affine.store %1, %arg0[0] : memref<?xf64>
    %2 = arith.addi %arg3, %arg2 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = affine.load %arg1[symbol(%3)] : memref<?xf64>
    affine.store %4, %arg0[1] : memref<?xf64>
    %5 = arith.muli %arg3, %c2_i32 : i32
    %6 = arith.addi %5, %arg2 : i32
    %7 = arith.index_cast %6 : i32 to index
    %8 = affine.load %arg1[symbol(%7)] : memref<?xf64>
    affine.store %8, %arg0[2] : memref<?xf64>
    %9 = arith.muli %arg3, %c3_i32 : i32
    %10 = arith.addi %9, %arg2 : i32
    %11 = arith.index_cast %10 : i32 to index
    %12 = affine.load %arg1[symbol(%11)] : memref<?xf64>
    affine.store %12, %arg0[3] : memref<?xf64>
    %13 = arith.muli %arg3, %c4_i32 : i32
    %14 = arith.addi %13, %arg2 : i32
    %15 = arith.index_cast %14 : i32 to index
    %16 = affine.load %arg1[symbol(%15)] : memref<?xf64>
    affine.store %16, %arg0[4] : memref<?xf64>
    %17 = arith.muli %arg3, %c5_i32 : i32
    %18 = arith.addi %17, %arg2 : i32
    %19 = arith.index_cast %18 : i32 to index
    %20 = affine.load %arg1[symbol(%19)] : memref<?xf64>
    affine.store %20, %arg0[5] : memref<?xf64>
    %21 = arith.muli %arg3, %c6_i32 : i32
    %22 = arith.addi %21, %arg2 : i32
    %23 = arith.index_cast %22 : i32 to index
    %24 = affine.load %arg1[symbol(%23)] : memref<?xf64>
    affine.store %24, %arg0[6] : memref<?xf64>
    %25 = arith.muli %arg3, %c7_i32 : i32
    %26 = arith.addi %25, %arg2 : i32
    %27 = arith.index_cast %26 : i32 to index
    %28 = affine.load %arg1[symbol(%27)] : memref<?xf64>
    affine.store %28, %arg0[7] : memref<?xf64>
    return
  }
  func.func @fft1D_512(%arg0: memref<512xf64>, %arg1: memref<512xf64>)  {
    %cst = arith.constant 6.400000e+01 : f64
    %cst_0 = arith.constant 5.120000e+02 : f64
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    %c6_i32 = arith.constant 6 : i32
    %c1_i32 = arith.constant 1 : i32
    %c5_i32 = arith.constant 5 : i32
    %cst_1 = arith.constant -6.2831853070000001 : f64
    %c462_i32 = arith.constant 462 : i32
    %c198_i32 = arith.constant 198 : i32
    %c396_i32 = arith.constant 396 : i32
    %c132_i32 = arith.constant 132 : i32
    %c330_i32 = arith.constant 330 : i32
    %c264_i32 = arith.constant 264 : i32
    %c56_i32 = arith.constant 56 : i32
    %c24_i32 = arith.constant 24 : i32
    %c48_i32 = arith.constant 48 : i32
    %c16_i32 = arith.constant 16 : i32
    %c40_i32 = arith.constant 40 : i32
    %c32_i32 = arith.constant 32 : i32
    %c504_i32 = arith.constant 504 : i32
    %c216_i32 = arith.constant 216 : i32
    %c432_i32 = arith.constant 432 : i32
    %c144_i32 = arith.constant 144 : i32
    %c360_i32 = arith.constant 360 : i32
    %c288_i32 = arith.constant 288 : i32
    %c72_i32 = arith.constant {polygeist.ssa_names = ["sx"]} 72 : i32
    %c66_i32 = arith.constant {polygeist.ssa_names = ["sx"]} 66 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst_2 = arith.constant 0.70710678118654757 : f64
    %cst_3 = arith.constant 0.000000e+00 : f64
    %cst_4 = arith.constant -1.000000e+00 : f64
    %c7_i32 = arith.constant 7 : i32
    %c3_i32 = arith.constant 3 : i32
    %alloca = memref.alloca() {hls.preserve, polygeist.varname = "smem"} : memref<576xf64>
    %alloca_5 = memref.alloca() {hls.preserve, polygeist.varname = "data_y"} : memref<8xf64>
    %alloca_6 = memref.alloca() {hls.preserve, polygeist.varname = "data_x"} : memref<8xf64>
    %alloca_7 = memref.alloca() {hls.preserve, polygeist.varname = "DATA_y"} : memref<512xf64>
    %alloca_8 = memref.alloca() {hls.preserve, polygeist.varname = "DATA_x"} : memref<512xf64>
    %alloca_9 = memref.alloca() {hls.preserve, polygeist.varname = "reversed8"} : memref<8xi32>
    affine.for %arg2 = 0 to 64 {
      %0 = arith.index_cast %arg2 : index to i32
      %1 = affine.load %arg0[%arg2] : memref<512xf64>
      affine.store %1, %alloca_6[0] : memref<8xf64>
      %2 = affine.load %arg0[%arg2 + 64] : memref<512xf64>
      affine.store %2, %alloca_6[1] : memref<8xf64>
      %3 = affine.load %arg0[%arg2 + 128] : memref<512xf64>
      affine.store %3, %alloca_6[2] : memref<8xf64>
      %4 = affine.load %arg0[%arg2 + 192] : memref<512xf64>
      affine.store %4, %alloca_6[3] : memref<8xf64>
      %5 = affine.load %arg0[%arg2 + 256] : memref<512xf64>
      affine.store %5, %alloca_6[4] : memref<8xf64>
      %6 = affine.load %arg0[%arg2 + 320] : memref<512xf64>
      affine.store %6, %alloca_6[5] : memref<8xf64>
      %7 = affine.load %arg0[%arg2 + 384] : memref<512xf64>
      affine.store %7, %alloca_6[6] : memref<8xf64>
      %8 = affine.load %arg0[%arg2 + 448] : memref<512xf64>
      affine.store %8, %alloca_6[7] : memref<8xf64>
      %9 = affine.load %arg1[%arg2] : memref<512xf64>
      affine.store %9, %alloca_5[0] : memref<8xf64>
      %10 = affine.load %arg1[%arg2 + 64] : memref<512xf64>
      affine.store %10, %alloca_5[1] : memref<8xf64>
      %11 = affine.load %arg1[%arg2 + 128] : memref<512xf64>
      affine.store %11, %alloca_5[2] : memref<8xf64>
      %12 = affine.load %arg1[%arg2 + 192] : memref<512xf64>
      affine.store %12, %alloca_5[3] : memref<8xf64>
      %13 = affine.load %arg1[%arg2 + 256] : memref<512xf64>
      affine.store %13, %alloca_5[4] : memref<8xf64>
      %14 = affine.load %arg1[%arg2 + 320] : memref<512xf64>
      affine.store %14, %alloca_5[5] : memref<8xf64>
      %15 = affine.load %arg1[%arg2 + 384] : memref<512xf64>
      affine.store %15, %alloca_5[6] : memref<8xf64>
      %16 = affine.load %arg1[%arg2 + 448] : memref<512xf64>
      %17 = arith.addf %1, %5 : f64
      %18 = arith.addf %9, %13 : f64
      %19 = arith.subf %1, %5 : f64
      %20 = arith.subf %9, %13 : f64
      %21 = arith.addf %2, %6 : f64
      %22 = arith.addf %10, %14 : f64
      %23 = arith.subf %2, %6 : f64
      %24 = arith.subf %10, %14 : f64
      %25 = arith.addf %3, %7 : f64
      %26 = arith.addf %11, %15 : f64
      %27 = arith.subf %3, %7 : f64
      %28 = arith.subf %11, %15 : f64
      %29 = arith.addf %4, %8 : f64
      %30 = arith.addf %12, %16 : f64
      %31 = arith.subf %4, %8 : f64
      %32 = arith.subf %12, %16 : f64
      %33 = arith.mulf %24, %cst_4 : f64
      %34 = arith.subf %23, %33 : f64
      %35 = arith.mulf %34, %cst_2 : f64
      %36 = arith.mulf %23, %cst_4 : f64
      %37 = arith.addf %36, %24 : f64
      %38 = arith.mulf %37, %cst_2 : f64
      %39 = arith.mulf %27, %cst_3 : f64
      %40 = arith.mulf %28, %cst_4 : f64
      %41 = arith.subf %39, %40 : f64
      %42 = arith.mulf %27, %cst_4 : f64
      %43 = arith.mulf %28, %cst_3 : f64
      %44 = arith.addf %42, %43 : f64
      %45 = arith.mulf %31, %cst_4 : f64
      %46 = arith.mulf %32, %cst_4 : f64
      %47 = arith.subf %45, %46 : f64
      %48 = arith.mulf %47, %cst_2 : f64
      %49 = arith.addf %45, %46 : f64
      %50 = arith.mulf %49, %cst_2 : f64
      %51 = arith.addf %17, %25 : f64
      %52 = arith.addf %18, %26 : f64
      %53 = arith.subf %17, %25 : f64
      %54 = arith.subf %18, %26 : f64
      %55 = arith.addf %21, %29 : f64
      %56 = arith.addf %22, %30 : f64
      %57 = arith.subf %21, %29 : f64
      %58 = arith.subf %22, %30 : f64
      %59 = arith.mulf %57, %cst_3 : f64
      %60 = arith.mulf %58, %cst_4 : f64
      %61 = arith.subf %59, %60 : f64
      %62 = arith.mulf %57, %cst_4 : f64
      %63 = arith.mulf %58, %cst_3 : f64
      %64 = arith.subf %62, %63 : f64
      %65 = arith.addf %51, %55 : f64
      affine.store %65, %alloca_6[0] : memref<8xf64>
      %66 = arith.addf %52, %56 : f64
      affine.store %66, %alloca_5[0] : memref<8xf64>
      %67 = arith.subf %51, %55 : f64
      affine.store %67, %alloca_6[1] : memref<8xf64>
      %68 = arith.subf %52, %56 : f64
      affine.store %68, %alloca_5[1] : memref<8xf64>
      %69 = arith.addf %53, %61 : f64
      affine.store %69, %alloca_6[2] : memref<8xf64>
      %70 = arith.addf %54, %64 : f64
      affine.store %70, %alloca_5[2] : memref<8xf64>
      %71 = arith.subf %53, %61 : f64
      affine.store %71, %alloca_6[3] : memref<8xf64>
      %72 = arith.subf %54, %64 : f64
      affine.store %72, %alloca_5[3] : memref<8xf64>
      %73 = arith.addf %19, %41 : f64
      %74 = arith.addf %20, %44 : f64
      %75 = arith.subf %19, %41 : f64
      %76 = arith.subf %20, %44 : f64
      %77 = arith.addf %35, %48 : f64
      %78 = arith.addf %38, %50 : f64
      %79 = arith.subf %35, %48 : f64
      %80 = arith.subf %38, %50 : f64
      %81 = arith.mulf %79, %cst_3 : f64
      %82 = arith.mulf %80, %cst_4 : f64
      %83 = arith.subf %81, %82 : f64
      %84 = arith.mulf %79, %cst_4 : f64
      %85 = arith.mulf %80, %cst_3 : f64
      %86 = arith.subf %84, %85 : f64
      %87 = arith.addf %73, %77 : f64
      affine.store %87, %alloca_6[4] : memref<8xf64>
      %88 = arith.addf %74, %78 : f64
      affine.store %88, %alloca_5[4] : memref<8xf64>
      %89 = arith.subf %73, %77 : f64
      affine.store %89, %alloca_6[5] : memref<8xf64>
      %90 = arith.subf %74, %78 : f64
      affine.store %90, %alloca_5[5] : memref<8xf64>
      %91 = arith.addf %75, %83 : f64
      affine.store %91, %alloca_6[6] : memref<8xf64>
      %92 = arith.addf %76, %86 : f64
      affine.store %92, %alloca_5[6] : memref<8xf64>
      %93 = arith.subf %75, %83 : f64
      affine.store %93, %alloca_6[7] : memref<8xf64>
      %94 = arith.subf %76, %86 : f64
      affine.store %94, %alloca_5[7] : memref<8xf64>
      affine.store %c0_i32, %alloca_9[0] : memref<8xi32>
      affine.store %c4_i32, %alloca_9[1] : memref<8xi32>
      affine.store %c2_i32, %alloca_9[2] : memref<8xi32>
      affine.store %c6_i32, %alloca_9[3] : memref<8xi32>
      affine.store %c1_i32, %alloca_9[4] : memref<8xi32>
      affine.store %c5_i32, %alloca_9[5] : memref<8xi32>
      affine.store %c3_i32, %alloca_9[6] : memref<8xi32>
      affine.store %c7_i32, %alloca_9[7] : memref<8xi32>
      %95 = arith.sitofp %0 : i32 to f64
      affine.for %arg3 = 1 to 8 {
        %112 = affine.load %alloca_9[%arg3] : memref<8xi32>
        %113 = arith.sitofp %112 : i32 to f64
        %114 = arith.mulf %113, %cst_1 : f64
        %115 = arith.divf %114, %cst_0 : f64
        %116 = arith.mulf %115, %95 {polygeist.ssa_names = ["phi"]} : f64
        %117 = math.cos %116 {polygeist.ssa_names = ["phi_x"]} : f64
        %118 = math.sin %116 {polygeist.ssa_names = ["phi_y"]} : f64
        %119 = affine.load %alloca_6[%arg3] : memref<8xf64>
        %120 = arith.mulf %119, %117 : f64
        %121 = affine.load %alloca_5[%arg3] : memref<8xf64>
        %122 = arith.mulf %121, %118 : f64
        %123 = arith.subf %120, %122 : f64
        affine.store %123, %alloca_6[%arg3] : memref<8xf64>
        %124 = arith.mulf %119, %118 : f64
        %125 = arith.mulf %121, %117 : f64
        %126 = arith.addf %124, %125 : f64
        affine.store %126, %alloca_5[%arg3] : memref<8xf64>
      }
      %96 = affine.load %alloca_6[0] : memref<8xf64>
      affine.store %96, %alloca_8[%arg2 * 8] : memref<512xf64>
      %97 = affine.load %alloca_6[1] : memref<8xf64>
      affine.store %97, %alloca_8[%arg2 * 8 + 1] : memref<512xf64>
      %98 = affine.load %alloca_6[2] : memref<8xf64>
      affine.store %98, %alloca_8[%arg2 * 8 + 2] : memref<512xf64>
      %99 = affine.load %alloca_6[3] : memref<8xf64>
      affine.store %99, %alloca_8[%arg2 * 8 + 3] : memref<512xf64>
      %100 = affine.load %alloca_6[4] : memref<8xf64>
      affine.store %100, %alloca_8[%arg2 * 8 + 4] : memref<512xf64>
      %101 = affine.load %alloca_6[5] : memref<8xf64>
      affine.store %101, %alloca_8[%arg2 * 8 + 5] : memref<512xf64>
      %102 = affine.load %alloca_6[6] : memref<8xf64>
      affine.store %102, %alloca_8[%arg2 * 8 + 6] : memref<512xf64>
      %103 = affine.load %alloca_6[7] : memref<8xf64>
      affine.store %103, %alloca_8[%arg2 * 8 + 7] : memref<512xf64>
      %104 = affine.load %alloca_5[0] : memref<8xf64>
      affine.store %104, %alloca_7[%arg2 * 8] : memref<512xf64>
      %105 = affine.load %alloca_5[1] : memref<8xf64>
      affine.store %105, %alloca_7[%arg2 * 8 + 1] : memref<512xf64>
      %106 = affine.load %alloca_5[2] : memref<8xf64>
      affine.store %106, %alloca_7[%arg2 * 8 + 2] : memref<512xf64>
      %107 = affine.load %alloca_5[3] : memref<8xf64>
      affine.store %107, %alloca_7[%arg2 * 8 + 3] : memref<512xf64>
      %108 = affine.load %alloca_5[4] : memref<8xf64>
      affine.store %108, %alloca_7[%arg2 * 8 + 4] : memref<512xf64>
      %109 = affine.load %alloca_5[5] : memref<8xf64>
      affine.store %109, %alloca_7[%arg2 * 8 + 5] : memref<512xf64>
      %110 = affine.load %alloca_5[6] : memref<8xf64>
      affine.store %110, %alloca_7[%arg2 * 8 + 6] : memref<512xf64>
      %111 = affine.load %alloca_5[7] : memref<8xf64>
      affine.store %111, %alloca_7[%arg2 * 8 + 7] : memref<512xf64>
    }
    affine.for %arg2 = 0 to 64 {
      %0 = arith.index_cast %arg2 : index to i32
      %1 = arith.shrsi %0, %c3_i32 {polygeist.ssa_names = ["hi"]} : i32
      %2 = arith.andi %0, %c7_i32 {polygeist.ssa_names = ["lo"]} : i32
      %3 = arith.muli %1, %c8_i32 : i32
      %4 = arith.addi %3, %2 {polygeist.ssa_names = ["offset"]} : i32
      %5 = arith.index_cast %4 : i32 to index
      %6 = affine.load %alloca_8[%arg2 * 8] : memref<512xf64>
      memref.store %6, %alloca[%5] : memref<576xf64>
      %7 = arith.addi %4, %c264_i32 : i32
      %8 = arith.index_cast %7 : i32 to index
      %9 = affine.load %alloca_8[%arg2 * 8 + 1] : memref<512xf64>
      memref.store %9, %alloca[%8] : memref<576xf64>
      %10 = arith.addi %4, %c66_i32 : i32
      %11 = arith.index_cast %10 : i32 to index
      %12 = affine.load %alloca_8[%arg2 * 8 + 4] : memref<512xf64>
      memref.store %12, %alloca[%11] : memref<576xf64>
      %13 = arith.addi %4, %c330_i32 : i32
      %14 = arith.index_cast %13 : i32 to index
      %15 = affine.load %alloca_8[%arg2 * 8 + 5] : memref<512xf64>
      memref.store %15, %alloca[%14] : memref<576xf64>
      %16 = arith.addi %4, %c132_i32 : i32
      %17 = arith.index_cast %16 : i32 to index
      %18 = affine.load %alloca_8[%arg2 * 8 + 2] : memref<512xf64>
      memref.store %18, %alloca[%17] : memref<576xf64>
      %19 = arith.addi %4, %c396_i32 : i32
      %20 = arith.index_cast %19 : i32 to index
      %21 = affine.load %alloca_8[%arg2 * 8 + 3] : memref<512xf64>
      memref.store %21, %alloca[%20] : memref<576xf64>
      %22 = arith.addi %4, %c198_i32 : i32
      %23 = arith.index_cast %22 : i32 to index
      %24 = affine.load %alloca_8[%arg2 * 8 + 6] : memref<512xf64>
      memref.store %24, %alloca[%23] : memref<576xf64>
      %25 = arith.addi %4, %c462_i32 : i32
      %26 = arith.index_cast %25 : i32 to index
      %27 = affine.load %alloca_8[%arg2 * 8 + 7] : memref<512xf64>
      memref.store %27, %alloca[%26] : memref<576xf64>
    }
    affine.for %arg2 = 0 to 64 {
      %0 = arith.index_cast %arg2 : index to i32
      %1 = arith.shrsi %0, %c3_i32 {polygeist.ssa_names = ["hi"]} : i32
      %2 = arith.andi %0, %c7_i32 {polygeist.ssa_names = ["lo"]} : i32
      %3 = arith.muli %2, %c66_i32 : i32
      %4 = arith.addi %3, %1 {polygeist.ssa_names = ["offset"]} : i32
      %5 = arith.index_cast %4 : i32 to index
      %6 = memref.load %alloca[%5] : memref<576xf64>
      affine.store %6, %alloca_8[%arg2 * 8] : memref<512xf64>
      %7 = arith.addi %4, %c32_i32 : i32
      %8 = arith.index_cast %7 : i32 to index
      %9 = memref.load %alloca[%8] : memref<576xf64>
      affine.store %9, %alloca_8[%arg2 * 8 + 4] : memref<512xf64>
      %10 = arith.addi %4, %c8_i32 : i32
      %11 = arith.index_cast %10 : i32 to index
      %12 = memref.load %alloca[%11] : memref<576xf64>
      affine.store %12, %alloca_8[%arg2 * 8 + 1] : memref<512xf64>
      %13 = arith.addi %4, %c40_i32 : i32
      %14 = arith.index_cast %13 : i32 to index
      %15 = memref.load %alloca[%14] : memref<576xf64>
      affine.store %15, %alloca_8[%arg2 * 8 + 5] : memref<512xf64>
      %16 = arith.addi %4, %c16_i32 : i32
      %17 = arith.index_cast %16 : i32 to index
      %18 = memref.load %alloca[%17] : memref<576xf64>
      affine.store %18, %alloca_8[%arg2 * 8 + 2] : memref<512xf64>
      %19 = arith.addi %4, %c48_i32 : i32
      %20 = arith.index_cast %19 : i32 to index
      %21 = memref.load %alloca[%20] : memref<576xf64>
      affine.store %21, %alloca_8[%arg2 * 8 + 6] : memref<512xf64>
      %22 = arith.addi %4, %c24_i32 : i32
      %23 = arith.index_cast %22 : i32 to index
      %24 = memref.load %alloca[%23] : memref<576xf64>
      affine.store %24, %alloca_8[%arg2 * 8 + 3] : memref<512xf64>
      %25 = arith.addi %4, %c56_i32 : i32
      %26 = arith.index_cast %25 : i32 to index
      %27 = memref.load %alloca[%26] : memref<576xf64>
      affine.store %27, %alloca_8[%arg2 * 8 + 7] : memref<512xf64>
    }
    affine.for %arg2 = 0 to 64 {
      %0 = arith.index_cast %arg2 : index to i32
      %1 = arith.shrsi %0, %c3_i32 {polygeist.ssa_names = ["hi"]} : i32
      %2 = arith.andi %0, %c7_i32 {polygeist.ssa_names = ["lo"]} : i32
      %3 = arith.muli %1, %c8_i32 : i32
      %4 = arith.addi %3, %2 {polygeist.ssa_names = ["offset"]} : i32
      %5 = arith.index_cast %4 : i32 to index
      %6 = affine.load %alloca_7[%arg2 * 8] : memref<512xf64>
      memref.store %6, %alloca[%5] : memref<576xf64>
      %7 = arith.addi %4, %c264_i32 : i32
      %8 = arith.index_cast %7 : i32 to index
      %9 = affine.load %alloca_7[%arg2 * 8 + 1] : memref<512xf64>
      memref.store %9, %alloca[%8] : memref<576xf64>
      %10 = arith.addi %4, %c66_i32 : i32
      %11 = arith.index_cast %10 : i32 to index
      %12 = affine.load %alloca_7[%arg2 * 8 + 4] : memref<512xf64>
      memref.store %12, %alloca[%11] : memref<576xf64>
      %13 = arith.addi %4, %c330_i32 : i32
      %14 = arith.index_cast %13 : i32 to index
      %15 = affine.load %alloca_7[%arg2 * 8 + 5] : memref<512xf64>
      memref.store %15, %alloca[%14] : memref<576xf64>
      %16 = arith.addi %4, %c132_i32 : i32
      %17 = arith.index_cast %16 : i32 to index
      %18 = affine.load %alloca_7[%arg2 * 8 + 2] : memref<512xf64>
      memref.store %18, %alloca[%17] : memref<576xf64>
      %19 = arith.addi %4, %c396_i32 : i32
      %20 = arith.index_cast %19 : i32 to index
      %21 = affine.load %alloca_7[%arg2 * 8 + 3] : memref<512xf64>
      memref.store %21, %alloca[%20] : memref<576xf64>
      %22 = arith.addi %4, %c198_i32 : i32
      %23 = arith.index_cast %22 : i32 to index
      %24 = affine.load %alloca_7[%arg2 * 8 + 6] : memref<512xf64>
      memref.store %24, %alloca[%23] : memref<576xf64>
      %25 = arith.addi %4, %c462_i32 : i32
      %26 = arith.index_cast %25 : i32 to index
      %27 = affine.load %alloca_7[%arg2 * 8 + 7] : memref<512xf64>
      memref.store %27, %alloca[%26] : memref<576xf64>
    }
    affine.for %arg2 = 0 to 64 {
      %0 = arith.index_cast %arg2 : index to i32
      %1 = affine.load %alloca_7[%arg2 * 8] : memref<512xf64>
      affine.store %1, %alloca_5[0] : memref<8xf64>
      %2 = affine.load %alloca_7[%arg2 * 8 + 1] : memref<512xf64>
      affine.store %2, %alloca_5[1] : memref<8xf64>
      %3 = affine.load %alloca_7[%arg2 * 8 + 2] : memref<512xf64>
      affine.store %3, %alloca_5[2] : memref<8xf64>
      %4 = affine.load %alloca_7[%arg2 * 8 + 3] : memref<512xf64>
      affine.store %4, %alloca_5[3] : memref<8xf64>
      %5 = affine.load %alloca_7[%arg2 * 8 + 4] : memref<512xf64>
      affine.store %5, %alloca_5[4] : memref<8xf64>
      %6 = affine.load %alloca_7[%arg2 * 8 + 5] : memref<512xf64>
      affine.store %6, %alloca_5[5] : memref<8xf64>
      %7 = affine.load %alloca_7[%arg2 * 8 + 6] : memref<512xf64>
      affine.store %7, %alloca_5[6] : memref<8xf64>
      %8 = affine.load %alloca_7[%arg2 * 8 + 7] : memref<512xf64>
      affine.store %8, %alloca_5[7] : memref<8xf64>
      %9 = arith.shrsi %0, %c3_i32 {polygeist.ssa_names = ["hi"]} : i32
      %10 = arith.andi %0, %c7_i32 {polygeist.ssa_names = ["lo"]} : i32
      %11 = arith.muli %10, %c66_i32 : i32
      %12 = arith.addi %11, %9 : i32
      %13 = arith.index_cast %12 : i32 to index
      %14 = memref.load %alloca[%13] : memref<576xf64>
      affine.store %14, %alloca_5[0] : memref<8xf64>
      %15 = arith.addi %12, %c8_i32 : i32
      %16 = arith.index_cast %15 : i32 to index
      %17 = memref.load %alloca[%16] : memref<576xf64>
      affine.store %17, %alloca_5[1] : memref<8xf64>
      %18 = arith.addi %12, %c16_i32 : i32
      %19 = arith.index_cast %18 : i32 to index
      %20 = memref.load %alloca[%19] : memref<576xf64>
      affine.store %20, %alloca_5[2] : memref<8xf64>
      %21 = arith.addi %12, %c24_i32 : i32
      %22 = arith.index_cast %21 : i32 to index
      %23 = memref.load %alloca[%22] : memref<576xf64>
      affine.store %23, %alloca_5[3] : memref<8xf64>
      %24 = arith.addi %12, %c32_i32 : i32
      %25 = arith.index_cast %24 : i32 to index
      %26 = memref.load %alloca[%25] : memref<576xf64>
      affine.store %26, %alloca_5[4] : memref<8xf64>
      %27 = arith.addi %12, %c40_i32 : i32
      %28 = arith.index_cast %27 : i32 to index
      %29 = memref.load %alloca[%28] : memref<576xf64>
      affine.store %29, %alloca_5[5] : memref<8xf64>
      %30 = arith.addi %12, %c48_i32 : i32
      %31 = arith.index_cast %30 : i32 to index
      %32 = memref.load %alloca[%31] : memref<576xf64>
      affine.store %32, %alloca_5[6] : memref<8xf64>
      %33 = arith.addi %12, %c56_i32 : i32
      %34 = arith.index_cast %33 : i32 to index
      %35 = memref.load %alloca[%34] : memref<576xf64>
      affine.store %35, %alloca_5[7] : memref<8xf64>
      affine.store %14, %alloca_7[%arg2 * 8] : memref<512xf64>
      affine.store %17, %alloca_7[%arg2 * 8 + 1] : memref<512xf64>
      affine.store %20, %alloca_7[%arg2 * 8 + 2] : memref<512xf64>
      affine.store %23, %alloca_7[%arg2 * 8 + 3] : memref<512xf64>
      affine.store %26, %alloca_7[%arg2 * 8 + 4] : memref<512xf64>
      affine.store %29, %alloca_7[%arg2 * 8 + 5] : memref<512xf64>
      affine.store %32, %alloca_7[%arg2 * 8 + 6] : memref<512xf64>
      affine.store %35, %alloca_7[%arg2 * 8 + 7] : memref<512xf64>
    }
    %alloca_10 = memref.alloca() {hls.preserve, polygeist.varname = "reversed8"} : memref<8xi32>
    affine.for %arg2 = 0 to 64 {
      %0 = arith.index_cast %arg2 : index to i32
      %1 = affine.load %alloca_8[%arg2 * 8] : memref<512xf64>
      %2 = affine.load %alloca_8[%arg2 * 8 + 1] : memref<512xf64>
      %3 = affine.load %alloca_8[%arg2 * 8 + 2] : memref<512xf64>
      %4 = affine.load %alloca_8[%arg2 * 8 + 3] : memref<512xf64>
      %5 = affine.load %alloca_8[%arg2 * 8 + 4] : memref<512xf64>
      %6 = affine.load %alloca_8[%arg2 * 8 + 5] : memref<512xf64>
      %7 = affine.load %alloca_8[%arg2 * 8 + 6] : memref<512xf64>
      %8 = affine.load %alloca_8[%arg2 * 8 + 7] : memref<512xf64>
      %9 = affine.load %alloca_7[%arg2 * 8] : memref<512xf64>
      %10 = affine.load %alloca_7[%arg2 * 8 + 1] : memref<512xf64>
      %11 = affine.load %alloca_7[%arg2 * 8 + 2] : memref<512xf64>
      %12 = affine.load %alloca_7[%arg2 * 8 + 3] : memref<512xf64>
      %13 = affine.load %alloca_7[%arg2 * 8 + 4] : memref<512xf64>
      %14 = affine.load %alloca_7[%arg2 * 8 + 5] : memref<512xf64>
      %15 = affine.load %alloca_7[%arg2 * 8 + 6] : memref<512xf64>
      %16 = affine.load %alloca_7[%arg2 * 8 + 7] : memref<512xf64>
      %17 = arith.addf %1, %5 : f64
      %18 = arith.addf %9, %13 : f64
      %19 = arith.subf %1, %5 : f64
      %20 = arith.subf %9, %13 : f64
      %21 = arith.addf %2, %6 : f64
      %22 = arith.addf %10, %14 : f64
      %23 = arith.subf %2, %6 : f64
      %24 = arith.subf %10, %14 : f64
      %25 = arith.addf %3, %7 : f64
      %26 = arith.addf %11, %15 : f64
      %27 = arith.subf %3, %7 : f64
      %28 = arith.subf %11, %15 : f64
      %29 = arith.addf %4, %8 : f64
      %30 = arith.addf %12, %16 : f64
      %31 = arith.subf %4, %8 : f64
      %32 = arith.subf %12, %16 : f64
      %33 = arith.mulf %24, %cst_4 : f64
      %34 = arith.subf %23, %33 : f64
      %35 = arith.mulf %34, %cst_2 : f64
      %36 = arith.mulf %23, %cst_4 : f64
      %37 = arith.addf %36, %24 : f64
      %38 = arith.mulf %37, %cst_2 : f64
      %39 = arith.mulf %27, %cst_3 : f64
      %40 = arith.mulf %28, %cst_4 : f64
      %41 = arith.subf %39, %40 : f64
      %42 = arith.mulf %27, %cst_4 : f64
      %43 = arith.mulf %28, %cst_3 : f64
      %44 = arith.addf %42, %43 : f64
      %45 = arith.mulf %31, %cst_4 : f64
      %46 = arith.mulf %32, %cst_4 : f64
      %47 = arith.subf %45, %46 : f64
      %48 = arith.mulf %47, %cst_2 : f64
      %49 = arith.addf %45, %46 : f64
      %50 = arith.mulf %49, %cst_2 : f64
      %51 = arith.addf %17, %25 : f64
      %52 = arith.addf %18, %26 : f64
      %53 = arith.subf %17, %25 : f64
      %54 = arith.subf %18, %26 : f64
      %55 = arith.addf %21, %29 : f64
      %56 = arith.addf %22, %30 : f64
      %57 = arith.subf %21, %29 : f64
      %58 = arith.subf %22, %30 : f64
      %59 = arith.mulf %57, %cst_3 : f64
      %60 = arith.mulf %58, %cst_4 : f64
      %61 = arith.subf %59, %60 : f64
      %62 = arith.mulf %57, %cst_4 : f64
      %63 = arith.mulf %58, %cst_3 : f64
      %64 = arith.subf %62, %63 : f64
      %65 = arith.addf %51, %55 : f64
      affine.store %65, %alloca_6[0] : memref<8xf64>
      %66 = arith.addf %52, %56 : f64
      affine.store %66, %alloca_5[0] : memref<8xf64>
      %67 = arith.subf %51, %55 : f64
      affine.store %67, %alloca_6[1] : memref<8xf64>
      %68 = arith.subf %52, %56 : f64
      affine.store %68, %alloca_5[1] : memref<8xf64>
      %69 = arith.addf %53, %61 : f64
      affine.store %69, %alloca_6[2] : memref<8xf64>
      %70 = arith.addf %54, %64 : f64
      affine.store %70, %alloca_5[2] : memref<8xf64>
      %71 = arith.subf %53, %61 : f64
      affine.store %71, %alloca_6[3] : memref<8xf64>
      %72 = arith.subf %54, %64 : f64
      affine.store %72, %alloca_5[3] : memref<8xf64>
      %73 = arith.addf %19, %41 : f64
      %74 = arith.addf %20, %44 : f64
      %75 = arith.subf %19, %41 : f64
      %76 = arith.subf %20, %44 : f64
      %77 = arith.addf %35, %48 : f64
      %78 = arith.addf %38, %50 : f64
      %79 = arith.subf %35, %48 : f64
      %80 = arith.subf %38, %50 : f64
      %81 = arith.mulf %79, %cst_3 : f64
      %82 = arith.mulf %80, %cst_4 : f64
      %83 = arith.subf %81, %82 : f64
      %84 = arith.mulf %79, %cst_4 : f64
      %85 = arith.mulf %80, %cst_3 : f64
      %86 = arith.subf %84, %85 : f64
      %87 = arith.addf %73, %77 : f64
      affine.store %87, %alloca_6[4] : memref<8xf64>
      %88 = arith.addf %74, %78 : f64
      affine.store %88, %alloca_5[4] : memref<8xf64>
      %89 = arith.subf %73, %77 : f64
      affine.store %89, %alloca_6[5] : memref<8xf64>
      %90 = arith.subf %74, %78 : f64
      affine.store %90, %alloca_5[5] : memref<8xf64>
      %91 = arith.addf %75, %83 : f64
      affine.store %91, %alloca_6[6] : memref<8xf64>
      %92 = arith.addf %76, %86 : f64
      affine.store %92, %alloca_5[6] : memref<8xf64>
      %93 = arith.subf %75, %83 : f64
      affine.store %93, %alloca_6[7] : memref<8xf64>
      %94 = arith.subf %76, %86 : f64
      affine.store %94, %alloca_5[7] : memref<8xf64>
      %95 = arith.shrsi %0, %c3_i32 {polygeist.ssa_names = ["hi"]} : i32
      affine.store %c0_i32, %alloca_10[0] : memref<8xi32>
      affine.store %c4_i32, %alloca_10[1] : memref<8xi32>
      affine.store %c2_i32, %alloca_10[2] : memref<8xi32>
      affine.store %c6_i32, %alloca_10[3] : memref<8xi32>
      affine.store %c1_i32, %alloca_10[4] : memref<8xi32>
      affine.store %c5_i32, %alloca_10[5] : memref<8xi32>
      affine.store %c3_i32, %alloca_10[6] : memref<8xi32>
      affine.store %c7_i32, %alloca_10[7] : memref<8xi32>
      %96 = arith.sitofp %95 : i32 to f64
      affine.for %arg3 = 1 to 8 {
        %113 = affine.load %alloca_10[%arg3] : memref<8xi32>
        %114 = arith.sitofp %113 : i32 to f64
        %115 = arith.mulf %114, %cst_1 : f64
        %116 = arith.divf %115, %cst : f64
        %117 = arith.mulf %116, %96 {polygeist.ssa_names = ["phi"]} : f64
        %118 = math.cos %117 {polygeist.ssa_names = ["phi_x"]} : f64
        %119 = math.sin %117 {polygeist.ssa_names = ["phi_y"]} : f64
        %120 = affine.load %alloca_6[%arg3] : memref<8xf64>
        %121 = arith.mulf %120, %118 : f64
        %122 = affine.load %alloca_5[%arg3] : memref<8xf64>
        %123 = arith.mulf %122, %119 : f64
        %124 = arith.subf %121, %123 : f64
        affine.store %124, %alloca_6[%arg3] : memref<8xf64>
        %125 = arith.mulf %120, %119 : f64
        %126 = arith.mulf %122, %118 : f64
        %127 = arith.addf %125, %126 : f64
        affine.store %127, %alloca_5[%arg3] : memref<8xf64>
      }
      %97 = affine.load %alloca_6[0] : memref<8xf64>
      affine.store %97, %alloca_8[%arg2 * 8] : memref<512xf64>
      %98 = affine.load %alloca_6[1] : memref<8xf64>
      affine.store %98, %alloca_8[%arg2 * 8 + 1] : memref<512xf64>
      %99 = affine.load %alloca_6[2] : memref<8xf64>
      affine.store %99, %alloca_8[%arg2 * 8 + 2] : memref<512xf64>
      %100 = affine.load %alloca_6[3] : memref<8xf64>
      affine.store %100, %alloca_8[%arg2 * 8 + 3] : memref<512xf64>
      %101 = affine.load %alloca_6[4] : memref<8xf64>
      affine.store %101, %alloca_8[%arg2 * 8 + 4] : memref<512xf64>
      %102 = affine.load %alloca_6[5] : memref<8xf64>
      affine.store %102, %alloca_8[%arg2 * 8 + 5] : memref<512xf64>
      %103 = affine.load %alloca_6[6] : memref<8xf64>
      affine.store %103, %alloca_8[%arg2 * 8 + 6] : memref<512xf64>
      %104 = affine.load %alloca_6[7] : memref<8xf64>
      affine.store %104, %alloca_8[%arg2 * 8 + 7] : memref<512xf64>
      %105 = affine.load %alloca_5[0] : memref<8xf64>
      affine.store %105, %alloca_7[%arg2 * 8] : memref<512xf64>
      %106 = affine.load %alloca_5[1] : memref<8xf64>
      affine.store %106, %alloca_7[%arg2 * 8 + 1] : memref<512xf64>
      %107 = affine.load %alloca_5[2] : memref<8xf64>
      affine.store %107, %alloca_7[%arg2 * 8 + 2] : memref<512xf64>
      %108 = affine.load %alloca_5[3] : memref<8xf64>
      affine.store %108, %alloca_7[%arg2 * 8 + 3] : memref<512xf64>
      %109 = affine.load %alloca_5[4] : memref<8xf64>
      affine.store %109, %alloca_7[%arg2 * 8 + 4] : memref<512xf64>
      %110 = affine.load %alloca_5[5] : memref<8xf64>
      affine.store %110, %alloca_7[%arg2 * 8 + 5] : memref<512xf64>
      %111 = affine.load %alloca_5[6] : memref<8xf64>
      affine.store %111, %alloca_7[%arg2 * 8 + 6] : memref<512xf64>
      %112 = affine.load %alloca_5[7] : memref<8xf64>
      affine.store %112, %alloca_7[%arg2 * 8 + 7] : memref<512xf64>
    }
    affine.for %arg2 = 0 to 64 {
      %0 = arith.index_cast %arg2 : index to i32
      %1 = arith.shrsi %0, %c3_i32 {polygeist.ssa_names = ["hi"]} : i32
      %2 = arith.andi %0, %c7_i32 {polygeist.ssa_names = ["lo"]} : i32
      %3 = arith.muli %1, %c8_i32 : i32
      %4 = arith.addi %3, %2 {polygeist.ssa_names = ["offset"]} : i32
      %5 = arith.index_cast %4 : i32 to index
      %6 = affine.load %alloca_8[%arg2 * 8] : memref<512xf64>
      memref.store %6, %alloca[%5] : memref<576xf64>
      %7 = arith.addi %4, %c288_i32 : i32
      %8 = arith.index_cast %7 : i32 to index
      %9 = affine.load %alloca_8[%arg2 * 8 + 1] : memref<512xf64>
      memref.store %9, %alloca[%8] : memref<576xf64>
      %10 = arith.addi %4, %c72_i32 : i32
      %11 = arith.index_cast %10 : i32 to index
      %12 = affine.load %alloca_8[%arg2 * 8 + 4] : memref<512xf64>
      memref.store %12, %alloca[%11] : memref<576xf64>
      %13 = arith.addi %4, %c360_i32 : i32
      %14 = arith.index_cast %13 : i32 to index
      %15 = affine.load %alloca_8[%arg2 * 8 + 5] : memref<512xf64>
      memref.store %15, %alloca[%14] : memref<576xf64>
      %16 = arith.addi %4, %c144_i32 : i32
      %17 = arith.index_cast %16 : i32 to index
      %18 = affine.load %alloca_8[%arg2 * 8 + 2] : memref<512xf64>
      memref.store %18, %alloca[%17] : memref<576xf64>
      %19 = arith.addi %4, %c432_i32 : i32
      %20 = arith.index_cast %19 : i32 to index
      %21 = affine.load %alloca_8[%arg2 * 8 + 3] : memref<512xf64>
      memref.store %21, %alloca[%20] : memref<576xf64>
      %22 = arith.addi %4, %c216_i32 : i32
      %23 = arith.index_cast %22 : i32 to index
      %24 = affine.load %alloca_8[%arg2 * 8 + 6] : memref<512xf64>
      memref.store %24, %alloca[%23] : memref<576xf64>
      %25 = arith.addi %4, %c504_i32 : i32
      %26 = arith.index_cast %25 : i32 to index
      %27 = affine.load %alloca_8[%arg2 * 8 + 7] : memref<512xf64>
      memref.store %27, %alloca[%26] : memref<576xf64>
    }
    affine.for %arg2 = 0 to 64 {
      %0 = arith.index_cast %arg2 : index to i32
      %1 = arith.shrsi %0, %c3_i32 {polygeist.ssa_names = ["hi"]} : i32
      %2 = arith.andi %0, %c7_i32 {polygeist.ssa_names = ["lo"]} : i32
      %3 = arith.muli %1, %c72_i32 : i32
      %4 = arith.addi %3, %2 {polygeist.ssa_names = ["offset"]} : i32
      %5 = arith.index_cast %4 : i32 to index
      %6 = memref.load %alloca[%5] : memref<576xf64>
      affine.store %6, %alloca_8[%arg2 * 8] : memref<512xf64>
      %7 = arith.addi %4, %c32_i32 : i32
      %8 = arith.index_cast %7 : i32 to index
      %9 = memref.load %alloca[%8] : memref<576xf64>
      affine.store %9, %alloca_8[%arg2 * 8 + 4] : memref<512xf64>
      %10 = arith.addi %4, %c8_i32 : i32
      %11 = arith.index_cast %10 : i32 to index
      %12 = memref.load %alloca[%11] : memref<576xf64>
      affine.store %12, %alloca_8[%arg2 * 8 + 1] : memref<512xf64>
      %13 = arith.addi %4, %c40_i32 : i32
      %14 = arith.index_cast %13 : i32 to index
      %15 = memref.load %alloca[%14] : memref<576xf64>
      affine.store %15, %alloca_8[%arg2 * 8 + 5] : memref<512xf64>
      %16 = arith.addi %4, %c16_i32 : i32
      %17 = arith.index_cast %16 : i32 to index
      %18 = memref.load %alloca[%17] : memref<576xf64>
      affine.store %18, %alloca_8[%arg2 * 8 + 2] : memref<512xf64>
      %19 = arith.addi %4, %c48_i32 : i32
      %20 = arith.index_cast %19 : i32 to index
      %21 = memref.load %alloca[%20] : memref<576xf64>
      affine.store %21, %alloca_8[%arg2 * 8 + 6] : memref<512xf64>
      %22 = arith.addi %4, %c24_i32 : i32
      %23 = arith.index_cast %22 : i32 to index
      %24 = memref.load %alloca[%23] : memref<576xf64>
      affine.store %24, %alloca_8[%arg2 * 8 + 3] : memref<512xf64>
      %25 = arith.addi %4, %c56_i32 : i32
      %26 = arith.index_cast %25 : i32 to index
      %27 = memref.load %alloca[%26] : memref<576xf64>
      affine.store %27, %alloca_8[%arg2 * 8 + 7] : memref<512xf64>
    }
    affine.for %arg2 = 0 to 64 {
      %0 = arith.index_cast %arg2 : index to i32
      %1 = arith.shrsi %0, %c3_i32 {polygeist.ssa_names = ["hi"]} : i32
      %2 = arith.andi %0, %c7_i32 {polygeist.ssa_names = ["lo"]} : i32
      %3 = arith.muli %1, %c8_i32 : i32
      %4 = arith.addi %3, %2 {polygeist.ssa_names = ["offset"]} : i32
      %5 = arith.index_cast %4 : i32 to index
      %6 = affine.load %alloca_7[%arg2 * 8] : memref<512xf64>
      memref.store %6, %alloca[%5] : memref<576xf64>
      %7 = arith.addi %4, %c288_i32 : i32
      %8 = arith.index_cast %7 : i32 to index
      %9 = affine.load %alloca_7[%arg2 * 8 + 1] : memref<512xf64>
      memref.store %9, %alloca[%8] : memref<576xf64>
      %10 = arith.addi %4, %c72_i32 : i32
      %11 = arith.index_cast %10 : i32 to index
      %12 = affine.load %alloca_7[%arg2 * 8 + 4] : memref<512xf64>
      memref.store %12, %alloca[%11] : memref<576xf64>
      %13 = arith.addi %4, %c360_i32 : i32
      %14 = arith.index_cast %13 : i32 to index
      %15 = affine.load %alloca_7[%arg2 * 8 + 5] : memref<512xf64>
      memref.store %15, %alloca[%14] : memref<576xf64>
      %16 = arith.addi %4, %c144_i32 : i32
      %17 = arith.index_cast %16 : i32 to index
      %18 = affine.load %alloca_7[%arg2 * 8 + 2] : memref<512xf64>
      memref.store %18, %alloca[%17] : memref<576xf64>
      %19 = arith.addi %4, %c432_i32 : i32
      %20 = arith.index_cast %19 : i32 to index
      %21 = affine.load %alloca_7[%arg2 * 8 + 3] : memref<512xf64>
      memref.store %21, %alloca[%20] : memref<576xf64>
      %22 = arith.addi %4, %c216_i32 : i32
      %23 = arith.index_cast %22 : i32 to index
      %24 = affine.load %alloca_7[%arg2 * 8 + 6] : memref<512xf64>
      memref.store %24, %alloca[%23] : memref<576xf64>
      %25 = arith.addi %4, %c504_i32 : i32
      %26 = arith.index_cast %25 : i32 to index
      %27 = affine.load %alloca_7[%arg2 * 8 + 7] : memref<512xf64>
      memref.store %27, %alloca[%26] : memref<576xf64>
    }
    affine.for %arg2 = 0 to 64 {
      %0 = arith.index_cast %arg2 : index to i32
      %1 = affine.load %alloca_7[%arg2 * 8] : memref<512xf64>
      affine.store %1, %alloca_5[0] : memref<8xf64>
      %2 = affine.load %alloca_7[%arg2 * 8 + 1] : memref<512xf64>
      affine.store %2, %alloca_5[1] : memref<8xf64>
      %3 = affine.load %alloca_7[%arg2 * 8 + 2] : memref<512xf64>
      affine.store %3, %alloca_5[2] : memref<8xf64>
      %4 = affine.load %alloca_7[%arg2 * 8 + 3] : memref<512xf64>
      affine.store %4, %alloca_5[3] : memref<8xf64>
      %5 = affine.load %alloca_7[%arg2 * 8 + 4] : memref<512xf64>
      affine.store %5, %alloca_5[4] : memref<8xf64>
      %6 = affine.load %alloca_7[%arg2 * 8 + 5] : memref<512xf64>
      affine.store %6, %alloca_5[5] : memref<8xf64>
      %7 = affine.load %alloca_7[%arg2 * 8 + 6] : memref<512xf64>
      affine.store %7, %alloca_5[6] : memref<8xf64>
      %8 = affine.load %alloca_7[%arg2 * 8 + 7] : memref<512xf64>
      affine.store %8, %alloca_5[7] : memref<8xf64>
      %9 = arith.shrsi %0, %c3_i32 {polygeist.ssa_names = ["hi"]} : i32
      %10 = arith.andi %0, %c7_i32 {polygeist.ssa_names = ["lo"]} : i32
      %11 = arith.muli %9, %c72_i32 : i32
      %12 = arith.addi %11, %10 : i32
      %13 = arith.index_cast %12 : i32 to index
      %14 = memref.load %alloca[%13] : memref<576xf64>
      affine.store %14, %alloca_5[0] : memref<8xf64>
      %15 = arith.addi %12, %c8_i32 : i32
      %16 = arith.index_cast %15 : i32 to index
      %17 = memref.load %alloca[%16] : memref<576xf64>
      affine.store %17, %alloca_5[1] : memref<8xf64>
      %18 = arith.addi %12, %c16_i32 : i32
      %19 = arith.index_cast %18 : i32 to index
      %20 = memref.load %alloca[%19] : memref<576xf64>
      affine.store %20, %alloca_5[2] : memref<8xf64>
      %21 = arith.addi %12, %c24_i32 : i32
      %22 = arith.index_cast %21 : i32 to index
      %23 = memref.load %alloca[%22] : memref<576xf64>
      affine.store %23, %alloca_5[3] : memref<8xf64>
      %24 = arith.addi %12, %c32_i32 : i32
      %25 = arith.index_cast %24 : i32 to index
      %26 = memref.load %alloca[%25] : memref<576xf64>
      affine.store %26, %alloca_5[4] : memref<8xf64>
      %27 = arith.addi %12, %c40_i32 : i32
      %28 = arith.index_cast %27 : i32 to index
      %29 = memref.load %alloca[%28] : memref<576xf64>
      affine.store %29, %alloca_5[5] : memref<8xf64>
      %30 = arith.addi %12, %c48_i32 : i32
      %31 = arith.index_cast %30 : i32 to index
      %32 = memref.load %alloca[%31] : memref<576xf64>
      affine.store %32, %alloca_5[6] : memref<8xf64>
      %33 = arith.addi %12, %c56_i32 : i32
      %34 = arith.index_cast %33 : i32 to index
      %35 = memref.load %alloca[%34] : memref<576xf64>
      affine.store %35, %alloca_5[7] : memref<8xf64>
      affine.store %14, %alloca_7[%arg2 * 8] : memref<512xf64>
      affine.store %17, %alloca_7[%arg2 * 8 + 1] : memref<512xf64>
      affine.store %20, %alloca_7[%arg2 * 8 + 2] : memref<512xf64>
      affine.store %23, %alloca_7[%arg2 * 8 + 3] : memref<512xf64>
      affine.store %26, %alloca_7[%arg2 * 8 + 4] : memref<512xf64>
      affine.store %29, %alloca_7[%arg2 * 8 + 5] : memref<512xf64>
      affine.store %32, %alloca_7[%arg2 * 8 + 6] : memref<512xf64>
      affine.store %35, %alloca_7[%arg2 * 8 + 7] : memref<512xf64>
    }
    affine.for %arg2 = 0 to 64 {
      %0 = affine.load %alloca_7[%arg2 * 8] : memref<512xf64>
      %1 = affine.load %alloca_7[%arg2 * 8 + 1] : memref<512xf64>
      %2 = affine.load %alloca_7[%arg2 * 8 + 2] : memref<512xf64>
      %3 = affine.load %alloca_7[%arg2 * 8 + 3] : memref<512xf64>
      %4 = affine.load %alloca_7[%arg2 * 8 + 4] : memref<512xf64>
      %5 = affine.load %alloca_7[%arg2 * 8 + 5] : memref<512xf64>
      %6 = affine.load %alloca_7[%arg2 * 8 + 6] : memref<512xf64>
      %7 = affine.load %alloca_7[%arg2 * 8 + 7] : memref<512xf64>
      %8 = affine.load %alloca_8[%arg2 * 8] : memref<512xf64>
      %9 = affine.load %alloca_8[%arg2 * 8 + 1] : memref<512xf64>
      %10 = affine.load %alloca_8[%arg2 * 8 + 2] : memref<512xf64>
      %11 = affine.load %alloca_8[%arg2 * 8 + 3] : memref<512xf64>
      %12 = affine.load %alloca_8[%arg2 * 8 + 4] : memref<512xf64>
      %13 = affine.load %alloca_8[%arg2 * 8 + 5] : memref<512xf64>
      %14 = affine.load %alloca_8[%arg2 * 8 + 6] : memref<512xf64>
      %15 = affine.load %alloca_8[%arg2 * 8 + 7] : memref<512xf64>
      %16 = arith.addf %8, %12 : f64
      %17 = arith.addf %0, %4 : f64
      %18 = arith.subf %8, %12 : f64
      %19 = arith.subf %0, %4 : f64
      %20 = arith.addf %9, %13 : f64
      %21 = arith.addf %1, %5 : f64
      %22 = arith.subf %9, %13 : f64
      %23 = arith.subf %1, %5 : f64
      %24 = arith.addf %10, %14 : f64
      %25 = arith.addf %2, %6 : f64
      %26 = arith.subf %10, %14 : f64
      %27 = arith.subf %2, %6 : f64
      %28 = arith.addf %11, %15 : f64
      %29 = arith.addf %3, %7 : f64
      %30 = arith.subf %11, %15 : f64
      %31 = arith.subf %3, %7 : f64
      %32 = arith.mulf %23, %cst_4 : f64
      %33 = arith.subf %22, %32 : f64
      %34 = arith.mulf %33, %cst_2 : f64
      %35 = arith.mulf %22, %cst_4 : f64
      %36 = arith.addf %35, %23 : f64
      %37 = arith.mulf %36, %cst_2 : f64
      %38 = arith.mulf %26, %cst_3 : f64
      %39 = arith.mulf %27, %cst_4 : f64
      %40 = arith.subf %38, %39 : f64
      %41 = arith.mulf %26, %cst_4 : f64
      %42 = arith.mulf %27, %cst_3 : f64
      %43 = arith.addf %41, %42 : f64
      %44 = arith.mulf %30, %cst_4 : f64
      %45 = arith.mulf %31, %cst_4 : f64
      %46 = arith.subf %44, %45 : f64
      %47 = arith.mulf %46, %cst_2 : f64
      %48 = arith.addf %44, %45 : f64
      %49 = arith.mulf %48, %cst_2 : f64
      %50 = arith.addf %16, %24 : f64
      %51 = arith.addf %17, %25 : f64
      %52 = arith.subf %16, %24 : f64
      %53 = arith.subf %17, %25 : f64
      %54 = arith.addf %20, %28 : f64
      %55 = arith.addf %21, %29 : f64
      %56 = arith.subf %20, %28 : f64
      %57 = arith.subf %21, %29 : f64
      %58 = arith.mulf %56, %cst_3 : f64
      %59 = arith.mulf %57, %cst_4 : f64
      %60 = arith.subf %58, %59 : f64
      %61 = arith.mulf %56, %cst_4 : f64
      %62 = arith.mulf %57, %cst_3 : f64
      %63 = arith.subf %61, %62 : f64
      %64 = arith.addf %50, %54 : f64
      affine.store %64, %alloca_6[0] : memref<8xf64>
      %65 = arith.addf %51, %55 : f64
      affine.store %65, %alloca_5[0] : memref<8xf64>
      %66 = arith.subf %50, %54 : f64
      affine.store %66, %alloca_6[1] : memref<8xf64>
      %67 = arith.subf %51, %55 : f64
      affine.store %67, %alloca_5[1] : memref<8xf64>
      %68 = arith.addf %52, %60 : f64
      affine.store %68, %alloca_6[2] : memref<8xf64>
      %69 = arith.addf %53, %63 : f64
      affine.store %69, %alloca_5[2] : memref<8xf64>
      %70 = arith.subf %52, %60 : f64
      affine.store %70, %alloca_6[3] : memref<8xf64>
      %71 = arith.subf %53, %63 : f64
      affine.store %71, %alloca_5[3] : memref<8xf64>
      %72 = arith.addf %18, %40 : f64
      %73 = arith.addf %19, %43 : f64
      %74 = arith.subf %18, %40 : f64
      %75 = arith.subf %19, %43 : f64
      %76 = arith.addf %34, %47 : f64
      %77 = arith.addf %37, %49 : f64
      %78 = arith.subf %34, %47 : f64
      %79 = arith.subf %37, %49 : f64
      %80 = arith.mulf %78, %cst_3 : f64
      %81 = arith.mulf %79, %cst_4 : f64
      %82 = arith.subf %80, %81 : f64
      %83 = arith.mulf %78, %cst_4 : f64
      %84 = arith.mulf %79, %cst_3 : f64
      %85 = arith.subf %83, %84 : f64
      %86 = arith.addf %72, %76 : f64
      affine.store %86, %alloca_6[4] : memref<8xf64>
      %87 = arith.addf %73, %77 : f64
      affine.store %87, %alloca_5[4] : memref<8xf64>
      %88 = arith.subf %72, %76 : f64
      affine.store %88, %alloca_6[5] : memref<8xf64>
      %89 = arith.subf %73, %77 : f64
      affine.store %89, %alloca_5[5] : memref<8xf64>
      %90 = arith.addf %74, %82 : f64
      affine.store %90, %alloca_6[6] : memref<8xf64>
      %91 = arith.addf %75, %85 : f64
      affine.store %91, %alloca_5[6] : memref<8xf64>
      %92 = arith.subf %74, %82 : f64
      affine.store %92, %alloca_6[7] : memref<8xf64>
      %93 = arith.subf %75, %85 : f64
      affine.store %93, %alloca_5[7] : memref<8xf64>
      affine.store %64, %arg0[%arg2] : memref<512xf64>
      affine.store %86, %arg0[%arg2 + 64] : memref<512xf64>
      affine.store %68, %arg0[%arg2 + 128] : memref<512xf64>
      affine.store %90, %arg0[%arg2 + 192] : memref<512xf64>
      affine.store %66, %arg0[%arg2 + 256] : memref<512xf64>
      affine.store %88, %arg0[%arg2 + 320] : memref<512xf64>
      affine.store %70, %arg0[%arg2 + 384] : memref<512xf64>
      affine.store %92, %arg0[%arg2 + 448] : memref<512xf64>
      affine.store %65, %arg1[%arg2] : memref<512xf64>
      affine.store %87, %arg1[%arg2 + 64] : memref<512xf64>
      affine.store %69, %arg1[%arg2 + 128] : memref<512xf64>
      affine.store %91, %arg1[%arg2 + 192] : memref<512xf64>
      affine.store %67, %arg1[%arg2 + 256] : memref<512xf64>
      affine.store %89, %arg1[%arg2 + 320] : memref<512xf64>
      affine.store %71, %arg1[%arg2 + 384] : memref<512xf64>
      affine.store %93, %arg1[%arg2 + 448] : memref<512xf64>
    }
    return
  }
}
