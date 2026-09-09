#set = affine_set<(d0) : (-d0 + 30 >= 0)>
module {
  func.func @_Z7sobel_yPA1920_yS0_(%arg0: memref<1080x1920xi64>, %arg1: memref<1080x1920xi64>)  {
    %false = arith.constant false
    %true = arith.constant true
    %c1080_i32 = arith.constant 1080 : i32
    %c128_i32 = arith.constant 128 : i32
    %c25_i32 = arith.constant 25 : i32
    %c129_i32 = arith.constant 129 : i32
    %c66_i32 = arith.constant 66 : i32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c255_i32 = arith.constant 255 : i32
    %c3_i32 = arith.constant 3 : i32
    %c32_i32 = arith.constant 32 : i32
    %c2_i32 = arith.constant 2 : i32
    %c-2_i32 = arith.constant -2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %0 = llvm.mlir.undef : i32
    %alloca = memref.alloca() {hls.preserve, polygeist.varname = "image_reg"} : memref<3x3xi32>
    %alloca_0 = memref.alloca() {hls.preserve, polygeist.varname = "sobel_x_kernel"} : memref<3x3xi32>
    %alloca_1 = memref.alloca() {hls.preserve, polygeist.varname = "sobel_y_kernel"} : memref<3x3xi32>
    affine.store %c-1_i32, %alloca_1[0, 0] : memref<3x3xi32>
    affine.store %c0_i32, %alloca_1[0, 1] : memref<3x3xi32>
    affine.store %c1_i32, %alloca_1[0, 2] : memref<3x3xi32>
    affine.store %c-2_i32, %alloca_1[1, 0] : memref<3x3xi32>
    affine.store %c0_i32, %alloca_1[1, 1] : memref<3x3xi32>
    affine.store %c2_i32, %alloca_1[1, 2] : memref<3x3xi32>
    affine.store %c-1_i32, %alloca_1[2, 0] : memref<3x3xi32>
    affine.store %c0_i32, %alloca_1[2, 1] : memref<3x3xi32>
    affine.store %c1_i32, %alloca_1[2, 2] : memref<3x3xi32>
    affine.store %c-1_i32, %alloca_0[0, 0] : memref<3x3xi32>
    affine.store %c-2_i32, %alloca_0[0, 1] : memref<3x3xi32>
    affine.store %c-1_i32, %alloca_0[0, 2] : memref<3x3xi32>
    affine.store %c0_i32, %alloca_0[1, 0] : memref<3x3xi32>
    affine.store %c0_i32, %alloca_0[1, 1] : memref<3x3xi32>
    affine.store %c0_i32, %alloca_0[1, 2] : memref<3x3xi32>
    affine.store %c1_i32, %alloca_0[2, 0] : memref<3x3xi32>
    affine.store %c2_i32, %alloca_0[2, 1] : memref<3x3xi32>
    affine.store %c1_i32, %alloca_0[2, 2] : memref<3x3xi32>
    %1:6 = affine.for %arg2 = 1 to 1919 iter_args(%arg3 = %0, %arg4 = %0, %arg5 = %0, %arg6 = %0, %arg7 = %0, %arg8 = %0) -> (i32, i32, i32, i32, i32, i32) {
      %2:6 = affine.for %arg9 = 1 to 1079 step 32 iter_args(%arg10 = %arg3, %arg11 = %arg4, %arg12 = %arg5, %arg13 = %arg6, %arg14 = %arg7, %arg15 = %arg8) -> (i32, i32, i32, i32, i32, i32) {
        %3 = arith.index_cast %arg9 : index to i32
        %4 = affine.load %arg0[%arg9 - 1, %arg2 - 1] : memref<1080x1920xi64>
        %5 = arith.trunci %4 : i64 to i32
        affine.store %5, %alloca[0, 0] : memref<3x3xi32>
        %6 = affine.load %arg0[%arg9 - 1, %arg2] : memref<1080x1920xi64>
        %7 = arith.trunci %6 : i64 to i32
        affine.store %7, %alloca[0, 1] : memref<3x3xi32>
        %8 = affine.load %arg0[%arg9 - 1, %arg2 + 1] : memref<1080x1920xi64>
        %9 = arith.trunci %8 : i64 to i32
        affine.store %9, %alloca[0, 2] : memref<3x3xi32>
        %10 = affine.load %arg0[%arg9, %arg2 - 1] : memref<1080x1920xi64>
        %11 = arith.trunci %10 : i64 to i32
        affine.store %11, %alloca[1, 0] : memref<3x3xi32>
        %12 = affine.load %arg0[%arg9, %arg2] : memref<1080x1920xi64>
        %13 = arith.trunci %12 : i64 to i32
        affine.store %13, %alloca[1, 1] : memref<3x3xi32>
        %14 = affine.load %arg0[%arg9, %arg2 + 1] : memref<1080x1920xi64>
        %15 = arith.trunci %14 : i64 to i32
        affine.store %15, %alloca[1, 2] : memref<3x3xi32>
        %16 = affine.load %arg0[%arg9 + 1, %arg2 - 1] : memref<1080x1920xi64>
        %17 = arith.trunci %16 : i64 to i32
        affine.store %17, %alloca[2, 0] : memref<3x3xi32>
        %18 = affine.load %arg0[%arg9 + 1, %arg2] : memref<1080x1920xi64>
        %19 = arith.trunci %18 : i64 to i32
        affine.store %19, %alloca[2, 1] : memref<3x3xi32>
        %20 = affine.load %arg0[%arg9 + 1, %arg2 + 1] : memref<1080x1920xi64>
        %21 = arith.trunci %20 : i64 to i32
        affine.store %21, %alloca[2, 2] : memref<3x3xi32>
        %22:13 = affine.for %arg16 = 0 to 32 iter_args(%arg17 = %arg10, %arg18 = %arg11, %arg19 = %arg12, %arg20 = %arg13, %arg21 = %arg14, %arg22 = %arg15, %arg23 = %11, %arg24 = %13, %arg25 = %15, %arg26 = %17, %arg27 = %19, %arg28 = %21, %arg29 = %true) -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1) {
          %23 = arith.select %arg29, %c3_i32, %arg18 : i32
          %24:12 = scf.if %arg29 -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1) {
            %25 = arith.index_cast %arg16 : index to i32
            %26:3 = affine.for %arg30 = 0 to 3 iter_args(%arg31 = %c0_i32, %arg32 = %c0_i32, %arg33 = %arg17) -> (i32, i32, i32) {
              %44:2 = affine.for %arg34 = 0 to 3 iter_args(%arg35 = %arg31, %arg36 = %arg32) -> (i32, i32) {
                %45 = affine.load %alloca[%arg30, %arg34] : memref<3x3xi32>
                %46 = arith.andi %45, %c255_i32 {polygeist.ssa_names = ["b"]} : i32
                %47 = arith.shrui %45, %c8_i32 : i32
                %48 = arith.andi %47, %c255_i32 {polygeist.ssa_names = ["g"]} : i32
                %49 = arith.shrui %45, %c16_i32 : i32
                %50 = arith.andi %49, %c255_i32 {polygeist.ssa_names = ["r"]} : i32
                %51 = arith.muli %50, %c66_i32 : i32
                %52 = arith.muli %48, %c129_i32 : i32
                %53 = arith.addi %51, %52 : i32
                %54 = arith.muli %46, %c25_i32 : i32
                %55 = arith.addi %53, %54 {polygeist.ssa_names = ["luma"]} : i32
                %56 = arith.addi %55, %c128_i32 : i32
                %57 = arith.shrsi %56, %c8_i32 {polygeist.ssa_names = ["luma"]} : i32
                %58 = arith.addi %57, %c16_i32 {polygeist.ssa_names = ["luma"]} : i32
                %59 = affine.load %alloca_0[%arg30, %arg34] : memref<3x3xi32>
                %60 = arith.muli %59, %58 : i32
                %61 = arith.addi %arg36, %60 {polygeist.ssa_names = ["resultx"]} : i32
                %62 = affine.load %alloca_1[%arg30, %arg34] : memref<3x3xi32>
                %63 = arith.muli %62, %58 : i32
                %64 = arith.addi %arg35, %63 {polygeist.ssa_names = ["resulty"]} : i32
                affine.yield %64, %61 : i32, i32
              }
              affine.yield %44#0, %44#1, %c3_i32 : i32, i32, i32
            }
            %27 = arith.cmpi sgt, %26#1, %c0_i32 : i32
            %28 = scf.if %27 -> (i32) {
              scf.yield %26#1 : i32
            } else {
              %44 = arith.muli %26#1, %c-1_i32 : i32
              scf.yield %44 : i32
            } {polygeist.ssa_names = ["resultx"]}
            %29 = arith.cmpi sgt, %26#0, %c0_i32 : i32
            %30 = scf.if %29 -> (i32) {
              scf.yield %26#0 : i32
            } else {
              %44 = arith.muli %26#0, %c-1_i32 : i32
              scf.yield %44 : i32
            } {polygeist.ssa_names = ["resulty"]}
            %31 = arith.addi %28, %30 {polygeist.ssa_names = ["temp"]} : i32
            %32 = arith.cmpi sgt, %31, %c32_i32 : i32
            %33 = arith.extui %32 : i1 to i32
            %34 = arith.extui %32 : i1 to i64
            affine.store %34, %arg1[%arg9 + %arg16, %arg2] : memref<1080x1920xi64>
            %35 = arith.addi %25, %c1_i32 : i32
            %36 = arith.cmpi slt, %35, %c32_i32 : i32
            %37 = arith.select %36, %arg26, %arg23 : i32
            %38 = arith.select %36, %arg27, %arg24 : i32
            %39 = arith.select %36, %arg28, %arg25 : i32
            %40:3 = affine.if #set(%arg16) -> (i32, i32, i32) {
              affine.store %arg23, %alloca[0, 0] : memref<3x3xi32>
              affine.store %arg24, %alloca[0, 1] : memref<3x3xi32>
              affine.store %arg25, %alloca[0, 2] : memref<3x3xi32>
              affine.store %arg26, %alloca[1, 0] : memref<3x3xi32>
              affine.store %arg27, %alloca[1, 1] : memref<3x3xi32>
              affine.store %arg28, %alloca[1, 2] : memref<3x3xi32>
              %44 = affine.load %arg0[%arg9 + %arg16 + 2, %arg2 - 1] : memref<1080x1920xi64>
              %45 = arith.trunci %44 : i64 to i32
              affine.store %45, %alloca[2, 0] : memref<3x3xi32>
              %46 = affine.load %arg0[%arg9 + %arg16 + 2, %arg2] : memref<1080x1920xi64>
              %47 = arith.trunci %46 : i64 to i32
              affine.store %47, %alloca[2, 1] : memref<3x3xi32>
              %48 = affine.load %arg0[%arg9 + %arg16 + 2, %arg2 + 1] : memref<1080x1920xi64>
              %49 = arith.trunci %48 : i64 to i32
              affine.store %49, %alloca[2, 2] : memref<3x3xi32>
              affine.yield %45, %47, %49 : i32, i32, i32
            } else {
              affine.yield %arg26, %arg27, %arg28 : i32, i32, i32
            }
            %41 = arith.addi %3, %25 : i32
            %42 = arith.addi %41, %c2_i32 : i32
            %43 = arith.cmpi ne, %42, %c1080_i32 : i32
            scf.yield %26#2, %30, %28, %31, %33, %37, %38, %39, %40#0, %40#1, %40#2, %43 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1
          } else {
            scf.yield %arg17, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %arg28, %false : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1
          }
          affine.yield %24#0, %23, %24#1, %24#2, %24#3, %24#4, %24#5, %24#6, %24#7, %24#8, %24#9, %24#10, %24#11 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1
        } {hls.pipeline = [{II = 0 : i32, rewind = false, style = "stp"}]}
        affine.yield %22#0, %22#1, %22#2, %22#3, %22#4, %22#5 : i32, i32, i32, i32, i32, i32
      }
      affine.yield %2#0, %2#1, %2#2, %2#3, %2#4, %2#5 : i32, i32, i32, i32, i32, i32
    }
    return
  }
}
