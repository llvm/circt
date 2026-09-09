module {
 func.func @local_scan(%arg0: memref<2048xi32>)  {
    affine.for %arg1 = 0 to 128 {
      affine.for %arg2 = 1 to 16 {
        %0 = affine.load %arg0[%arg2 + %arg1 * 16 - 1] : memref<2048xi32>
        %1 = affine.load %arg0[%arg2 + %arg1 * 16] : memref<2048xi32>
        %2 = arith.addi %1, %0 : i32
        affine.store %2, %arg0[%arg2 + %arg1 * 16] : memref<2048xi32>
      }
    }
    return
  }
  func.func @sum_scan(%arg0: memref<128xi32>, %arg1: memref<2048xi32>)  {
    %c0_i32 = arith.constant 0 : i32
    affine.store %c0_i32, %arg0[0] : memref<128xi32>
    affine.for %arg2 = 1 to 128 {
      %0 = affine.load %arg0[%arg2 - 1] : memref<128xi32>
      %1 = affine.load %arg1[%arg2 * 16 - 1] : memref<2048xi32>
      %2 = arith.addi %0, %1 : i32
      affine.store %2, %arg0[%arg2] : memref<128xi32>
    }
    return
  }
  func.func @last_step_scan(%arg0: memref<2048xi32>, %arg1: memref<128xi32>)  {
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 16 {
        %0 = affine.load %arg0[%arg3 + %arg2 * 16] : memref<2048xi32>
        %1 = affine.load %arg1[%arg2] : memref<128xi32>
        %2 = arith.addi %0, %1 : i32
        affine.store %2, %arg0[%arg3 + %arg2 * 16] : memref<2048xi32>
      }
    }
    return
  }
  func.func @init(%arg0: memref<2048xi32>)  {
    %c0_i32 = arith.constant {polygeist.ssa_names = ["i"]} 0 : i32
    affine.for %arg1 = 0 to 2048 {
      affine.store %c0_i32, %arg0[%arg1] : memref<2048xi32>
    }
    return
  }
  func.func @hist(%arg0: memref<2048xi32>, %arg1: memref<2048xi32>, %arg2: i32)  {
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c512_i32 = arith.constant 512 : i32
    affine.for %arg3 = 0 to 512 {
      %0 = arith.index_cast %arg3 : index to i32
      affine.for %arg4 = 0 to 4 {
        %1 = affine.load %arg1[%arg4 + %arg3 * 4] : memref<2048xi32>
        %2 = arith.shrsi %1, %arg2 : i32
        %3 = arith.andi %2, %c3_i32 : i32
        %4 = arith.muli %3, %c512_i32 : i32
        %5 = arith.addi %4, %0 : i32
        %6 = arith.addi %5, %c1_i32 {polygeist.ssa_names = ["bucket_indx"]} : i32
        %7 = arith.index_cast %6 : i32 to index
        %8 = memref.load %arg0[%7] : memref<2048xi32>
        %9 = arith.addi %8, %c1_i32 : i32
        memref.store %9, %arg0[%7] : memref<2048xi32>
      }
    }
    return
  }
  func.func @update(%arg0: memref<2048xi32>, %arg1: memref<2048xi32>, %arg2: memref<2048xi32>, %arg3: i32)  {
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c512_i32 = arith.constant 512 : i32
    affine.for %arg4 = 0 to 512 {
      %0 = arith.index_cast %arg4 : index to i32
      affine.for %arg5 = 0 to 4 {
        %1 = affine.load %arg2[%arg5 + %arg4 * 4] : memref<2048xi32>
        %2 = arith.shrsi %1, %arg3 : i32
        %3 = arith.andi %2, %c3_i32 : i32
        %4 = arith.muli %3, %c512_i32 : i32
        %5 = arith.addi %4, %0 {polygeist.ssa_names = ["bucket_indx"]} : i32
        %6 = arith.index_cast %5 : i32 to index
        %7 = memref.load %arg1[%6] : memref<2048xi32>
        %8 = arith.index_cast %7 : i32 to index
        memref.store %1, %arg0[%8] : memref<2048xi32>
        %9 = memref.load %arg1[%6] : memref<2048xi32>
        %10 = arith.addi %9, %c1_i32 : i32
        memref.store %10, %arg1[%6] : memref<2048xi32>
      }
    }
    return
  }
  func.func @ss_sort(%arg0: memref<2048xi32>, %arg1: memref<2048xi32>, %arg2: memref<2048xi32>, %arg3: memref<128xi32>)  {
    %c512_i32 = arith.constant 512 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = affine.for %arg4 = 0 to 32 step 2 iter_args(%arg5 = %c0_i32) -> (i32) {
      %1 = arith.index_cast %arg4 : index to i32
      affine.for %arg6 = 0 to 2048 {
        affine.store %c0_i32, %arg2[%arg6] : memref<2048xi32>
      }
      %2 = arith.cmpi eq, %arg5, %c0_i32 : i32
      scf.if %2 {
        affine.for %arg6 = 0 to 512 {
          %4 = arith.index_cast %arg6 : index to i32
          affine.for %arg7 = 0 to 4 {
            %5 = affine.load %arg0[%arg7 + %arg6 * 4] : memref<2048xi32>
            %6 = arith.shrsi %5, %1 : i32
            %7 = arith.andi %6, %c3_i32 : i32
            %8 = arith.muli %7, %c512_i32 : i32
            %9 = arith.addi %8, %4 : i32
            %10 = arith.addi %9, %c1_i32 {polygeist.ssa_names = ["bucket_indx"]} : i32
            %11 = arith.index_cast %10 : i32 to index
            %12 = memref.load %arg2[%11] : memref<2048xi32>
            %13 = arith.addi %12, %c1_i32 : i32
            memref.store %13, %arg2[%11] : memref<2048xi32>
          }
        }
      } else {
        affine.for %arg6 = 0 to 512 {
          %4 = arith.index_cast %arg6 : index to i32
          affine.for %arg7 = 0 to 4 {
            %5 = affine.load %arg1[%arg7 + %arg6 * 4] : memref<2048xi32>
            %6 = arith.shrsi %5, %1 : i32
            %7 = arith.andi %6, %c3_i32 : i32
            %8 = arith.muli %7, %c512_i32 : i32
            %9 = arith.addi %8, %4 : i32
            %10 = arith.addi %9, %c1_i32 {polygeist.ssa_names = ["bucket_indx"]} : i32
            %11 = arith.index_cast %10 : i32 to index
            %12 = memref.load %arg2[%11] : memref<2048xi32>
            %13 = arith.addi %12, %c1_i32 : i32
            memref.store %13, %arg2[%11] : memref<2048xi32>
          }
        }
      }
      affine.for %arg6 = 0 to 128 {
        affine.for %arg7 = 1 to 16 {
          %4 = affine.load %arg2[%arg7 + %arg6 * 16 - 1] : memref<2048xi32>
          %5 = affine.load %arg2[%arg7 + %arg6 * 16] : memref<2048xi32>
          %6 = arith.addi %5, %4 : i32
          affine.store %6, %arg2[%arg7 + %arg6 * 16] : memref<2048xi32>
        }
      }
      affine.store %c0_i32, %arg3[0] : memref<128xi32>
      affine.for %arg6 = 1 to 128 {
        %4 = affine.load %arg3[%arg6 - 1] : memref<128xi32>
        %5 = affine.load %arg2[%arg6 * 16 - 1] : memref<2048xi32>
        %6 = arith.addi %4, %5 : i32
        affine.store %6, %arg3[%arg6] : memref<128xi32>
      }
      affine.for %arg6 = 0 to 128 {
        affine.for %arg7 = 0 to 16 {
          %4 = affine.load %arg2[%arg7 + %arg6 * 16] : memref<2048xi32>
          %5 = affine.load %arg3[%arg6] : memref<128xi32>
          %6 = arith.addi %4, %5 : i32
          affine.store %6, %arg2[%arg7 + %arg6 * 16] : memref<2048xi32>
        }
      }
      %3 = arith.extui %2 : i1 to i32
      scf.if %2 {
        affine.for %arg6 = 0 to 512 {
          %4 = arith.index_cast %arg6 : index to i32
          affine.for %arg7 = 0 to 4 {
            %5 = affine.load %arg0[%arg7 + %arg6 * 4] : memref<2048xi32>
            %6 = arith.shrsi %5, %1 : i32
            %7 = arith.andi %6, %c3_i32 : i32
            %8 = arith.muli %7, %c512_i32 : i32
            %9 = arith.addi %8, %4 {polygeist.ssa_names = ["bucket_indx"]} : i32
            %10 = arith.index_cast %9 : i32 to index
            %11 = memref.load %arg2[%10] : memref<2048xi32>
            %12 = arith.index_cast %11 : i32 to index
            memref.store %5, %arg1[%12] : memref<2048xi32>
            %13 = memref.load %arg2[%10] : memref<2048xi32>
            %14 = arith.addi %13, %c1_i32 : i32
            memref.store %14, %arg2[%10] : memref<2048xi32>
          }
        }
      } else {
        affine.for %arg6 = 0 to 512 {
          %4 = arith.index_cast %arg6 : index to i32
          affine.for %arg7 = 0 to 4 {
            %5 = affine.load %arg1[%arg7 + %arg6 * 4] : memref<2048xi32>
            %6 = arith.shrsi %5, %1 : i32
            %7 = arith.andi %6, %c3_i32 : i32
            %8 = arith.muli %7, %c512_i32 : i32
            %9 = arith.addi %8, %4 {polygeist.ssa_names = ["bucket_indx"]} : i32
            %10 = arith.index_cast %9 : i32 to index
            %11 = memref.load %arg2[%10] : memref<2048xi32>
            %12 = arith.index_cast %11 : i32 to index
            memref.store %5, %arg0[%12] : memref<2048xi32>
            %13 = memref.load %arg2[%10] : memref<2048xi32>
            %14 = arith.addi %13, %c1_i32 : i32
            memref.store %14, %arg2[%10] : memref<2048xi32>
          }
        }
      }
      affine.yield %3 : i32
    }
    return
  }
}
