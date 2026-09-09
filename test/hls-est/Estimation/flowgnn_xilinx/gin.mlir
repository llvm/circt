module {
  memref.global @messages_pong : memref<4x125x100xf32>
  memref.global @messages_ping : memref<4x125x100xf32>
  func.func @GIN_compute_graphs(%arg0: i32, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: memref<?xi32>, %arg4: memref<?x1xf32>, %arg5: memref<?x!llvm.struct<(array<9 x i32>)>>, %arg6: memref<?x2xi32>, %arg7: memref<?x!llvm.struct<(array<3 x i32>)>>, %arg8: memref<?x173x100xf32>, %arg9: memref<?x5x13x100xf32>, %arg10: memref<?x5x200x100xf32>, %arg11: memref<?x5x200xf32>, %arg12: memref<?x5x100x200xf32>, %arg13: memref<?x5x100xf32>, %arg14: memref<?x1x100xf32>, %arg15: memref<?x1xf32>)  {
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = memref.get_global @messages_ping : memref<4x125x100xf32>
    %2 = memref.get_global @messages_pong : memref<4x125x100xf32>
    %3:3 = affine.for %arg16 = 0 to %0 iter_args(%arg17 = %c0_i32, %arg18 = %c0_i32, %arg19 = %c-1_i32) -> (i32, i32, i32) {
      %4 = affine.load %arg1[%arg16] : memref<?xi32>
      %5 = affine.load %arg2[%arg16] : memref<?xi32>
      %6 = affine.load %arg3[%arg16] : memref<?xi32>
      %7 = arith.cmpi ne, %6, %c0_i32 : i32
      %8 = scf.if %7 -> (i32) {
        %19 = arith.addi %arg19, %c1_i32 : i32
        %20 = arith.index_cast %19 : i32 to index
        %21 = "polygeist.subindex"(%arg10, %20) : (memref<?x5x200x100xf32>, index) -> memref<5x200x100xf32>
        %22 = "polygeist.subindex"(%arg11, %20) : (memref<?x5x200xf32>, index) -> memref<5x200xf32>
        %23 = "polygeist.subindex"(%arg12, %20) : (memref<?x5x100x200xf32>, index) -> memref<5x100x200xf32>
        %24 = "polygeist.subindex"(%arg13, %20) : (memref<?x5x100xf32>, index) -> memref<5x100xf32>
        %25 = "polygeist.subindex"(%arg9, %20) : (memref<?x5x13x100xf32>, index) -> memref<5x13x100xf32>
        %26 = "polygeist.subindex"(%arg14, %20) : (memref<?x1x100xf32>, index) -> memref<1x100xf32>
        %27 = "polygeist.subindex"(%arg15, %20) : (memref<?x1xf32>, index) -> memref<1xf32>
        func.call @_Z12load_weightsPA200_A100_fPA200_fPA100_S2_PS_PA13_S_S6_Pf(%21, %22, %23, %24, %25, %26, %27) : (memref<5x200x100xf32>, memref<5x200xf32>, memref<5x100x200xf32>, memref<5x100xf32>, memref<5x13x100xf32>, memref<1x100xf32>, memref<1xf32>) -> ()
        scf.yield %19 : i32
      } else {
        scf.yield %arg19 : i32
      }
      %9 = arith.index_cast %arg17 : i32 to index
      %10 = "polygeist.subindex"(%arg6, %9) : (memref<?x2xi32>, index) -> memref<?x2xi32>
      %11 = "polygeist.subindex"(%arg7, %9) : (memref<?x!llvm.struct<(array<3 x i32>)>>, index) -> memref<?x!llvm.struct<(array<3 x i32>)>>
      func.call @_Z10load_graphP6edge_tPSt5arrayIiLm3EEii(%10, %11, %4, %5) : (memref<?x2xi32>, memref<?x!llvm.struct<(array<3 x i32>)>>, i32, i32) -> ()
      %12 = arith.index_cast %arg18 : i32 to index
      %13 = "polygeist.subindex"(%arg5, %12) : (memref<?x!llvm.struct<(array<9 x i32>)>>, index) -> memref<?x!llvm.struct<(array<9 x i32>)>>
      %14 = arith.index_cast %8 : i32 to index
      %15 = "polygeist.subindex"(%arg8, %14) : (memref<?x173x100xf32>, index) -> memref<173x100xf32>
      %16 = "polygeist.subindex"(%arg4, %arg16) : (memref<?x1xf32>, index) -> memref<?xf32>
      affine.for %arg20 = 0 to 6 {
        %19 = arith.index_cast %arg20 : index to i32
        %20 = arith.remsi %arg20, %c2 : index
        %21 = arith.cmpi slt, %20, %c0 : index
        %22 = arith.addi %20, %c2 : index
        %23 = arith.select %21, %22, %20 : index
        %24 = arith.cmpi eq, %23, %c0 : index
        scf.if %24 {
          func.call @_Z18compute_CONV_layeriPA125_A100_fS1_PSt5arrayIiLm9EEPS_Pfi(%19, %1, %2, %13, %15, %16, %4) : (i32, memref<4x125x100xf32>, memref<4x125x100xf32>, memref<?x!llvm.struct<(array<9 x i32>)>>, memref<173x100xf32>, memref<?xf32>, i32) -> ()
        } else {
          func.call @_Z18compute_CONV_layeriPA125_A100_fS1_PSt5arrayIiLm9EEPS_Pfi(%19, %2, %1, %13, %15, %16, %4) : (i32, memref<4x125x100xf32>, memref<4x125x100xf32>, memref<?x!llvm.struct<(array<9 x i32>)>>, memref<173x100xf32>, memref<?xf32>, i32) -> ()
        }
      }
      %17 = arith.addi %arg18, %4 : i32
      %18 = arith.addi %arg17, %5 : i32
      affine.yield %18, %17, %8 : i32, i32, i32
    }
    return
  }
  func.func private @_Z12load_weightsPA200_A100_fPA200_fPA100_S2_PS_PA13_S_S6_Pf(memref<5x200x100xf32>, memref<5x200xf32>, memref<5x100x200xf32>, memref<5x100xf32>, memref<5x13x100xf32>, memref<1x100xf32>, memref<1xf32>) 
  func.func private @_Z10load_graphP6edge_tPSt5arrayIiLm3EEii(memref<?x2xi32>, memref<?x!llvm.struct<(array<3 x i32>)>>, i32, i32) 
  func.func private @_Z18compute_CONV_layeriPA125_A100_fS1_PSt5arrayIiLm9EEPS_Pfi(i32, memref<4x125x100xf32>, memref<4x125x100xf32>, memref<?x!llvm.struct<(array<9 x i32>)>>, memref<173x100xf32>, memref<?xf32>, i32) 
}
