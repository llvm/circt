module {
memref.global @messages_ping : memref<4x5000x80x!llvm.struct<(array<4 x f32>)>>
  memref.global @messages_pong : memref<4x5000x80x!llvm.struct<(array<4 x f32>)>>
  func.func @PNA_compute_graphs(%arg0: i32, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: memref<?xi32>, %arg4: memref<?x1xf32>, %arg5: memref<?x!llvm.struct<(array<9 x i32>)>>, %arg6: memref<?x2xi32>, %arg7: memref<?x173x80xf32>, %arg8: memref<?x4x80x3x4x80xf32>, %arg9: memref<?x4x80xf32>, %arg10: memref<?x40x80xf32>, %arg11: memref<?x40xf32>, %arg12: memref<?x20x40xf32>, %arg13: memref<?x20xf32>, %arg14: memref<?x1x20xf32>, %arg15: memref<?x1xf32>, %arg16: memref<?xf32>)  {
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = memref.get_global @messages_pong : memref<4x5000x80x!llvm.struct<(array<4 x f32>)>>
    %2 = memref.get_global @messages_ping : memref<4x5000x80x!llvm.struct<(array<4 x f32>)>>
    %3:3 = affine.for %arg17 = 0 to %0 iter_args(%arg18 = %c0_i32, %arg19 = %c0_i32, %arg20 = %c-1_i32) -> (i32, i32, i32) {
      %4 = affine.load %arg1[%arg17] : memref<?xi32>
      %5 = affine.load %arg2[%arg17] : memref<?xi32>
      %6 = affine.load %arg3[%arg17] : memref<?xi32>
      %7 = arith.cmpi ne, %6, %c0_i32 : i32
      %8 = scf.if %7 -> (i32) {
        %18 = arith.addi %arg20, %c1_i32 : i32
        %19 = arith.index_cast %18 : i32 to index
        %20 = "polygeist.subindex"(%arg8, %19) : (memref<?x4x80x3x4x80xf32>, index) -> memref<4x80x3x4x80xf32>
        %21 = "polygeist.subindex"(%arg9, %19) : (memref<?x4x80xf32>, index) -> memref<4x80xf32>
        %22 = "polygeist.subindex"(%arg10, %19) : (memref<?x40x80xf32>, index) -> memref<40x80xf32>
        %23 = "polygeist.subindex"(%arg11, %19) : (memref<?x40xf32>, index) -> memref<40xf32>
        %24 = "polygeist.subindex"(%arg12, %19) : (memref<?x20x40xf32>, index) -> memref<20x40xf32>
        %25 = "polygeist.subindex"(%arg13, %19) : (memref<?x20xf32>, index) -> memref<20xf32>
        %26 = "polygeist.subindex"(%arg14, %19) : (memref<?x1x20xf32>, index) -> memref<1x20xf32>
        %27 = "polygeist.subindex"(%arg15, %19) : (memref<?x1xf32>, index) -> memref<1xf32>
        %28 = memref.load %arg16[%19] : memref<?xf32>
        func.call @_Z12load_weightsPA80_A3_A4_A80_fPS_S4_PfPA40_fS5_PA20_fS5_f(%20, %21, %22, %23, %24, %25, %26, %27, %28) : (memref<4x80x3x4x80xf32>, memref<4x80xf32>, memref<40x80xf32>, memref<40xf32>, memref<20x40xf32>, memref<20xf32>, memref<1x20xf32>, memref<1xf32>, f32) -> ()
        scf.yield %18 : i32
      } else {
        scf.yield %arg20 : i32
      }
      %9 = arith.index_cast %arg18 : i32 to index
      %10 = "polygeist.subindex"(%arg6, %9) : (memref<?x2xi32>, index) -> memref<?x2xi32>
      func.call @_Z10load_graphP6edge_tii(%10, %4, %5) : (memref<?x2xi32>, i32, i32) -> ()
      func.call @_Z14reset_messagesPA5000_A80_St5arrayIfLm4EEi(%1, %4) : (memref<4x5000x80x!llvm.struct<(array<4 x f32>)>>, i32) -> ()
      %11 = arith.index_cast %arg19 : i32 to index
      %12 = "polygeist.subindex"(%arg5, %11) : (memref<?x!llvm.struct<(array<9 x i32>)>>, index) -> memref<?x!llvm.struct<(array<9 x i32>)>>
      %13 = arith.index_cast %8 : i32 to index
      %14 = "polygeist.subindex"(%arg7, %13) : (memref<?x173x80xf32>, index) -> memref<173x80xf32>
      %15 = "polygeist.subindex"(%arg4, %arg17) : (memref<?x1xf32>, index) -> memref<?xf32>
      affine.for %arg21 = 0 to 5 {
        %18 = arith.index_cast %arg21 : index to i32
        %19 = arith.remsi %arg21, %c2 : index
        %20 = arith.cmpi slt, %19, %c0 : index
        %21 = arith.addi %19, %c2 : index
        %22 = arith.select %20, %21, %19 : index
        %23 = arith.cmpi eq, %22, %c0 : index
        scf.if %23 {
          func.call @_Z18compute_CONV_layeriPA5000_A80_St5arrayIfLm4EES3_PS_IiLm9EEPA80_fPfi(%18, %2, %1, %12, %14, %15, %4) : (i32, memref<4x5000x80x!llvm.struct<(array<4 x f32>)>>, memref<4x5000x80x!llvm.struct<(array<4 x f32>)>>, memref<?x!llvm.struct<(array<9 x i32>)>>, memref<173x80xf32>, memref<?xf32>, i32) -> ()
        } else {
          func.call @_Z18compute_CONV_layeriPA5000_A80_St5arrayIfLm4EES3_PS_IiLm9EEPA80_fPfi(%18, %1, %2, %12, %14, %15, %4) : (i32, memref<4x5000x80x!llvm.struct<(array<4 x f32>)>>, memref<4x5000x80x!llvm.struct<(array<4 x f32>)>>, memref<?x!llvm.struct<(array<9 x i32>)>>, memref<173x80xf32>, memref<?xf32>, i32) -> ()
        }
      }
      %16 = arith.addi %arg19, %4 : i32
      %17 = arith.addi %arg18, %5 : i32
      affine.yield %17, %16, %8 : i32, i32, i32
    }
    return
  }
  func.func private @_Z12load_weightsPA80_A3_A4_A80_fPS_S4_PfPA40_fS5_PA20_fS5_f(memref<4x80x3x4x80xf32>, memref<4x80xf32>, memref<40x80xf32>, memref<40xf32>, memref<20x40xf32>, memref<20xf32>, memref<1x20xf32>, memref<1xf32>, f32) 
  func.func private @_Z10load_graphP6edge_tii(memref<?x2xi32>, i32, i32) 
  func.func private @_Z14reset_messagesPA5000_A80_St5arrayIfLm4EEi(memref<4x5000x80x!llvm.struct<(array<4 x f32>)>>, i32) 
  func.func private @_Z18compute_CONV_layeriPA5000_A80_St5arrayIfLm4EES3_PS_IiLm9EEPA80_fPfi(i32, memref<4x5000x80x!llvm.struct<(array<4 x f32>)>>, memref<4x5000x80x!llvm.struct<(array<4 x f32>)>>, memref<?x!llvm.struct<(array<9 x i32>)>>, memref<173x80xf32>, memref<?xf32>, i32) 
}
