hir.func @readA at %t(
  %Ai :!hir.memref<16*16*i32, r>,
  %Aw : !hir.memref<16*16*i32, packing=[1], w>){

  %x = hir.alloc():!hir.interface<i32>

  hir.return
  }
