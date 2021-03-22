hir.func @readA at %t(
  %Ai :!hir.memref<16*16*i32, r>,
  %Aw : !hir.memref<16*16*i32, packing=[1], w>){

  %x = hir.alloc():!hir.interface<"in" i32>
  %y = hir.alloc():!hir.interface<(i32, f32)>
  %z = hir.alloc():!hir.interface<("out" (i32, f32),"in" [3x4xf32], ("in" i1, "out" i1, "out" i8))>
  %t = hir.alloc():!hir.interface<[3x4x(i1,f32)]>
  %tt = hir.alloc():!hir.interface<([3x4xi1],[3x4xf32])>

  hir.return
  }
