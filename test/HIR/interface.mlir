hir.func @readA at %t(

  %Ai :!hir.memref<16*16*i32, r>,
  %Aw : !hir.memref<16*16*i32, packing=[1], w>){

  %0 = hir.constant 0 
  %x = hir.alloca("interface"):!hir.interface<"in" i32>
  %p = hir.alloca("interface"):!hir.interface<array[3x4xi1]>
  %y = hir.alloca("interface"):!hir.interface<group(i32, f32)>
  %z = hir.alloca("interface"):!hir.interface<("out" group(i32, f32),"in" array[3x4xf32], group("in" i1, "out" i1, "out" i8))>
  %p = hir.alloca("interface"):!hir.interface<array[3x4x(i1,f32)]>
  %pp = hir.alloca("interface"):!hir.interface<group([3x4xi1],[3x4xf32])>
  %a = hir.slice %p[%0,0] : !hir.interface<array[3x4xgroup(i1,f32)]>[const,!hir.const] -> !hir.interface<i32>
  %b = hir.slice %p[%0,0] : !hir.interface<array[3x4xgroup(i1,f32)]>[i32,!hir.const] -> !hir.interface<i32>

  hir.return
  }
