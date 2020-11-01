hir.def @Add at %t (%A:!hir.memref<8*128*i32,packing=[0], r>, 
%B : !hir.memref<128*i32, r>, 
%C:!hir.memref<128*i64, w>){

  %0 = hir.constant 0 :!hir.const<i32>
  %1 = hir.constant 1 :!hir.const<i3>
  %128 = hir.constant 128 :!hir.const<i32>
  hir.for %i:i32 = %0:!hir.const<i32> to %128:!hir.const<i32> step %1:!hir.const<i3> iter_time(%ti = %t tstep %1:!hir.const<i3>){
    %a = hir.mem_read %A[%0,%i] at %ti :!hir.memref<8*128*i32,packing=[0],r>[!hir.const<i32>,i32] -> i32
    %b = hir.mem_read %B[%i] at %ti : !hir.memref<128*i32, r>[i32] -> i32
    //%b = hir.mem_read %B[%i] at %ti offset %1:!hir.const<i3> : !hir.memref<128*i32, r>[i32] -> i32
    %c = hir.add (%a, %b) : (i32, i32) -> (i64)
    hir.mem_write %c to %C[%i] at %ti offset %1:!hir.const<i3> : (i64, !hir.memref<128*i64, w>[i32])
  }
  hir.return
}
