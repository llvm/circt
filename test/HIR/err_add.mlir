hir.def @Array_Add at %t (%A:!hir.memref<128*i32, r>, 
%B : !hir.memref<128*i32, r>, 
%C:!hir.memref<128*i32, w>){

  %0 = hir.constant 0 :!hir.const<i7>
  %1 = hir.constant 1 :!hir.const<i7>
  %128 = hir.constant 128 :!hir.const<i7>
  hir.for %i:i7 = %0:!hir.const<i7> to %128:!hir.const<i7> step %1:!hir.const<i7> iter_time(%ti = %t tstep %1:!hir.const<i7>){
    %a = hir.mem_read %A[%i] at %ti :!hir.memref<128*i32,packing=[0],r>[i7] -> i32
    %b = hir.mem_read %B[%i] at %ti : !hir.memref<128*i32, r>[i7] -> i32
    %c = hir.add (%a, %b) : (i32, i32) -> (i32)
    hir.mem_write %c to %C[%i] at %ti offset %1:!hir.const<i7> : (i32, !hir.memref<128*i32, w>[i7])
  }
  hir.return
}
