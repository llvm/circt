hir.def @Add at %t (%A:!hir.memref<128*i32,r>, 
%B : !hir.memref<128*i32,r>, 
%C:!hir.memref<128*i32,w>){

  %0 = hir.constant 0 :i32
  %1 = hir.constant 1 :i32
  %c1 = hir.constant 1 :!hir.const<i32>
  %128 = hir.constant 128 :i32
  hir.for %i:i32 = %0:i32 to %128:i32 step %1:i32 iter_time(%ti = %t tstep %c1:!hir.const<i32>){
    %a = hir.mem_read %A[%i] at %ti : !hir.memref<128*i32,r>[i32] -> i32
    %b = hir.mem_read %B[%i] at %ti : !hir.memref<128*i32,r>[i32] -> i32
    %c = hir.add at %ti offset %c1:!hir.const<i32> (%a,%b) : (i32,i32) -> i32
    hir.mem_write %c to %C[%i] at %ti offset %c1:!hir.const<i32> :
    (i32,!hir.memref<128*i32,w>[i32])
  }
  hir.return
}
