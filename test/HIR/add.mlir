hir.def @Array_Add at %t (%A:!hir.memref<128*i32, r>, 
%B : !hir.memref<128*i32, r>, 
%C:!hir.memref<128*i32, w>){

  %0 = hir.constant 0 :!hir.const
  %1 = hir.constant 1 :!hir.const
  %128 = hir.constant 128 :!hir.const
  hir.for %i:i8 = %0:!hir.const to %128:!hir.const  step %1:!hir.const iter_time(%ti = %t offset %1){
    hir.yield at %ti offset %1
    %a = hir.mem_read %A[%i] at %ti :!hir.memref<128*i32,r>[i8] -> i32
    %b = hir.mem_read %B[%i] at %ti : !hir.memref<128*i32, r>[i8] -> i32
    %c = hir.add (%a, %b) : (i32, i32) -> (i32)
    %i1 = hir.delay %i by %1 : !hir.const at %ti : i8 -> i8 
    hir.mem_write %c to %C[%i1] at %ti offset %1 : (i32, !hir.memref<128*i32, w>[i8])
  }
  hir.return
}
