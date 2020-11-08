hir.def @Add at %t (%A:!hir.memref<128*i32, r>, 
%B : !hir.memref<128*i32, r>, 
%C:!hir.memref<128*i32, w>){

  %0 = hir.constant 0 :!hir.const<i32>
  %1 = hir.constant 1 :!hir.const<i32>
  %3 = hir.constant 3 :!hir.const<i32>
  %128 = hir.constant 128 :!hir.const<i32>
  %Tr,%Tw = hir.alloc() : !hir.memref<1*i32,  r>, !hir.memref<1*i32, w>
  hir.mem_write %0 to %Tw[%0] at %t:!hir.const<i32> : (i32, !hir.memref<1*i32, w>[i32])
  hir.for %i:i32 = %0:!hir.const<i32> to %128:!hir.const<i32> step %1:!hir.const<i32> iter_time(%ti = %t tstep %1:!hir.const<i32>){
    %a = hir.mem_read %A[%i] at %ti :!hir.memref<128*i32,packing=[0],r>[i32] -> i32
    %b = hir.mem_read %B[%i] at %ti : !hir.memref<128*i32, r>[i32] -> i32
    //%mult = hir.call @mult_comb(%a, %b) at %ti offset %1: !hir.const<i32> {input_delay=[0,0],output_delay=[0]}: (i32, i32) -> (i32)
    %mult = hir.call @mult_pipelined(%a, %b) at %ti offset %1: !hir.const<i32> {input_delay=[0,0],output_delay=[2]}: (i32, i32) -> (i32)
    %temp = hir.mem_read %Tr[%0] at %ti offset %3: !hir.memref<1*i32, r>[i32] -> i32
    %res = hir.add(%mult,%temp)
    hir.mem_write %res to %Tw[%0] at %ti offset %3:!hir.const<i32> : (i32, !hir.memref<1*i32, w>[i32])
  }
  %temp = hir.mem_read %Tr[%0] at %ti offset %3: !hir.memref<1*i32, r>[i32] -> i32
  hir.mem_write %c to %C[%i] at %ti offset %1:!hir.const<i32> : (i32, !hir.memref<128*i32, w>[i32])
  hir.return
}
