#bram_r = {"rd"=1}
#bram_w = {"wr"=1}
// RUN: circt-opt %s
hir.func @Array_Add at %t (%A:!hir.memref<128xi32, #bram_r>, 
%B : !hir.memref<128xi32, #bram_r>, 
%C:!hir.memref<128xi32, #bram_w>){

  %0 = hir.constant (0) : !hir.const
  %1 = hir.constant (1) : !hir.const
  %128 = hir.constant (128) : !hir.const

  hir.for %i:i8 = %0:!hir.const to %128:!hir.const  
  step %1:!hir.const iter_time(%ti = %t  +  %1){
    hir.yield at %ti  +  %1

    %a = hir.load %A[%i] at %ti 
    :!hir.memref<128xi32,#bram_r>[i8] -> i32
    %b = hir.load %B[%i] at %ti 
    : !hir.memref<128xi32, #bram_r>[i8] -> i32

    %c = hir.add (%a, %b) : (i32, i32) -> (i32)

    hir.store %c to %C[%i] at %ti  +  %1 
    : (i32, !hir.memref<128xi32, #bram_w>[i8])
  }
  hir.return
}
