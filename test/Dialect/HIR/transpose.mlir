// RUN: circt-opt %s
hir.func @transpose at %t(
  %Ai :!hir.memref<16*16*i32, r>,
  %Co : !hir.memref<16*16*i32, w>) {
    
  %0 = hir.constant 0 
  %1 = hir.constant 1 
  %16 = hir.constant 16 

  hir.for %i : i32 = %0 : !hir.const to %16 : !hir.const 
    step %1:!hir.const iter_time(%ti = %t offset %1 ){

      %tf =hir.for %j : i32 = %0 : !hir.const to %16 : !hir.const 
        step %1:!hir.const iter_time(%tj = %ti offset %1){
          %v =  hir.mem_read %Ai[%i, %j] at %tj 
              : !hir.memref<16*16*i32, r>[i32, i32] -> i32
          %j1 = hir.delay %j by %1 at %tj: i32 -> i32
          hir.mem_write %v to %Co[%j1, %i] at %tj offset %1 
              : (i32,!hir.memref<16*16*i32, w>[i32, i32])
          hir.yield at %tj offset %1
      }

      hir.yield at %tf offset %1
  }
  hir.return
}
