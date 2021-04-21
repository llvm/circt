// RUN: circt-opt %s
hir.func @histogram at %t(
  %A :!hir.memref<64*64*i8, r>,
  %B : !hir.memref<256*i32, w>){

  %0  = hir.constant 0
  %1  = hir.constant 1
  %2  = hir.constant 2
  %4  = hir.constant 4
  %16 = hir.constant 16
  %64 = hir.constant 64
  %256 = hir.constant 256

  %buffr, %buffw = hir.alloc() : !hir.memref<256*i32, r>, !hir.memref<256*i32, w>
  
  %t2 = hir.for %i : i32 = %0 : !hir.const to %256 : !hir.const 
    step %1:!hir.const iter_time(%ti = %t offset %1 ){
      hir.yield at %ti offset %1
      hir.mem_write %0 to %buffw[%i] at %ti
        : (!hir.const,!hir.memref<256*i32,w>[i32])
  }

  %t3=hir.for %i : i32 = %0 : !hir.const to %16: !hir.const 
    step %1:!hir.const iter_time(%ti = %t2 offset %1 ){
      %t_next=hir.for %j : i32 = %0 : !hir.const to %16: !hir.const 
        step %1:!hir.const iter_time(%tj = %ti offset %1 ){
          %v = hir.mem_read %A[%i,%j] at %tj
              : !hir.memref<64*64*i8,r> [i32,i32] -> i8
          %count = hir.mem_read %buffr[%v] at %tj offset %1
              : !hir.memref<256*i32,r> [i8] -> i32
          %new_count = hir.add(%count,%1) :(i32,!hir.const)->(i32)
          hir.mem_write %new_count to %buffw[%v] at %tj offset %2
              : (i32,!hir.memref<256*i32, w>[i8])
          hir.yield at %tj offset %2
      }
      hir.yield at %t_next offset %1
  }

  hir.for %i : i32 = %0 : !hir.const to %256 : !hir.const 
    step %1:!hir.const iter_time(%ti = %t3 offset %4 ){
      hir.yield at %ti offset %1
      %count = hir.mem_read %buffr[%i] at %ti
            : !hir.memref<256*i32,r> [i32] -> i32
      %i1 = hir.delay %i by %1 at %ti: i32 -> i32
      hir.mem_write %count to %B[%i1] at %ti offset %1
        : (i32,!hir.memref<256*i32,w>[i32])
  }
  hir.return
}
