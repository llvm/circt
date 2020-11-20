hir.func @loop_fuse at %t(
  %Ai :!hir.memref<64*i32, r>,
  %Co : !hir.memref<64*i32, w>) {
    
  %0  = hir.constant 0
  %1  = hir.constant 1
  %2  = hir.constant 2
  %3  = hir.constant 3
  %4  = hir.constant 4
  %5  = hir.constant 5
  %8  = hir.constant 8
  %16 = hir.constant 16
  %64 = hir.constant 64
  %65 = hir.constant 65
  %66 = hir.constant 66
  %67 = hir.constant 66

  %W1r, %W1w = hir.alloc() : !hir.memref<2*i32, packing=[], r>, 
                          !hir.memref<2*i32, packing=[], w>
  %Br, %Bw = hir.alloc() : !hir.memref<64*i32, r>, 
                          !hir.memref<64*i32, w>

  %valA = hir.mem_read %Ai[%0] at %t 
      : !hir.memref<64*i32, r>[!hir.const] -> i32
  %valA1 = hir.delay %valA by %1 at %t offset %1: i32 -> i32

  %valB = hir.mem_read %Ai[%1] at %t offset %1
      : !hir.memref<64*i32, r>[!hir.const] -> i32

  hir.mem_write %valA1 to %W1w[%0] at %t offset %2 
      : (i32,!hir.memref<2*i32, packing=[],w>[!hir.const])
  hir.mem_write %valB to %W1w[%1] at %t offset %2 
      : (i32,!hir.memref<2*i32, packing=[],w>[!hir.const])

  hir.for %i : i32 = %1 : !hir.const to %64 : !hir.const 
    step %1:!hir.const iter_time(%ti = %t offset %3 ){
      hir.yield at %ti offset %1

      %v0 = hir.mem_read %W1r[%0] at %ti offset %1
          : !hir.memref<2*i32,packing=[], r>[!hir.const] -> i32
      %v1 = hir.mem_read %W1r[%1] at %ti offset %1
          : !hir.memref<2*i32,packing=[], r>[!hir.const] -> i32
      %iPlus1 = hir.add (%i,%1) : (i32,!hir.const) -> (i32)
      %v =  hir.mem_read %Ai[%iPlus1] at %ti 
          : !hir.memref<64*i32, r>[i32] -> i32

      hir.mem_write %v1 to %W1w[%0] at %ti offset %1
          : (i32,!hir.memref<2*i32,packing=[], w>[!hir.const])
      hir.mem_write %v to %W1w[%1] at %ti offset %1
          : (i32,!hir.memref<2*i32,packing=[], w>[!hir.const])

      %r  = hir.call @weighted_sum(%v0, %v1) at %ti offset %1
          : (i32, i32) -> (i32 delay 1)
      %i2 = hir.delay %i by %2 at %ti: i32 -> i32 
      hir.mem_write %r to %Bw[%i2] at %ti offset %2
          : (i32,!hir.memref<64*i32, w>[i32])
  }
  
  %W2r, %W2w = hir.alloc() : !hir.memref<2*i32, packing=[], r>, 
                          !hir.memref<2*i32, packing=[], w>

  %t2 = hir.delay %t by %64 at %t : !hir.time -> !hir.time
  %valC = hir.mem_read %Br[%1] at %t2
      : !hir.memref<64*i32, r>[!hir.const] -> i32

  %valC1 = hir.delay %valC by %1 at %t2 offset %1: i32 -> i32
  %valD = hir.mem_read %Br[%2] at %t2 offset %1
      : !hir.memref<64*i32, r>[!hir.const] -> i32

  hir.mem_write %valC1 to %W2w[%0] at %t2 offset %2
      : (i32,!hir.memref<2*i32, packing=[],w>[!hir.const])
  hir.mem_write %valD to %W2w[%1] at %t2 offset %2
      : (i32,!hir.memref<2*i32, packing=[],w>[!hir.const])

  hir.for %i : i32 = %2 : !hir.const to %64 : !hir.const 
    step %1:!hir.const iter_time(%ti = %t2 offset %3 ){
      hir.yield at %ti offset %1

      %v0 = hir.mem_read %W2r[%0] at %ti offset %1
          : !hir.memref<2*i32,packing=[], r>[!hir.const] -> i32
      %v1 = hir.mem_read %W2r[%1] at %ti offset %1
          : !hir.memref<2*i32,packing=[], r>[!hir.const] -> i32
      %iPlus1 = hir.add (%i,%1) : (i32,!hir.const) -> (i32)
      %v =  hir.mem_read %Br[%iPlus1] at %ti 
          : !hir.memref<64*i32, r>[i32] -> i32

      hir.mem_write %v1 to %W2w[%0] at %ti offset %1
          : (i32,!hir.memref<2*i32,packing=[], w>[!hir.const])
      hir.mem_write %v to %W2w[%1] at %ti offset %1 
          : (i32,!hir.memref<2*i32,packing=[], w>[!hir.const])

      %r  = hir.call @max(%v0, %v1) at %ti offset %1
          : (i32, i32) -> (i32 delay 1)

      %i2 = hir.delay %i by %2 at %ti: i32 -> i32 
      hir.mem_write %r to %Co[%i2] at %ti offset %2
          : (i32,!hir.memref<64*i32, w>[i32])
  } 

  hir.return
}
