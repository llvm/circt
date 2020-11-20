hir.func @stencil_1d at %t(
  %Ai :!hir.memref<64*i32, r>,
  %Bw : !hir.memref<64*i32, w>,
  %w0 :i32,
  %w1 :i32) {
  %0  = hir.constant 0
  %1  = hir.constant 1
  %2  = hir.constant 2
  %3  = hir.constant 3
  %4  = hir.constant 4
  %5  = hir.constant 5
  %64 = hir.constant 64

  %SRr, %SRw = hir.alloc() : !hir.memref<2*i32, packing=[], r>, 
                          !hir.memref<2*i32, packing=[], w>


  %valA = hir.mem_read %Ai[%0] at %t 
      : !hir.memref<64*i32, r>[!hir.const] -> i32
  %valA1 = hir.delay %valA by %1 at %t offset %1: i32 -> i32

  %valB = hir.mem_read %Ai[%1] at %t offset %1
      : !hir.memref<64*i32, r>[!hir.const] -> i32

  hir.mem_write %valA1 to %SRw[%0] at %t offset %2 
      : (i32,!hir.memref<2*i32, packing=[],w>[!hir.const])
  hir.mem_write %valB to %SRw[%1] at %t offset %2 
      : (i32,!hir.memref<2*i32, packing=[],w>[!hir.const])

  %Wr, %Ww = hir.alloc() : !hir.memref<2*i32, packing=[], r>, 
                          !hir.memref<2*i32, packing=[], w>
  hir.mem_write %w0 to %Ww[%0] at %t 
      : (i32,!hir.memref<2*i32, packing=[],w>[!hir.const])
  hir.mem_write %w1 to %Ww[%1] at %t 
      : (i32,!hir.memref<2*i32, packing=[],w>[!hir.const])

  hir.for %i : i32 = %1 : !hir.const to %64 : !hir.const 
    step %1:!hir.const iter_time(%ti = %t offset %3 ){
      hir.yield at %ti offset %1

      %v0 = hir.mem_read %SRr[%0] at %ti offset %1
          : !hir.memref<2*i32,packing=[], r>[!hir.const] -> i32
      %v1 = hir.mem_read %SRr[%1] at %ti offset %1
          : !hir.memref<2*i32,packing=[], r>[!hir.const] -> i32
      %iPlus1 = hir.add (%i,%1) : (i32,!hir.const) -> (i32)
      %v =  hir.mem_read %Ai[%iPlus1] at %ti 
          : !hir.memref<64*i32, r>[i32] -> i32

      hir.mem_write %v1 to %SRw[%0] at %ti offset %1
          : (i32,!hir.memref<2*i32,packing=[], w>[!hir.const])
      hir.mem_write %v to %SRw[%1] at %ti offset %1
          : (i32,!hir.memref<2*i32,packing=[], w>[!hir.const])

      %wt0 =  hir.mem_read %Wr[%0] at %ti offset %1
          : !hir.memref<2*i32, packing=[], r>[!hir.const] -> i32
      %wt1 =  hir.mem_read %Wr[%1] at %ti offset %1
          : !hir.memref<2*i32, packing=[], r>[!hir.const] -> i32

      %r  = hir.call @weighted_sum(%v0,%wt0, %v1,%wt1) at %ti offset %1
          : (i32, i32,i32,i32) -> (i32 delay 1)

      %i2 = hir.delay %i by %2 at %ti: i32 -> i32 
      hir.mem_write %r to %Bw[%i2] at %ti offset %2
          : (i32,!hir.memref<64*i32, w>[i32])
  }
  hir.return
}
