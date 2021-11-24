// RUN: circt-opt %s
#bram_r = {"rd_latency" = 1}
#bram_w = {"wr_latency" = 1}
#reg_r = {"rd_latency" = 0}
#reg_w = {"wr_latency" = 1}


hir.func.extern @weighted_sum at %t(%v0:i32,%wt0:i32, %v1:i32,%wt1:i32)->(%result:i32 delay 1)

hir.func @stencil_1d at %t(
  %Ai :!hir.memref<64xi32> ports [#bram_r],
  %Bw : !hir.memref<64xi32> ports [#bram_w],
  %w0 :i32,
  %w1 :i32) {
  %0  = arith.constant 0:index
  %1  = arith.constant 1:index
  %c0_i6  = hw.constant 0:i6
  %c1_i6  = hw.constant 1:i6
  %c1_i7  = hw.constant 1:i7
  %c64_i7 = hw.constant 64:i7

  %SR = hir.alloca "reg"  : !hir.memref<(bank 2)xi32> ports [#reg_r,#reg_w]


  %valA = hir.load %Ai[port 0][%c0_i6] at %t 
      : !hir.memref<64xi32>
  %valA1 = hir.delay %valA by 1 at %t + 1: i32

  %valB = hir.load %Ai[port 0][%c1_i6] at %t + 1
      : !hir.memref<64xi32>

  hir.store %valA1 to %SR[port 1][%0] at %t + 2 
      : !hir.memref<(bank 2)xi32>
  hir.store %valB to %SR[port 1][%1] at %t + 2 
      : !hir.memref<(bank 2)xi32>

  %W = hir.alloca "reg"  : !hir.memref<(bank 2)xi32> ports [#reg_r,#reg_w]
  hir.store %w0 to %W[port 1][%0] at %t 
      : !hir.memref<(bank 2)xi32>
  hir.store %w1 to %W[port 1][%1] at %t 
      : !hir.memref<(bank 2)xi32>

  hir.for %i : i7 = %c1_i7  to %c64_i7 step %c1_i7 iter_time(%ti = %t + 3 ){
      %v0 = hir.load %SR[port 0][%0] at %ti + 1
          : !hir.memref<(bank 2)xi32>
      %v1 = hir.load %SR[port 0][%1] at %ti + 1
          : !hir.memref<(bank 2)xi32>
      %i_i6 = comb.extract %i from 0: (i7)->(i6)
      %iPlus1 = comb.add  %i_i6,%c1_i6  : i6
      %v =  hir.load %Ai[port 0][%iPlus1] at %ti 
          : !hir.memref<64xi32>

      hir.store %v1 to %SR[port 1][%0] at %ti + 1
          : !hir.memref<(bank 2)xi32>
      hir.store %v to %SR[port 1][%1] at %ti + 1
          : !hir.memref<(bank 2)xi32>

      %wt0 =  hir.load %W[port 0][%0] at %ti + 1
          : !hir.memref<(bank 2)xi32>
      %wt1 =  hir.load %W[port 0][%1] at %ti + 1
          : !hir.memref<(bank 2)xi32>

      %r  = hir.call @weighted_sum(%v0,%wt0, %v1,%wt1) at %ti + 1
          : !hir.func<(i32, i32,i32,i32) -> (i32 delay 1)>

      %i2 = hir.delay %i_i6 by 2 at %ti: i6
      hir.store %r to %Bw[port 0][%i2] at %ti + 2
          : !hir.memref<64xi32>
      hir.next_iter at %ti + 1
  }
  hir.return
}
