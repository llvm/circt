// RUN: circt-opt %s
#bram_r = {"rd_latency" = 1}
#bram_w = {"wr_latency" = 1}

hir.func @histogram at %t(
  %A :!hir.memref<16x16xi8> ports [#bram_r],
  %B : !hir.memref<256xi32> ports [#bram_w]){

  %c0_i32   = hw.constant 0:i32
  %c1_i32   = hw.constant 1:i32
  %1   = hw.constant 1
  %2   = hw.constant 2
  %4   = hw.constant 4
  %16  = hw.constant 16
  %64  = hw.constant 64
  %c0_i9   = hw.constant 0:i9
  %c1_i9   = hw.constant 1:i9
  %c256_i9 = hw.constant 256:i9
  %c0_i5   = hw.constant 0:i5
  %c1_i5   = hw.constant 1:i5
  %c16_i5   = hw.constant 16:i5

  %buff = hir.alloca "bram" : !hir.memref<256xi32> ports [#bram_r,#bram_w]
  
  %t2 = hir.for %i : i9 = %c0_i9  to %c256_i9  step %c1_i9 iter_time(%ti = %t + 1 ){
      %i_i8 = comb.extract %i from 0 :(i9)->(i8)
      hir.store %c0_i32 to %buff[port 1][%i_i8] at %ti
        : !hir.memref<256xi32>
      hir.next_iter at %ti + 1
  }

  %t3 = hir.for %i : i5 = %c0_i5  to %c16_i5 step %c1_i5 iter_time(%ti = %t2 + 1 ){
      %i_i4 = comb.extract %i from 0 : (i5)->(i4)
      %t_next=hir.for %j : i5 = %c0_i5  to %c16_i5 step %c1_i5 iter_time(%tj = %ti + 1 ){
          %j_i4 = comb.extract %j from 0 : (i5)->(i4)
          %v = hir.load %A[port 0][%i_i4,%j_i4] at %tj
              : !hir.memref<16x16xi8>
          %count = hir.load %buff[port 0][%v] at %tj + 1
              : !hir.memref<256xi32> 
          %new_count = comb.add %count,%c1_i32  :i32
          hir.store %new_count to %buff[port 1][%v] at %tj + 2
              : !hir.memref<256xi32>
          hir.next_iter at %tj + 2
      }
      hir.next_iter at %t_next + 1
  }

  hir.for %i : i9 = %c0_i9  to %c256_i9 step %c1_i9 iter_time(%ti = %t3 + 4 ){
      %i_i8 = comb.extract %i from 0 :(i9)->(i8)
      %count = hir.load %buff[port 0][%i_i8] at %ti
            : !hir.memref<256xi32> 
      %i1 = hir.delay %i_i8 by 1 at %ti: i8
      hir.store %count to %B[port 0][%i1] at %ti + 1
        : !hir.memref<256xi32>
      hir.next_iter at %ti + 1
  }
  hir.return
}
