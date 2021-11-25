// RUN: circt-opt %s
#bram_r = {"rd_latency"=1}
#bram_w = {"wr_latency"=1}

hir.func @transpose_hir at %t(
  %Ai :!hir.memref<16x16xi32> ports [#bram_r],
  %Co : !hir.memref<16x16xi32> ports [#bram_w]) {
    
    %c0_i5 = hw.constant 0:i5 
    %c1_i5 = hw.constant 1:i5 
    %c16_i5 = hw.constant 16:i5 

    hir.for %i : i5 = %c0_i5  to %c16_i5 step %c1_i5 iter_time(%ti = %t + 1 ){
      %tf =hir.for %j : i5 = %c0_i5  to %c16_i5 step %c1_i5 iter_time(%tj = %ti + 1){
          %i_i4 = comb.extract %i from 0:(i5)->(i4)
          %j_i4 = comb.extract %j from 0:(i5)->(i4)
          %v =  hir.load %Ai[port 0][%i_i4, %j_i4] at %tj 
          : !hir.memref<16x16xi32>
          %j1 = hir.delay %j_i4 by 1 at %tj: i4
          hir.store %v to %Co[port 0][%j1, %i_i4] at %tj + 1 
          : !hir.memref<16x16xi32>
          hir.next_iter at %tj + 1
      }

      hir.next_iter at %tf + 1
    }
    hir.return
}
