// RUN: circt-opt %s
#bram_r = {"rd_latency" = 1}
#bram_w = {"wr_latency" = 1}
hir.func @Array_Add at %t (%A:!hir.memref<128xi32> ports [#bram_r], 
%B : !hir.memref<128xi32> ports [#bram_r], 
%C:!hir.memref<128xi32> ports [#bram_w]){

  %c0_i8 = hw.constant 0: i8
  %c1_i8 = hw.constant 1:i8 
  %c128_i8 = hw.constant 128:i8 
  hir.for %i:i8 = %c0_i8 to %c128_i8  step %c1_i8 iter_time(%ti = %t + 1){
    %i_i7 = comb.extract %i from 0: (i8)->(i7)
    %a = hir.load %A[port 0][%i_i7] at %ti :!hir.memref<128xi32> delay 1
    %b = hir.load %B[port 0][%i_i7] at %ti : !hir.memref<128xi32> delay 1
    %c = comb.add  %a, %b  : i32
    %i_delayed_i7 = hir.delay %i_i7 by 1 at %ti : i7 
    hir.store %c to %C[port 0][%i_delayed_i7] at %ti + 1 : !hir.memref<128xi32>
    hir.next_iter at %ti + 1
  }
  hir.return
}
