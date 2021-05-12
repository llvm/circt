// RUN: circt-opt %s
hir.func @convolution at %t(%x : i32){
  %1 = hir.constant(1) : !hir.const
  %t2 = hir.unroll_for %i = 0 to 3 step 1 iter_time(%ti = %t+%1){
    hir.yield at %ti 
    %64 = hir.constant (64) : !hir.const
    %128 = hir.constant (128) : !hir.const
    %2 = hir.add (%i,%128) :(!hir.const, !hir.const) -> (!hir.const)
  }
  %t3 = hir.delay %t2 by %1 at %t2 : !hir.time -> !hir.time
  hir.return
}
