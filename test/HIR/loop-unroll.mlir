// RUN: circt-opt %s
hir.func @convolution at %t(%x : i32){
  hir.unroll_for %i = 0 to 3 step 1 iter_time(%ti = %t){
    hir.yield at %ti 
    %0 = hir.constant (0) : !hir.const
    %1 = hir.constant (1) : !hir.const
  }
  hir.return
}
