// RUN: circt-opt %s
hir.func @test at %t() -> (%res:i4){
  %0 = arith.constant 0 : index
  %1 = arith.constant 1 : index
  %15 = arith.constant 15 : index

  %tt1 = hir.for %i:index = %0 to %15 step %1 iter_time(%ti = %t+1){
    %64  = arith.constant 64 : index
    %res = arith.addi %i,%64 :index
    hir.next_iter at %ti 
  }{unroll}

  %c0 = hw.constant 0   : i4
  %c1 = hw.constant 1   : i4
  %c15 = hw.constant 15 : i4

  %tt2 = hir.for %i:i4 = %c0 to %c15 step %c1 iter_time(%ti = %tt1+1){
    %c2  = hw.constant 2 : i4
    %res1 = comb.add %i, %c2 :i4
    hir.next_iter at %ti+1 
  }
  %b = hw.constant 1:i1
  %zz, %t_blah = hir.while %b  iter_args (%x = %c15:i4) iter_time(%tw = %tt2 + 2){
    %bb = hw.constant 1:i1
    %xx = hir.delay %x by 1 at %tw : i4
    hir.next_iter condition %bb iter_args(%xx) at %tw+1:(i4)
  }

  hir.comment "IfOp"
  %c = hw.constant 1 :i1
  %res2= hir.if %c  at time(%tf=%t_blah) -> (i4){
    hir.yield (%zz) :(i4)
  }else{ 
    hir.yield (%c0) :(i4)
  } 
  hir.return (%res2) :(i4)
}
