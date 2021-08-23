// RUN: circt-opt %s
hir.func @test at %t(){
  %0 = constant 0 : index
  %1 = constant 1 : index
  %15 = constant 15 : index

  %tt1 = hir.for %i:index = %0 to %15 step %1 iter_time(%ti = %t+1){
    %64  = constant 64 : index
    %res = addi %i,%64 :index
    hir.next_iter at %ti 
  }{unroll}

  %c0 = constant 0   : i4
  %c1 = constant 1   : i4
  %c15 = constant 15 : i4

  %tt2 = hir.for %i:i4 = %c0 to %c15 step %c1 iter_time(%ti = %t+1){
    %c2  = constant 2 : i4
    %res = hir.addi (%i, %c2) at %ti:i4
    hir.next_iter at %ti+1 
  }
  %b = constant 1:i1
  hir.while(%b) at iter_time(%tw = %t + 2){
    %bb = constant 1:i1
    hir.condition %bb 
    hir.next_iter at %tw+1
  }
  hir.return
}
