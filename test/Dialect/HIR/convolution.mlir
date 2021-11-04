#r_bram = {"rd_latency"=1}
#r_reg = {"rd_latency"=1}
#w_reg = {"wr_latency"=1}
#w_bram = {"wr_latency"=1}
// RUN: circt-opt %s
hir.func.extern @weighted_average at %t(%wndw : !hir.memref<(bank 3)x(bank 3)xi32> ports [#r_reg,#w_reg]) -> (%res: i32 delay 2)

hir.func @convolution at %t(
%A :!hir.memref<16x16xi32> ports [#r_bram],
%B :!hir.memref<16x16xi32> ports [#w_bram]){

  %buff = hir.alloca("bram") :!hir.memref<(bank 3)x16xi32> ports [#r_bram,#w_bram]
  %wndw = hir.alloca("bram") :!hir.memref<(bank 3)x(bank 3)xi32> ports [#r_reg, #w_reg]

  %0 =  arith.constant 0 :index  
  %1 =  arith.constant 1 :index
  %2 =  arith.constant 2 :index
  %3 =  arith.constant 3 :index
  %4 =  arith.constant 4 :index
  %ub = arith.constant 15:index
  %c0_i4 = arith.constant 0 :i4 
  %c1_i4 = arith.constant 1 :i4
  %ub_i4 = arith.constant 15 :i4

  //Read from input. Update line buffer. Input values to each row of window.
  hir.for %i : i4 = %c0_i4 to %ub_i4 step %c1_i4 iter_time(%ti = %t + 1 ){
    %tf=hir.for %j : i4 = %c0_i4 to %ub_i4 step %c1_i4 iter_time(%tj = %ti + 1 ){
      %v =  hir.load %A[port 0][%i, %j] at %tj 
      : !hir.memref<16x16xi32>
      %v0 = hir.load %buff[port 0][%0,%j] at %tj
      : !hir.memref<(bank 3)x16xi32>
      %v1 = hir.load %buff[port 0][%1,%j] at %tj
      : !hir.memref<(bank 3)x16xi32>
      %v2 = hir.load %buff[port 0][%2,%j] at %tj
      : !hir.memref<(bank 3)x16xi32>

      %j1 = hir.delay %j by 1 at %tj :i4
      hir.store %v1 to %buff[port 1][%0,%j1] at %tj + 1 
      : !hir.memref<(bank 3)x16xi32>
      hir.store %v2 to %buff[port 1][%1,%j1] at %tj + 1 
      : !hir.memref<(bank 3)x16xi32>
      hir.store %v to %buff[port 1][%2,%j1] at %tj + 1 
      : !hir.memref<(bank 3)x16xi32>

      hir.store %v1 to %wndw[port 1][%0,%0] at %tj + 1
      : !hir.memref<(bank 3)x(bank 3)xi32>
      hir.store %v2 to %wndw[port 1][%1,%0] at %tj + 1
      : !hir.memref<(bank 3)x(bank 3)xi32>
      hir.store %v to %wndw[port 1][%2,%0] at %tj + 1
      : !hir.memref<(bank 3)x(bank 3)xi32>
      hir.next_iter at %tj + 1
    }
    hir.next_iter at %tf + 1
  }

  hir.for %i : i4 = %c0_i4  to %ub_i4 step %c1_i4 iter_time(%ti = %t + 1 ){
    %tf=hir.for %j : i4 = %c0_i4 to %ub_i4 step %c1_i4 iter_time(%tj = %ti + 1 ){
      hir.for %k1:index = %0 to %3 step %1 iter_time(%tk1 = %tj){
        hir.for %k2:index = %0 to %2 step %1 iter_time(%tk2 = %tk1){
          %v = hir.load %wndw[port 0][%k1,%k2] at %tk2 + 1
          : !hir.memref<(bank 3)x(bank 3)xi32>
          %k2Plus1 = arith.addi %k2, %1: index
          hir.store %v to %wndw[port 1][%k1,%k2Plus1] at %tk2 + 1
            :!hir.memref<(bank 3)x(bank 3)xi32>
          hir.next_iter at %tk2
        }{unroll}
        hir.next_iter at %tk1 
      }{unroll}
      hir.next_iter at %tj + 1
    }
    hir.next_iter at %tf + 1
  }

  hir.for %i : i4 = %c0_i4 to %ub_i4 step %c1_i4 iter_time(%ti = %t + 1 ){
    %t_end=hir.for %j : i4 = %c0_i4 to %ub_i4 step %c1_i4 iter_time(%tj = %ti + 1 ){
      %b1 = comb.icmp ugt %i, %c1_i4 : i4
      %b2 = comb.icmp ugt %j, %c1_i4 : i4
      %b3 = comb.and %b1, %b2  : i1

      hir.if %b3  at time(%tf = %tj){
        %v = hir.call @weighted_average(%wndw) at %tf + 2
          :!hir.func<(!hir.memref<(bank 3)x(bank 3)xi32> ports [#r_reg,#w_reg]) -> (i32 delay 2)>
        %i4 = hir.delay %i by 4 at %tf : i4 
        %j4 = hir.delay %j by 4 at %tf : i4 
        hir.store %v to %B[port 0][%i4,%j4] at %tf + 4
          :!hir.memref<16x16xi32>
        hir.yield
      }else{
        hir.yield
      }      
      hir.next_iter at %tj + 1
    }
    hir.next_iter at %t_end + 1
  }
  hir.return
}

