#bram_r = {"rd_latency"= 1}
#bram_w = {"wr_latency"= 1}
#reg_r  = {"rd_latency" = 0}
#reg_w  = {"wr_latency"= 1}
hir.func.extern @i32Multiplier at %t (%a:i32, %b:i32) ->(%result: i32 delay 4)

hir.func @gesummv at %t(
%alpha:i32, 
%beta:i32, 
%tmp:!hir.memref<8xi32> ports [#bram_w] , 
%A:!hir.memref<8x8xi32>ports [#bram_r],
%B:!hir.memref<8x8xi32>ports [#bram_r],
%X:!hir.memref<8xi32>ports [#bram_r],
%Y:!hir.memref<8xi32>ports [#bram_w]
){


  %0 = arith.constant 0:index
  %c0_i32 = hw.constant 0:i32
  %c0_i4 = hw.constant 0:i4
  %c1_i4 = hw.constant 1:i4
  %c4_i4 = hw.constant 4:i4
  %c5_i4 = hw.constant 5:i4
  %c6_i4 = hw.constant 6:i4
  %c8_i4 = hw.constant 8:i4
  %c9_i4 = hw.constant 9:i4


  hir.for %i :i4 = %c0_i4  to %c8_i4 step %c1_i4  iter_time(%ti = %t  +  1 ){
    %tmpreg = hir.alloca "reg"  :!hir.memref<(bank 1)xi32> ports [#reg_r, #reg_w]
    %yreg = hir.alloca "reg"  :!hir.memref<(bank 1)xi32> ports [#reg_r, #reg_w]

    hir.store %c0_i32 to %tmpreg[port 1][%0] at %ti 
      : !hir.memref<(bank 1)xi32>
    hir.store %c0_i32 to %yreg[port 1][%0] at %ti 
      : !hir.memref<(bank 1)xi32>
    
    %i_i3 = comb.extract %i from 0 :(i4)->(i3)
    %tf=hir.for %j :i4 = %c0_i4  to %c8_i4  step %c1_i4  iter_time(%tj = %ti  +  1 ){
        %j_i3 = comb.extract %j from 0 :(i4)->(i3)
        %a_i_j = hir.load %A[port 0][%i_i3,%j_i3] at %tj
        : !hir.memref<8x8xi32>
        %b_i_j = hir.load %B[port 0][%i_i3,%j_i3] at %tj
        : !hir.memref<8x8xi32>
        %x_j = hir.load %X[port 0][%j_i3] at %tj
        : !hir.memref<8xi32>

        %t1 = hir.call @i32Multiplier(%a_i_j,%x_j) at %tj+1 
          : !hir.func<(i32,i32)->(i32 delay 4)>
        %tmp_in = hir.load %tmpreg[port 0][%0] at %tj + 5
          : !hir.memref<(bank 1)xi32>
        %tmp_next = comb.add %t1, %tmp_in :i32
        hir.store %tmp_next to %tmpreg[port 1][%0] at %tj+5 
          : !hir.memref<(bank 1)xi32>

        %t2 = hir.call @i32Multiplier(%b_i_j,%x_j) at %tj+1
          : !hir.func<(i32,i32)->(i32 delay 4)>
        %y = hir.load %yreg[port 0][%0] at %tj + 5
          :!hir.memref<(bank 1)xi32>
        %y_next = comb.add %t1, %y :i32
        hir.store %y_next to %yreg[port 1][%0] at %tj+5 
          : !hir.memref<(bank 1)xi32>
        hir.next_iter at %tj + 1
    }
    %tmp_in = hir.load %tmpreg[port 0][%0] at %tf + 5
      :!hir.memref<(bank 1)xi32>
    hir.store %tmp_in to %tmp[port 0][%i_i3] at %tf + 5 
      : !hir.memref<8xi32>
    %y = hir.load %yreg[port 0][%0] at %tf + 5
      :!hir.memref<(bank 1)xi32>
    %alpha_tmp = hir.call @i32Multiplier(%alpha,%tmp_in) at %tf+5
      : !hir.func<(i32,i32)->(i32 delay 4)>
    %beta_y = hir.call @i32Multiplier(%beta,%y) at %tf+5
      : !hir.func<(i32,i32)->(i32 delay 4)>
    %y_next = comb.add %alpha_tmp, %beta_y : i32 

    %i9 = hir.delay %i_i3 by 9 at %ti : i3
    hir.store %y_next to %Y[port 0][%i9] at %tf + 9
      : !hir.memref<8xi32>

    hir.next_iter at %tf + 5
  }

  hir.return
}
