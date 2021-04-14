#bram_r = {"rd"= 1}
#bram_w = {"wr"= 1}
#reg_r  = {"rd" = 0}
#reg_w  = {"wr"= 1}

hir.func @gesummv at %t(
%alpha:i32, 
%beta:i32, 
%tmp:!hir.memref<8xi32, #bram_w> , 
%A:!hir.memref<8x8xi32,#bram_r>,
%B:!hir.memref<8x8xi32,#bram_r>,
%X:!hir.memref<8xi32,#bram_r>,
%Y:!hir.memref<8xi32,#bram_w>
){

  %buff_r,%buff_w = hir.alloca("bram") :!hir.memref<2x16xf32,[1], #bram_r>,
                    !hir.memref<2x16xf32,[1], #bram_w>

  %wndw_r,%wndw_w = hir.alloca("bram") :!hir.memref<2x2xf32,[0,1], #reg_r>,
  !hir.memref<2x2xf32,[0,1], #reg_w>

  %0 = hir.constant (0):!hir.const
  %1 = hir.constant (1):!hir.const
  %4 = hir.constant (4):!hir.const
  %5 = hir.constant (5):!hir.const
  %6 = hir.constant (6):!hir.const
  %8 = hir.constant (8):!hir.const
  %9 = hir.constant (9):!hir.const


  hir.for %i :i32 = %0 :!hir.const to %8 :!hir.const step %1 :!hir.const 
    iter_time(%ti = %t  +  %1 ){
    %tmpreg_r,%tmpreg_w = hir.alloca("reg") :!hir.memref<1xi32,[0], #reg_r>,
                    !hir.memref<1xi32,[0], #reg_w>
    %yreg_r,%yreg_w = hir.alloca("reg") :!hir.memref<1xi32,[0], #reg_r>,
                    !hir.memref<1xi32,[0], #reg_w>

    hir.store %0 to %tmpreg_w[%0] at %ti 
      :(!hir.const, !hir.memref<1xi32, [0], #reg_w>[!hir.const])
    hir.store %0 to %yreg_w[%0] at %ti 
      :(!hir.const, !hir.memref<1xi32, [0], #reg_w>[!hir.const])
    
    %tf=hir.for %j :i32 = %0 :!hir.const to %8 :!hir.const step %1 :!hir.const 
      iter_time(%tj = %ti  +  %1 ){
        

        %a_i_j = hir.load %A[%i,%j] at %tj
        : !hir.memref<8x8xi32, #bram_r>[i32,i32] -> i32
        %b_i_j = hir.load %B[%i,%j] at %tj
        : !hir.memref<8x8xi32, #bram_r>[i32,i32] -> i32
        %x_j = hir.load %X[%j] at %tj
        : !hir.memref<8xi32, #bram_r>[i32] -> i32

        %t1 = hir.call @i32Multiplier(%a_i_j,%x_j) at %tj+%1 
          : !hir.func<(i32,i32)->(i32 delay 4)>
        %tmp_in = hir.load %tmpreg_r[%0] at %tj + %5
          : !hir.memref<1xi32,[0], #reg_r>[!hir.const] -> i32
        %tmp_next = hir.call @i32Adder(%t1,%tmp_in) at %tj+%5
          : !hir.func<(i32,i32)->(i32)>
        hir.store %tmp_next to %tmpreg_w[%0] at %tj+%5 
          : (i32, !hir.memref<1xi32, [0], #reg_w>[!hir.const])

        %t2 = hir.call @i32Multiplier(%b_i_j,%x_j) at %tj+%1
          : !hir.func<(i32,i32)->(i32 delay 4)>
        %y = hir.load %yreg_r[%0] at %tj + %5
          :!hir.memref<1xi32,[0], #reg_r>[!hir.const] -> i32
        %y_next = hir.call @i32Adder(%t1,%y) at %tj+%5
          : !hir.func<(i32,i32)->(i32)>
        hir.store %y_next to %yreg_w[%0] at %tj+%5 
          : (i32, !hir.memref<1xi32, [0], #reg_w>[!hir.const])
        hir.yield at %tj + %1
    }
    %tmp_in = hir.load %tmpreg_r[%0] at %tf + %5
      :!hir.memref<1xi32,[0], #reg_r>[!hir.const] -> i32
    hir.store %tmp_in to %tmp[%i] at %tf + %5 
      :(i32, !hir.memref<8xi32,  #bram_w>[i32])
    %y = hir.load %yreg_r[%0] at %tf + %5
      :!hir.memref<1xi32,[0], #reg_r>[!hir.const] -> i32
    %alpha_tmp = hir.call @i32Multiplier(%alpha,%tmp_in) at %tf+%5
      : !hir.func<(i32,i32)->(i32 delay 4)>
    %beta_y = hir.call @i32Multiplier(%beta,%y) at %tf+%5
      : !hir.func<(i32,i32)->(i32 delay 4)>
    %y_next = hir.call @i32Adder(%alpha_tmp,%beta_y) at %tf+%9
      : !hir.func<(i32,i32)->(i32)>

    %i9 = hir.delay %i by %9 at %ti : i32 -> i32
    hir.store %y_next to %Y[%i9] at %tf + %9
      :(i32, !hir.memref<8xi32,  #bram_w>[i32])

    hir.yield at %tf + %5
  }

  hir.return
}
