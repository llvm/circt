// RUN: circt-opt %s
#bram_r = {"rd_latency"=1}
#reg_r = {"rd_latency"=0}
#bram_w = {"wr_latency"=1}
#reg_w = {"wr_latency"=1}
hir.func.extern @mult at %t (%a:i32, %b:i32) -> (%result:i32 delay 2)

hir.func @readA at %t(
  %Ai :!hir.memref<16x16xi32> ports [#bram_r],
  %Aw : !hir.memref<16x(bank 16)xi32> ports [#bram_w]){

    %0 = arith.constant 0:index 
    %1 = arith.constant 1:index 
    %16 = arith.constant 16:index
    %c0_i5 = hw.constant 0:i5 
    %c1_i5 = hw.constant 1:i5 
    %c16_i5 = hw.constant 16:i5
    %c2_i5 = hw.constant 2:i5 
    %c4_i5 = hw.constant 4:i5 

  //Write to block ram A.
  hir.for %i : i5 = %c0_i5  to %c16_i5  step %c1_i5 iter_time(%ti = %t + 1){
    %tk_end=hir.for %k:index = %0 to %16 step %1 iter_time(%tk = %ti){
      %i_i4 = comb.extract %i from 0:(i5)->(i4)
      %k_i4 = hir.cast %k :index->i4
      %v =  hir.load %Ai[port 0][%i_i4, %k_i4] at %tk : !hir.memref<16x16xi32>
      %i1 = hir.delay %i_i4 by 1  at %tk : i4 
      hir.store %v to %Aw[port 0][%i1, %k] at %tk + 1 : !hir.memref<16x(bank 16)xi32>
      hir.next_iter at %tk + 1
    }
    hir.next_iter at %tk_end + 1
  }
  hir.return
}

hir.func @readB at %t(
  %Bi : !hir.memref<16x16xi32> ports [#bram_r], 
  %Bw : !hir.memref<(bank 16)x(bank 16)xi32> ports [#reg_w]) -> (%t_done:!hir.time){

    %0 = arith.constant 0:index 
    %1 = arith.constant 1:index 
    %16 = arith.constant 16:index
    %c0_i5 = hw.constant 0:i5 
    %c1_i5 = hw.constant 1:i5 
    %c16_i5 = hw.constant 16:i5
    %c2_i5 = hw.constant 2:i5 
    %c4_i5 = hw.constant 4:i5 

  //Write to register array B.
  %buff = hir.alloca("reg") : !hir.memref<(bank 1)xi32> ports [#reg_r, #reg_w]

  hir.for %j : i5 = %c0_i5 to %c16_i5 step %c1_i5 iter_time(%tj = %t + 1){
    %tk_end =hir.for %k : i5 = %c0_i5 to %c16_i5 step %c1_i5 iter_time(%tk = %tj + 1){
      %j_i4 = comb.extract %j from 0:(i5) -> (i4)
      %k_i4 = comb.extract %k from 0:(i5) -> (i4)
      %v =  hir.load %Bi[port 0][%k_i4,%j_i4] at %tk : !hir.memref<16x16xi32>
      hir.store %v to %buff[port 1][%0] at %tk + 1 :!hir.memref<(bank 1)xi32>
      hir.next_iter at %tk + 1 
    }
    hir.next_iter at %tk_end + 1 
  }

  %tt = hir.time %t + 3 : !hir.time 
  %t_exec=hir.for %j:index = %0 to %16 step %1 iter_time(%tj = %tt){
    %tj1 = hir.time %tj + 1 :!hir.time 
    %tk_end=hir.for %k:index = %0 to %16 step %1 iter_time(%tk = %tj1){
        %v =  hir.load %buff[port 0][%0] at %tk : !hir.memref<(bank 1)xi32>
        hir.store %v to %Bw[port 0][%k, %j] at %tk : !hir.memref<(bank 16)x(bank 16)xi32>
        hir.next_iter at %tk + 1
    }
    hir.next_iter at %tk_end + 1
  }
  hir.return (%t_exec) : (!hir.time)
} 

hir.func @kernel at %t(
  %A : !hir.memref<16x(bank 16)xi32> ports [#bram_r],
  %B : !hir.memref<(bank 16)x(bank 16)xi32> ports [#reg_r],
  %C: !hir.memref<16x(bank 16)xi32> ports [#bram_w]){

    %0 = arith.constant 0:index 
    %1 = arith.constant 1:index 
    %16 = arith.constant 16:index
    %c0_i5 = hw.constant 0:i5 
    %c1_i5 = hw.constant 1:i5 
    %c16_i5 = hw.constant 16:i5
    %c2_i5 = hw.constant 2:i5 
    %c4_i5 = hw.constant 4:i5 
    %c0_i32 = hw.constant 0:i32 

   hir.for %i : i5 = %c0_i5  to %c16_i5  step %c1_i5  iter_time(%ti = %t + 1){
     %A_reg = hir.alloca("reg") : !hir.memref<(bank 16)x(bank 16)xi32> ports[#reg_r,#reg_w]
     %i_reg = hir.alloca("reg") : !hir.memref<(bank 16)xi4> ports[#reg_r,#reg_w]
     %i_i4 = comb.extract %i from 0 : (i5)->(i4)
     hir.for %k:index = %0 to %16 step %1 iter_time(%tk = %ti){
       %a = hir.load %A[port 0][%i_i4,%k] at %tk : !hir.memref<16x(bank 16)xi32>
       hir.store %a to %A_reg[port 1][%0,%k] at %tk+1 :!hir.memref<(bank 16)x(bank 16)xi32>
       hir.next_iter at %tk + 1 
     }
     hir.store %i_i4 to %i_reg[port 1][%0] at %ti :!hir.memref<(bank 16)xi4>
     hir.for %j :index = %1 to %16 step %1 iter_time(%tj = %ti+1){
      %j-1 = arith.subi %j,%1 :index
      %i_prev = hir.load %i_reg[port 0][%j-1] at %tj:!hir.memref<(bank 16)xi4>
      hir.store %i_prev to %i_reg[port 1][%0] at %tj :!hir.memref<(bank 16)xi4>
      %tk_end=hir.for %k:index = %0 to %16 step %1 iter_time(%tk = %tj){
        %a = hir.load %A_reg[port 0][%j-1,%k] at %tk+1 : !hir.memref<(bank 16)x(bank 16)xi32>
        hir.store %a to %A_reg[port 1][%j,%k] at %tk+1 :!hir.memref<(bank 16)x(bank 16)xi32>
        hir.next_iter at %tk + 1 
      }
      hir.next_iter at %tj + 1 
     }

     hir.for %j :index = %0 to %16 step %1 iter_time(%tj = %ti){
      %C_reg = hir.alloca("reg") : !hir.memref<(bank 17)xi32> ports[#reg_r,#reg_w]
      hir.store %c0_i32 to %C_reg[port 1][%0] at %tj + 3 : !hir.memref<(bank 17)xi32>
      %tk_end=hir.for %k:index = %0 to %16 step %1 iter_time(%tk = %tj){
        %a = hir.load %A_reg[port 0][%j, %k] at %tk + 2  : !hir.memref<(bank 16)x(bank 16)xi32>
        %b = hir.load %B[port 0][%k, %j] at %tk+2 : !hir.memref<(bank 16)x(bank 16)xi32>
        %ab = hir.call @mult(%a, %b) at %tk + 2 : !hir.func<(i32, i32) -> (i32 delay 2)>
        %c_prev = hir.load %C_reg[port 0][%k] at %tk + 4 : !hir.memref<(bank 17)xi32>
        %c = comb.add %ab, %c_prev : i32
        %kPlus1 = arith.addi  %k, %1  : index
        hir.store %c to %C_reg[port 1][%kPlus1] at %tk + 4  : !hir.memref<(bank 17)xi32>
        hir.next_iter at %tk + 1 
      }
      %acc = hir.load %C_reg[port 0][%16] at %tj + 20 : !hir.memref<(bank 17)xi32>
      %i1 = hir.load %i_reg[port 0][%j] at %tj+1 :  !hir.memref<(bank 16)xi4>
      %i2 = hir.delay %i1 by 19 at %tj+1 : i4 
      hir.store %acc to %C[port 0][%i2, %j] at %tj + 20 : !hir.memref<16x(bank 16)xi32>
      hir.next_iter at %tj + 1 
    }
    hir.next_iter at %ti + 1 
  } 
  hir.return 
}

hir.func @writeC at %t(
  %Cr: !hir.memref<16x(bank 16)xi32> ports [#bram_r],
  %Co : !hir.memref<16x16xi32> ports [#bram_w]) {

    %0 = arith.constant 0:index 
    %1 = arith.constant 1:index 
    %16 = arith.constant 16:index
    %c0_i5 = hw.constant 0:i5 
    %c1_i5 = hw.constant 1:i5 
    %c16_i5 = hw.constant 16:i5
    %c2_i5 = hw.constant 2:i5 
    %c4_i5 = hw.constant 4:i5 
    %c0_i32 = hw.constant 0:i32 

  hir.for %i : i5 = %c0_i5  to %c16_i5  step %c1_i5  iter_time(%ti = %t + 1){
    %tnext = hir.for %j:index = %0 to %16 step %1 iter_time(%tj = %ti){
      %i_i4 = comb.extract %i from 0: (i5)->(i4)
      %v = hir.load %Cr[port 0][%i_i4, %j] at %tj : !hir.memref<16x(bank 16)xi32>
      %j_i4 = hir.cast %j : index -> i4
      hir.store %v to %Co[port 0][%i_i4, %j_i4] at %tj + 1  : !hir.memref<16x16xi32>
      hir.next_iter at %tj+1 
    }
    hir.next_iter at %tnext + 1
  }
  hir.return
}

hir.func @matmul at %t(
  %Ai :!hir.memref<16x16xi32> ports [#bram_r],
  %Bi : !hir.memref<16x16xi32> ports [#bram_r], 
  %Co : !hir.memref<16x16xi32> ports [#bram_w]) {

  %A = hir.alloca("bram") : !hir.memref<16x(bank 16)xi32> ports [#bram_r, #bram_w]
  %B = hir.alloca("reg") : !hir.memref<(bank 16)x(bank 16)xi32> ports [#reg_r,#reg_w]
  %C = hir.alloca("reg") : !hir.memref<16x(bank 16)xi32> ports [#bram_r,#bram_w]
  %Ar = hir.memref.extract %A[port 0] :!hir.memref<16x(bank 16)xi32> port [#bram_r]
  %Aw = hir.memref.extract %A[port 1] :!hir.memref<16x(bank 16)xi32> port [#bram_w]
  %Br = hir.memref.extract %B[port 0] :!hir.memref<(bank 16)x(bank 16)xi32> port [#reg_r]
  %Bw = hir.memref.extract %B[port 1] :!hir.memref<(bank 16)x(bank 16)xi32> port [#reg_w]
  %Cr = hir.memref.extract %C[port 0] :!hir.memref<16x(bank 16)xi32> port [#bram_r]
  %Cw = hir.memref.extract %C[port 1] :!hir.memref<16x(bank 16)xi32> port [#bram_w]
  
  hir.call @readA(%Ai,%Aw) at %t
  :!hir.func<(!hir.memref<16x16xi32> ports [#bram_r],!hir.memref<16x(bank 16)xi32> ports [#bram_w])>

  %t_exec = hir.call @readB(%Bi,%Bw) at %t 
  :!hir.func<(!hir.memref<16x16xi32> ports [#bram_r],!hir.memref<(bank 16)x(bank 16)xi32> ports [#reg_w]) -> (!hir.time)>

  hir.call @kernel(%Ar,%Br,%Cw) at %t_exec 
  : !hir.func<(!hir.memref<16x(bank 16)xi32> ports [#bram_r],
    !hir.memref<(bank 16)x(bank 16)xi32> ports [#reg_r],
    !hir.memref<16x(bank 16)xi32> ports [#bram_w])>

  %t_rd = hir.time %t_exec + 32 : !hir.time 
  hir.call @writeC(%Cr,%Co) at %t_rd 
    :!hir.func<(!hir.memref<16x(bank 16)xi32> ports [#bram_r], !hir.memref<16x16xi32> ports [#bram_w])>

  hir.return
}
