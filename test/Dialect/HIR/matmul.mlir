// RUN: circt-opt %s
#rd = {rd_latency=1}
#wr = {wr_latency=1}
#reg_rd = {rd_latency=0}
#reg_wr = {wr_latency=1}

hir.func @readA at %t(
%Ai :!hir.memref<16x16xi32> ports [#rd],
%A : !hir.memref<16x(bank 16)xi32> ports[#wr]) ->(!hir.time){

  %0  = constant 0  :index
  %1  = constant 1:index
  %15 = constant 15 :index
  %c0_i4  = constant 0:i4
  %c1_i4  = constant 1 :i4
  %c15_i4 = constant 15:i4

  //Write to block ram A.
  %t_done = hir.for %i : i4 = %c0_i4 to %c15_i4 step %c1_i4 iter_time(%ti = %t + 1){
    %tk_end=hir.for %k : index = %0 to %15 step %1 iter_time(%tk = %ti){
      %k_i4 = index_cast %k :index to i4
      %v =  hir.load %Ai[port 0][%i, %k_i4] at %tk : !hir.memref<16x16xi32> delay 1
      %i1 = hir.delay %i by 1  at %tk : i4
      hir.store %v to %A[port 0][%i1, %k] at %tk + 1 : !hir.memref<16x(bank 16)xi32> delay 1
      hir.yield at %tk + 1 
    }
    hir.yield at %tk_end + 1
  }
  hir.return (%t_done) :(!hir.time)
}{inline}

hir.func @readB at %t(
%Bi : !hir.memref<16x16xi32> ports [#rd], 
%Bw : !hir.memref<(bank 16)x(bank 16)xi32> ports [#wr]) -> (!hir.time){

  %0  = constant 0:index
  %1  = constant 1:index
  %15 = constant 15:index
  %c0_i4  = constant 0:i4
  %c1_i4  = constant 1 :i4
  %c15_i4 = constant 15:i4
  %c0_i1  = constant 0:i1

  //Write to register array B.
  %buff = hir.alloca("reg") : !hir.memref<1xi32> ports [#reg_rd,#reg_wr]

  hir.for %j : i4 = %c0_i4  to %c15_i4 step %c1_i4 iter_time(%tj = %t + 1){
    %tk_end =hir.for %k : i4 = %c0_i4 to %c15_i4 step %c1_i4 iter_time(%tk = %tj + 1){
      %v =  hir.load %Bi[port 0][%k,%j] at %tk : !hir.memref<16x16xi32> delay 1
      hir.store %v to %buff[port 1][%c0_i1] at %tk + 1 : !hir.memref<1xi32>
      hir.yield at %tk + 1 
    }
    hir.yield at %tk_end + 1 
  }

  %tt = hir.time %t + 3 : !hir.time
  %t_j_loop_done=hir.for %j:index = %0 to %15 step %1 iter_time(%tj = %tt){
    %tj1 = hir.time %tj + 1 :!hir.time
    %tk_end=hir.for %k:index = %0 to %15 step %1 iter_time(%tk = %tj+1){
      %v =  hir.load %buff[port 0][%c0_i1] at %tk : !hir.memref<1xi32>
      hir.store %v to %Bw[port 0][%k, %j] at %tk : !hir.memref<(bank 16)x(bank 16)xi32>
      hir.yield at %tk + 1
    }
    hir.yield at %tk_end + 1
  }
  %t_done = hir.time %t_j_loop_done + 16: !hir.time
  hir.return (%t_done) : (!hir.time)
}{inline} 

hir.func @kernel at %t(
%A : !hir.memref<16x(bank 16)xi32> ports [#rd],
%B : !hir.memref<(bank 16)x(bank 16)xi32> ports [#reg_rd],
%C: !hir.memref<16x(bank 16)xi32> ports [#wr])->(!hir.time){

  %0  = constant 0 :index 
  %1  = constant 1 :index 
  %15 = constant 15:index  
  %16 = constant 16:index  
  %c0_i4  = constant 0 :i4 
  %c0_i32  = constant 0 :i32 
  %c1_i4  = constant 1 :i4 
  %c15_i4 = constant 15:i4  

  %t_i_loop_done = hir.for %i : i4 = %c0_i4 to %c15_i4 step %c1_i4 iter_time(%ti = %t + 1){
  hir.for %j : index = %0 to %15 step %1 iter_time(%tj = %ti){
      %C_bus = hir.bus.instantiate : tensor<17x!hir.bus<i32>>
      %c_bus = hir.tensor.extract %C_bus[%0] : tensor<17x!hir.bus<i32>> -> !hir.bus<i32> ports ["send"]
      hir.send %c0_i32 to %c_bus[0] at %tj + 3 : i32 to !hir.bus<i32>

      %tk_end = hir.for %k:index = %0 to %15 step %1 iter_time(%tk = %tj){
        %isFirstCol = cmpi "eq", %k , %0 : index
        %a = hir.if %isFirstCol at %tk -> (i32 delay 1){
          %a = hir.load %A[port 0][%i, %k] at %tk  : !hir.memref<16x(bank 16)xi32>
          hir.yield (%a) at %tk : (i32 delay 1)
        }else{
          hir.yield (%c0_i32) at %tk : (i32 delay 1) //FIXME
        }

        %b = hir.load %B[port 0][%k, %j] at %tk : !hir.memref<(bank 16)x(bank 16)xi32>

        %ab = hir.call @mult(%a, %b) at %tk + 1 : !hir.func<(i32, i32) -> (i32 delay 2)>
        %c_bus_k = hir.tensor.extract %C_bus[%k] : tensor<17x!hir.bus<i32>> -> !hir.bus<i32> ports ["recv"]
        %c_prev = hir.recv %c_bus_k[0] at %tk + 3 : !hir.bus<i32> -> i32
        %c = hir.call @add(%ab, %c_prev) at %tk + 3  : !hir.func<(i32, i32) -> (i32 delay 1)>
        %kPlus1 = addi %k, %1 :index
        %c_bus_kplus1 = hir.tensor.extract %C_bus[%kPlus1]: tensor<17x!hir.bus<i32>> -> !hir.bus<i32> ports["send"]
        hir.send %c to %c_bus_kplus1[0] at %tk + 4  : i32 to !hir.bus<i32>
        hir.yield at %tk + 1 
      }
        %c_bus_last = hir.tensor.extract %C_bus[%16]: tensor<17x!hir.bus<i32>> -> !hir.bus<i32> ports["recv"]
      %acc = hir.recv %c_bus_last[0] at %tk_end + 3 : !hir.bus<i32> -> i32
      hir.store %acc to %C[port 0][%i, %j] at %tk_end + 3 : !hir.memref<16x(bank 16)xi32>
      hir.yield at %tj + 1 
    }
    hir.yield at %ti + 1 
  } 
  %t_done = hir.time %t_i_loop_done + 32 :!hir.time 

  hir.return (%t_done) :(!hir.time)
}{inline}

hir.func @writeC at %t(
%Ci: !hir.memref<16x(bank 16)xi32> ports [#rd],
%Co : !hir.memref<16x16xi32> ports [#wr]) -> (!hir.time){

  %0  = constant 0 :index 
  %1  = constant 1 :index 
  %15 = constant 15:index  
  %c0_i4  = constant 0:i4
  %c1_i4  = constant 1:i4
  %c15_i4 = constant 15:i4

  %t_i_loop_done = hir.for %i : i4 = %c0_i4 to %c15_i4 step %c1_i4 iter_time(%ti = %t){
    %tnext = hir.for %j:index = %0 to %15 step %1 iter_time(%tj = %ti){
      %v = hir.load %Ci[port 0][%i, %j] at %tj : !hir.memref<16x(bank 16)xi32>
      %j_i4 = index_cast %j : index to i4
      hir.store %v to %Co[port 0][%i, %j_i4] at %tj + 1  : !hir.memref<16x16xi32>
      hir.yield at %tj + 1 
    }
    hir.yield at %tnext + 1
  }
  %t_done = hir.time %t_i_loop_done + 16:!hir.time
  hir.return (%t_done):(!hir.time)
}{inline}

hir.func @matmul at %t(
%Ai :!hir.memref<16x16xi32> ports [#rd],
%Bi : !hir.memref<16x16xi32> ports [#rd], 
%Co : !hir.memref<16x16xi32> ports [#wr]) ->(!hir.time){

  %32 = constant 32 

  %A = hir.alloca("BRAM_2P") : !hir.memref<16x(bank 16)xi32> ports[#rd,#wr]
  %B = hir.alloca("REG") : !hir.memref<(bank 16)x(bank 16)xi32> ports[#reg_rd,#reg_wr]
  %C = hir.alloca("BRAM_2P") : !hir.memref<16x(bank 16)xi32> ports[#rd,#wr]

  %A_w = hir.memref.extract ports [1] from %A: !hir.memref<16x(bank 16)xi32> ports [#wr]
  %t1 = hir.call @readA(%Ai,%A_w) at %t
  :!hir.func<(!hir.memref<16x16xi32> ports [#rd],!hir.memref<16x(bank 16)xi32> ports[#wr])->(!hir.time)>

  %B_w = hir.memref.extract ports [1] from %B : !hir.memref<(bank 16)x(bank 16)xi32> ports [#reg_wr]
  %t2 = hir.call @readB(%Bi,%B_w) at %t 
  :!hir.func<(!hir.memref<16x16xi32> ports[#rd],
  !hir.memref<(bank 16)x(bank 16)xi32> ports[#reg_wr]) -> (!hir.time)>

  %t_kernel_start = hir.time.max(%t1,%t2):!hir.time

  %A_r = hir.memref.extract ports [0] from %A : !hir.memref<16x(bank 16)xi32> ports [#rd]
  %B_r = hir.memref.extract ports [0] from %B : !hir.memref<(bank 16)x(bank 16)xi32> ports [#reg_rd]
  %C_w = hir.memref.extract ports [1] from %C : !hir.memref<16x(bank 16)xi32> ports [#wr]

  %t_kernel_done = hir.call @kernel(%A_r,%B_r,%C_w) at %t_kernel_start 
  : !hir.func<(!hir.memref<16x(bank 16)xi32> ports [#rd],
  !hir.memref<(bank 16)x(bank 16)xi32> ports[#reg_rd],
  !hir.memref<16x(bank 16)xi32> ports [#wr]) ->(!hir.time)>

  %C_r = hir.memref.extract ports [0] from %C : !hir.memref<16x(bank 16)xi32> ports [#rd]
  %t_done = hir.call @writeC(%C_r,%Co) at %t_kernel_done
  :!hir.func<(!hir.memref<16x(bank 16)xi32> ports [#rd], !hir.memref<16x16xi32> ports[#wr]) ->(!hir.time)>

  hir.return (%t_done) :(!hir.time)
}
