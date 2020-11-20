hir.func @readA at %t(
  %Ai :!hir.memref<16*16*i32, r>,
  %Aw : !hir.memref<16*16*i32, packing=[1], w>){

  %0 = hir.constant 0 
  %1 = hir.constant 1 
  %2 = hir.constant 2 
  %4 = hir.constant 4 
  %16 = hir.constant 16 

  //Write to block ram A.
  hir.for %i : i32 = %0 : !hir.const to %16 : !hir.const step %1 : !hir.const iter_time(%ti = %t offset %1){
    %tk_end=hir.unroll_for %k = 0 to 16 step 1 iter_time(%tk = %ti){
      hir.yield at %tk offset %1 

      %v =  hir.mem_read %Ai[%i, %k] at %tk : !hir.memref<16*16*i32, r>[i32, !hir.const] -> i32
      %i1 = hir.delay %i by %1  at %tk : i32 -> i32 
      hir.mem_write %v to %Aw[%i1, %k] at %tk offset %1 : (i32, !hir.memref<16*16*i32, packing=[1], w>[i32, !hir.const])
    }
    hir.yield at %tk_end offset %1
  }
  hir.return
}

hir.func @readB at %t(
  %Bi : !hir.memref<16*16*i32, r>, 
  %Bw : !hir.memref<16*16*i32, packing=[], w>) -> (!hir.time){

  %0 = hir.constant 0 
  %1 = hir.constant 1 
  %2 = hir.constant 2 
  %3 = hir.constant 3 
  %4 = hir.constant 4 
  %16 = hir.constant 16 

  //Write to register array B.
  %buff,%buffw = hir.alloc() : !hir.memref<1*i32, packing=[], r>, !hir.memref<1*i32, packing=[], w>

  hir.for %j : i32 = %0 : !hir.const to %16 : !hir.const step %1 : !hir.const iter_time(%tj = %t offset %1){
    %tk_end =hir.for %k : i32 = %0 : !hir.const to %16 : !hir.const step %1 : !hir.const iter_time(%tk = %tj offset %1){
      hir.yield at %tk offset %1 
      %v =  hir.mem_read %Bi[%k,%j] at %tk : !hir.memref<16*16*i32, r>[i32,i32] -> i32
      hir.mem_write %v to %buffw[%0] at %tk offset %1 : (i32, !hir.memref<1*i32, packing=[], w>[!hir.const])
    }
    hir.yield at %tk_end offset %1 
  }

  %tt = hir.delay %t by %3 at %t : !hir.time -> !hir.time
  %t_exec=hir.unroll_for %j = 0 to 16 step 1 iter_time(%tj = %tt){
    %tj1 = hir.delay %tj by %1 at %tj :!hir.time -> !hir.time
    %tk_end=hir.unroll_for %k = 0 to 16 step 1 iter_time(%tk = %tj1){
        hir.yield at %tk offset %1
        %v =  hir.mem_read %buff[%0] at %tk : !hir.memref<1*i32, packing=[], r>[!hir.const] -> i32
        hir.mem_write %v to %Bw[%k, %j] at %tk : (i32, !hir.memref<16*16*i32, packing=[], w>[!hir.const, !hir.const])
    }
    hir.yield at %tk_end offset %1
  }
  hir.return (%t_exec) : (!hir.time)
} 

hir.func @kernel at %t(
  %A : !hir.memref<16*16*i32, packing=[1], r>,
  %B : !hir.memref<16*16*i32, packing=[], r>,
  %C: !hir.memref<16*16*i32, packing=[1], w>){

  %0 = hir.constant 0 
  %1 = hir.constant 1 
  %2 = hir.constant 2 
  %3 = hir.constant 3 
  %4 = hir.constant 4 
  %16 = hir.constant 16 

   hir.for %i : i32 = %0 : !hir.const to %16 : !hir.const step %1 : !hir.const iter_time(%ti = %t offset %1){
    hir.yield at %ti offset %1 
    hir.unroll_for %j = 0 to 16 step 1 iter_time(%tj = %ti){
      hir.yield at %tj offset %1 
      %C_bus = hir.alloc() : !hir.wire<17*i32>
      hir.wire_write %0 to %C_bus[%0] at %tj offset %3 : (!hir.const, !hir.wire<17*i32>[!hir.const])

      %tk_end=hir.unroll_for %k = 0 to 16 step 1 iter_time(%tk = %tj){
        hir.yield at %tk offset %1 
        %i_delayed = hir.delay %i by %k at %ti : i32 -> i32 // hoist from j-loop
        %a = hir.mem_read %A[%i_delayed, %k] at %ti offset %k  : !hir.memref<16*16*i32, packing=[1], r>[i32, !hir.const] -> i32 // hoist from j-loop
        %kPlus1 = hir.add(%k, %1) : (!hir.const, !hir.const) -> (!hir.const)
        %a_delayed = hir.delay %a by %j at %ti offset %kPlus1 : i32 -> i32
        %b = hir.mem_read %B[%k, %j] at %tk : !hir.memref<16*16*i32, packing=[], r>[!hir.const, !hir.const] -> i32
        %ab = hir.call @mult(%a_delayed, %b) at %tk offset %1 : (i32, i32) -> (i32 delay 2)
        %c_prev = hir.wire_read %C_bus[%k] at %tk offset %3 : !hir.wire<17*i32>[!hir.const] -> i32
        %c = hir.call @add(%ab, %c_prev) at %tk offset %3  : (i32, i32) -> (i32 delay 1)
        hir.wire_write %c to %C_bus[%kPlus1] at %tk offset %4  : (i32, !hir.wire<17*i32>[!hir.const])
      }
      %acc = hir.wire_read %C_bus[%16] at %tk_end offset %3 : !hir.wire<17*i32>[!hir.const] -> i32
      %i1 = hir.delay %i by %j at %ti : i32 -> i32 // hoist from j-loop
      %i2 = hir.delay %i1 by %16 at %ti : i32 -> i32 // hoist from j-loop
      %i3 = hir.delay %i2 by %3 at %ti : i32 -> i32 // hoist from j-loop
      hir.mem_write %acc to %C[%i3, %j] at %tk_end offset %3 : (i32, !hir.memref<16*16*i32, packing=[1], w>[i32, !hir.const])
    }
  } 
  hir.return 
}

hir.func @writeC at %t(
  %Cr: !hir.memref<16*16*i32, packing=[1], r>,
  %Co : !hir.memref<16*16*i32, w>) {

  %0 = hir.constant 0 
  %1 = hir.constant 1 
  %2 = hir.constant 2 
  %3 = hir.constant 3 
  %4 = hir.constant 4 
  %16 = hir.constant 16 

  hir.for %i : i32 = %0 : !hir.const to %16 : !hir.const step %1 : !hir.const iter_time(%ti = %t offset %1){
    %tnext = hir.unroll_for %j = 0 to 16 step 1 iter_time(%tj = %ti){
      hir.yield at %tj offset %1 
      %v = hir.mem_read %Cr[%i, %j] at %tj : !hir.memref<16*16*i32, packing=[1], r>[i32, !hir.const] -> i32
      hir.mem_write %v to %Co[%i, %j] at %tj offset %1  : (i32, !hir.memref<16*16*i32, w>[i32, !hir.const])
    }
    hir.yield at %tnext offset %1
  }
  hir.return
}

hir.func @matmul at %t(
  %Ai :!hir.memref<16*16*i32, r>,
  %Bi : !hir.memref<16*16*i32, r>, 
  %Co : !hir.memref<16*16*i32, w>) {

    %32 = hir.constant 32 
  
  %A,%Aw = hir.alloc() : !hir.memref<16*16*i32, packing=[1],r>, !hir.memref<16*16*i32, packing=[1], w>
  %B,%Bw = hir.alloc() : !hir.memref<16*16*i32, packing=[], r>, !hir.memref<16*16*i32, packing=[], w>
  %Cr,%C = hir.alloc() : !hir.memref<16*16*i32, packing=[1], r>, !hir.memref<16*16*i32, packing=[1], w>
  
  hir.call @readA(%Ai,%Aw) at %t
  :(!hir.memref<16*16*i32, r>,!hir.memref<16*16*i32, packing=[1], w>)

  %t_exec = hir.call @readB(%Bi,%Bw) at %t 
  :(!hir.memref<16*16*i32, r>,!hir.memref<16*16*i32, packing=[], w>) -> (!hir.time)

  hir.call @kernel(%A,%B,%C) at %t_exec 
  : (!hir.memref<16*16*i32, packing=[1], r>,
  !hir.memref<16*16*i32, packing=[], r>,
  !hir.memref<16*16*i32, packing=[1], w>)

  %t_rd = hir.delay %t_exec by %32 at %t_exec : !hir.time -> !hir.time
  hir.call @writeC(%Cr,%Co) at %t_rd 
    :(!hir.memref<16*16*i32, packing=[1], r>, !hir.memref<16*16*i32, w>)

  hir.return
}
