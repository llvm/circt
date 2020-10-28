// matmul kernel
// N is a constant
// %A, %B = hir.new_mem {ports=["r", "rw"]} : 
// !hir.memref<3dx4xi32, r>, !hir.memref<3dx4xi32, rw>

//<NxNxi8>{layout=[0][1], port=[rd, wr], 
//    storage_type=bram2P}
// %B = alloc() : hir.memref<NxNxi8>{layout=[1, 0], port=[rd, wr], 
//    storage_type=reg}
// %C = alloc() : hir.memref<NxNxi8>{layout=[0][1], port=[rd, wr], 
//    storage_type=bram2P}

hir.def @hirMatmulKernel at %t(
  %Ai :!hir.memref<16*16*i32, r>,
  %Bi : !hir.memref<16*16*i32, r>, 
  %Co : !hir.memref<16*16*i32, w>) -> () {

  %1 = hir.constant 1 : !hir.const<i32>
  %2 = hir.constant 2 : !hir.const<i32>
  %3 = hir.constant 3 : !hir.const<i32>
  %4 = hir.constant 4 : !hir.const<i32>
  %0 = hir.constant 0 : !hir.const<i32>
  %15 = hir.constant 15 : !hir.const<i32>
  %16 = hir.constant 16 : !hir.const<i32>
  %32 = hir.constant 32 : !hir.const<i32>
  
  %A,%Aw = hir.alloc() : !hir.memref<16*16*i32, packing=[1],r>, !hir.memref<16*16*i32, packing=[1], w>
  %B,%Bw = hir.alloc() : !hir.memref<16*16*i32, packing=[], r>, !hir.memref<16*16*i32, packing=[], w>
  %Cr,%C = hir.alloc() : !hir.memref<16*16*i32, packing=[1], r>, !hir.memref<16*16*i32, packing=[1], w>

  //Write to block ram A.
  hir.for %i : i32 = %0 : !hir.const<i32> to %16 : !hir.const<i32> step %1 : !hir.const<i32> iter_time(%ti = %t){
    %tk_end=hir.unroll_for %k = 0 to 16 step 1 iter_time(%tk = %ti){
      hir.yield at %tk offset %1 : !hir.const<i32>

      %v =  hir.mem_read %Ai[%i, %k] at %tk : !hir.memref<16*16*i32, r>[i32, !hir.const<i32>] -> i32
      %i1 = hir.delay %i by %1 : !hir.const<i32> at %tk : i32 -> i32 
      hir.mem_write %v to %Aw[%i1, %k] at %tk offset %1:!hir.const<i32> : (i32, !hir.memref<16*16*i32, packing=[1], w>[i32, !hir.const<i32>])
    }
    hir.yield at %tk_end 
  }

  //Write to register array B.
  %t_exec=hir.unroll_for %j = 0 to 16 step 1 iter_time(%tj = %t){
    hir.for %c : i32 = %0 : !hir.const<i32> to %16 : !hir.const<i32> step %1 : !hir.const<i32> iter_time(%tc = %tj){
      %cc = hir.subtract(%15, %c) : (!hir.const<i32>, i32) -> (i32)
      hir.yield at %tc offset %1 : !hir.const<i32>
      %v =  hir.mem_read %Bi[%cc,%j] at %tc : !hir.memref<16*16*i32, r>[i32,!hir.const<i32>] -> i32
      hir.mem_write %v to %Bw[%0,%j] at %tc offset %1 : !hir.const<i32> : (i32, !hir.memref<16*16*i32, packing=[], w>[!hir.const<i32>, !hir.const<i32>])
    }
    %tjPlus2 = hir.delay %tj by %2 : !hir.const<i32> at %tj : !hir.time -> !hir.time // hoist from j-loop
    %tk_end=hir.unroll_for %k = 0 to 15 step 1 iter_time(%tk = %tjPlus2){
      hir.for %c : i32 = %k : !hir.const<i32> to %15 : !hir.const<i32> step %1 : !hir.const<i32> iter_time(%tc = %tk){
        %kPlus1 = hir.add(%k, %1) : (!hir.const<i32>, !hir.const<i32>) -> (!hir.const<i32>)
        %v2 =  hir.mem_read %B[%k, %j] at %tc : !hir.memref<16*16*i32, packing=[],r>[!hir.const<i32>, !hir.const<i32>] -> i32
        hir.mem_write %v2 to %Bw[%kPlus1, %j] at %tc : (i32, !hir.memref<16*16*i32, packing=[], w>[!hir.const<i32>, !hir.const<i32>])
        hir.yield at %tc offset %1: !hir.const<i32>
      } 
      hir.yield at %tk offset %1: !hir.const<i32>
    }
    hir.yield at %tk_end
  }


  hir.for %i : i32 = %0 : !hir.const<i32> to %16 : !hir.const<i32> step %1 : !hir.const<i32> iter_time(%ti = %t_exec){
    hir.yield at %ti offset %1 : !hir.const<i32>
    hir.unroll_for %j = 0 to 16 step 1 iter_time(%tj = %ti){
      hir.yield at %tj offset %1 : !hir.const<i32>
      %C_bus = hir.alloc() : !hir.wire<17*i32>
      hir.wire_write %0 to %C_bus[%0] at %tj offset %3 : !hir.const<i32> : (!hir.const<i32>, !hir.wire<17*i32>[!hir.const<i32>])

      %tk_end=hir.unroll_for %k = 0 to 16 step 1 iter_time(%tk = %tj){
        hir.yield at %tk offset %1 : !hir.const<i32>
        %i_delayed = hir.delay %i by %k : !hir.const<i32> at %ti : i32 -> i32 // hoist from j-loop
        %a = hir.mem_read %A[%i_delayed, %k] at %ti offset %k : !hir.const<i32> : !hir.memref<16*16*i32, packing=[1], r>[i32, !hir.const<i32>] -> i32 // hoist from j-loop
        %a_delayed = hir.delay %a by %j : !hir.const<i32> at %ti offset %k : !hir.const<i32> : i32 -> i32
        %b = hir.mem_read %B[%k, %j] at %tk : !hir.memref<16*16*i32, packing=[], r>[!hir.const<i32>, !hir.const<i32>] -> i32
        %ab = hir.call @mult(%a_delayed, %b) at %tk offset %1: !hir.const<i32> : (i32, i32) -> (i32)
        %c_prev = hir.wire_read %C_bus[%k] at %tk offset %3 : !hir.const<i32> : !hir.wire<17*i32>[!hir.const<i32>] -> i32
        %c = hir.call @add(%ab, %c_prev) at %tk offset %3 : !hir.const<i32> : (i32, i32) -> (i32)
        %kPlus1 = hir.add(%k, %1) : (!hir.const<i32>, !hir.const<i32>) -> (!hir.const<i32>)
        hir.wire_write %c to %C_bus[%kPlus1] at %tk offset %4 : !hir.const<i32> : (i32, !hir.wire<17*i32>[!hir.const<i32>])
      }
      %acc = hir.wire_read %C_bus[%16] at %tk_end offset %3 : !hir.const<i32> : !hir.wire<17*i32>[!hir.const<i32>] -> i32
      %i1 = hir.delay %i by %j : !hir.const<i32> at %ti : i32 -> i32 // hoist from j-loop
      %i2 = hir.delay %i1 by %16 : !hir.const<i32> at %ti : i32 -> i32 // hoist from j-loop
      %i3 = hir.delay %i2 by %3 : !hir.const<i32> at %ti : i32 -> i32 // hoist from j-loop
      hir.mem_write %acc to %C[%i3, %j] at %tk_end offset %3 : !hir.const<i32> : (i32, !hir.memref<16*16*i32, packing=[1], w>[i32, !hir.const<i32>])
    }
  }

  %t_rd = hir.delay %t_exec by %32 :!hir.const<i32> at %t_exec : !hir.time -> !hir.time
  hir.for %i : i32 = %0 : !hir.const<i32> to %16 : !hir.const<i32> step %1 : !hir.const<i32> iter_time(%ti = %t_rd){
    %tnext = hir.unroll_for %j = 0 to 16 step 1 iter_time(%tj = %ti){
      hir.yield at %tj offset %1 : !hir.const<i32>
      %v = hir.mem_read %Cr[%i, %j] at %tj : !hir.memref<16*16*i32, packing=[1], r>[i32, !hir.const<i32>] -> i32
      hir.mem_write %v to %Co[%i, %j] at %tj offset %1 : !hir.const<i32> : (i32, !hir.memref<16*16*i32, w>[i32, !hir.const<i32>])
    }
    hir.yield at %tnext 
  }
   
  hir.return
}
