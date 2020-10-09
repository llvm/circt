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

hir.def @MatmulKernel at %t(
  %A : !hir.memref<16*16*i32, packing=[1], r>, 
  %B : !hir.memref<16*16*i32, packing=[], r>, 
  %C : !hir.memref<16*16*i32, packing=[1], w>) -> (i32) {

  %1 = hir.constant 1 : !hir.const<i32>
  %3 = hir.constant 3 : !hir.const<i32>
  %4 = hir.constant 4 : !hir.const<i32>
  %0 = hir.constant 0 : !hir.const<i32>
  %16 = hir.constant 16 : !hir.const<i32>
  hir.for %i : i32 = %0 : !hir.const<i32> to %16 : !hir.const<i32> step %1 : !hir.const<i32> iter_time(%ti = %t){
    hir.yield at %ti offset %1 : !hir.const<i32>
    hir.unroll_for %j = 0 to 16 step 1 iter_time(%tj = %ti){
      hir.yield at %tj offset %1 : !hir.const<i32>
      %C_bus = hir.alloc() : !hir.wire<16*i32>
      hir.wire_write %0 to %C_bus[%0] at %tj offset %3 : !hir.const<i32> : (!hir.const<i32>, !hir.wire<16*i32>[!hir.const<i32>])

      %tk_end=hir.unroll_for %k = 0 to 16 step 1 iter_time(%tk = %tj){
        hir.yield at %tk offset %1 : !hir.const<i32>
        %i_delayed = hir.delay %i by %k : !hir.const<i32> at %ti : i32 -> i32 // hoist from j-loop
        %a = hir.mem_read %A[%i_delayed, %k] at %ti offset %k : !hir.const<i32> : !hir.memref<16*16*i32, packing=[1], r>[i32, !hir.const<i32>] -> i32 // hoist from j-loop
        %a_delayed = hir.delay %a by %j : !hir.const<i32> at %ti offset %k : !hir.const<i32> : i32 -> i32
        %b = hir.mem_read %B[%j, %k] at %tk : !hir.memref<16*16*i32, packing=[], r>[!hir.const<i32>, !hir.const<i32>] -> i32
        %ab = hir.call @mult(%a_delayed, %b) at %tk : (i32, i32) -> (i32)
        %c_prev = hir.wire_read %C_bus[%k] at %tk offset %3 : !hir.const<i32> : !hir.wire<16*i32>[!hir.const<i32>] -> i32
        %c = hir.call @add(%ab, %c_prev) at %tk offset %3 : !hir.const<i32> : (i32, i32) -> (i32)
        %kPlus1 = hir.add(%k, %1) : (!hir.const<i32>, !hir.const<i32>) -> (!hir.const<i32>)
        hir.wire_write %c to %C_bus[%kPlus1] at %tk offset %4 : !hir.const<i32> : (i32, !hir.wire<16*i32>[!hir.const<i32>])
      }
      %127 = hir.constant 127 : !hir.const<i32>
      %acc = hir.wire_read %C_bus[%127] at %tk_end offset %3 : !hir.const<i32> : 
      !hir.wire<16*i32>[!hir.const<i32>] -> 
      i32
      hir.mem_write %acc to %C[%i, %j] at %tk_end offset %3 : !hir.const<i32> : 
      (i32, !hir.memref<16*16*i32, packing=[1], w>[i32, !hir.const<i32>])
    }
  }
  hir.return
}
