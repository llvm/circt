// matmul kernel
// N is a constant
// %A, %B = hir.new_mem {ports=["r","rw"]} :
// !hir.memref<3dx4xi32,r>, !hir.memref<3dx4xi32,rw>

//<NxNxi8>{layout=[0][1],port=[rd,wr], 
//                          storage_type=bram2P}
// %B = alloc(): hir.memref<NxNxi8>{layout=[1,0],port=[rd,wr],
//                          storage_type=reg}
// %C = alloc(): hir.memref<NxNxi8>{layout=[0][1],port=[rd,wr], 
//                          storage_type=bram2P}

hir.def @MatmulKernel at %t
(%A:!hir.memref<128*128*i32,packing=[1],r>,%B:!hir.memref<128*128*i32,packing=[],r>,
%C:!hir.memref<128*128*i32,packing=[1],w>) ->
(i32) {
  %s1 = hir.constant 1 : !hir.const<i32>
  %s3 = hir.constant 3: !hir.const<i32>
  %s4 = hir.constant 4: !hir.const<i32>
  %s0 = hir.constant 0 : !hir.const<i32>
  %s128 = hir.constant 128 : !hir.const<i32>
  %128 = hir.cast %s128 at %t : !hir.const<i32> -> i32
  %1 = hir.cast %s1 at %t : !hir.const<i32> -> i32
  hir.for %i:i32 = %s0:!hir.const<i32> to %128:i32 step %1:i32 iter_time(%ti = %t){
    hir.yield at %ti offset %s1:!hir.const<i32>

    hir.unroll_for %j = 0 to 128 step 1 iter_time(%tj = %ti){
      hir.yield at %tj offset %s1:!hir.const<i32>
      %C_bus = hir.wire : !hir.wire<128*i32>
      hir.wire_write %s0 to %C_bus[%s0] at %tj offset %s3:!hir.const<i32> :
      (!hir.const<i32>,!hir.wire<128*i32>[!hir.const<i32>])

      %tk_end=hir.unroll_for %k = 0 to 128 step 1 iter_time(%tk = %tj tstep 1){
        %i_delayed = hir.delay %i by %k:!hir.const<i32> at %ti: i32 -> i32 // hoist from j-loop
        %a = hir.mem_read %A[%i_delayed, %k] at %ti offset %k:!hir.const<i32>:
        !hir.memref<128*128*i32,packing=[1],r>[i32,!hir.const<i32>] -> i32 // hoist from j-loop
        %a_delayed = hir.delay %a by %j:!hir.const<i32> at %ti offset %k:!hir.const<i32> : i32 ->i32
        %b = hir.mem_read %B[%j,%k] at %tk :
        !hir.memref<128*128*i32,packing=[],r>[!hir.const<i32>,!hir.const<i32>] -> i32
        %ab = hir.call @mult(%a_delayed,%b) at %tk
        :(i32,i32)->(i32)
        %c_prev = hir.wire_read %C_bus[%k] at %tk offset %s3:!hir.const<i32> :
        !hir.wire<128*i32>[!hir.const<i32>] ->
        i32
        %c  = hir.call @add(%ab,%c_prev) at %tk offset %s3:!hir.const<i32>
        :(i32,i32)->(i32)
        %kPlus1 = hir.const_add(%k,%s1) : (!hir.const<i32>,!hir.const<i32>) -> !hir.const<i32>
        hir.wire_write %c to %C_bus[%kPlus1] at %tk offset %s4:!hir.const<i32>:
        (i32,!hir.wire<128*i32>[!hir.const<i32>])
      }
      %s127 = hir.constant 127 : !hir.const<i32>
      %acc = hir.wire_read %C_bus[%s127] at %tk_end offset %s3:!hir.const<i32>:
      !hir.wire<128*i32>[!hir.const<i32>] ->
      i32
      hir.mem_write %acc to %C[%i,%j] at %tk_end offset %s3:!hir.const<i32> :
      (i32,!hir.memref<128*128*i32,packing=[1],w>[i32,!hir.const<i32>])
    }
  }
  hir.return
}
