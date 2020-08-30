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

hir.def @MatmulKernel at %t (%A:!hir.memref ,%B:!hir.memref,%C:!hir.memref) -> (!hir.val) {
  %s1 = hir.constant 1 : !hir.const
  %s3 = hir.constant 3: !hir.const
  %s4 = hir.constant 4: !hir.const
  %s0 = hir.constant 0 : !hir.const
  %0 = hir.cast %s0 at %t: !hir.const -> !hir.val
  %s128 = hir.constant 128 : !hir.const
  %128 = hir.cast %s128 at %t : !hir.const -> !hir.val
  %1 = hir.cast %s1 at %t : !hir.const -> !hir.val
  hir.for %i = %0 to %128 step %1 iter_time(%ti = %t): !hir.val{
    hir.yield at %ti offset %s1
    hir.unroll_for %j = 0 to 128 step 1 iter_time(%tj = %ti){
      hir.yield at %tj offset %s1
      %C_bus = hir.wire {type = "128xi32"}
      hir.wire_write %0 to %C_bus at %tj offset %s3 :
      (!hir.val,!hir.wire)
      %tk_end=hir.unroll_for %k = 0 to 128 step 1 iter_time(%tk = %tj tstep 1){
        %i_delayed = hir.delay %i by %k at %ti: !hir.val // hoist from j-loop
        %a = hir.mem_read %A[%i_delayed, %k] at %ti offset %k:
        !hir.memref[!hir.val,!hir.const] -> !hir.val// hoist from j-loop
        %a_delayed = hir.delay %a by %j at %ti offset %k : !hir.val
        %b = hir.mem_read %B[%j,%k] at %tk :
        !hir.memref[!hir.const,!hir.const] -> !hir.val
        %ab = hir.call @mult(%a_delayed,%b) at %tk
        :(!hir.val,!hir.val)->(!hir.val)
        %c_prev = hir.wire_read %C_bus[%k] at %tk offset %s3 : !hir.wire ->
        !hir.val
        %c  = hir.call @add(%ab,%c_prev) at %tk offset %s3
        :(!hir.val,!hir.val)->(!hir.val)
        %kPlus1 = hir.const_add(%k,%s1)
        hir.wire_write %c to %C_bus[%kPlus1] at %tk offset %s4:
        (!hir.val,!hir.wire)
      }
      %s127 = hir.constant 127 : !hir.const
      %acc = hir.wire_read %C_bus[%s127] at %tk_end offset %s3: !hir.wire ->
      !hir.val
      hir.mem_write %acc to %C[%i,%j] at %tk_end offset %s3 :
      (!hir.val,!hir.memref[!hir.val,!hir.const])
    }
  }
  hir.return
}
