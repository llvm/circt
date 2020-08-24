// matmul kernel
// N is a constant
// %A = alloc(): hir.memref<NxNxi8>{layout=[0][1],port=[rd,wr], 
//                          storage_type=bram2P}
// %B = alloc(): hir.memref<NxNxi8>{layout=[1,0],port=[rd,wr],
//                          storage_type=reg}
// %C = alloc(): hir.memref<NxNxi8>{layout=[0][1],port=[rd,wr], 
//                          storage_type=bram2P}

hir.def @MatmulKernel at %t (%A:!hir.mem_interface ,%B:!hir.mem_interface,%C:!hir.mem_interface) -> (!hir.int) {
  %s1 = hir.constant 1 : !hir.static_int
  %s3 = hir.constant 3: !hir.static_int
  %s4 = hir.constant 4: !hir.static_int
  %s0 = hir.constant 0 : !hir.static_int
  %0 = hir.cast %s0 at %t: !hir.static_int -> !hir.int
  %s128 = hir.constant 128 : !hir.static_int
  %128 = hir.cast %s128 at %t : !hir.static_int -> !hir.int
  %1 = hir.cast %s1 at %t : !hir.static_int -> !hir.int
  hir.for %i = %0 to %128 step %1 iter_time(%ti = %t): !hir.int{
    hir.yield at %ti offset %s1
    hir.unroll_for %j = 0 to 128 step 1 iter_time(%tj = %ti){
      hir.yield at %tj offset %s1
      %C_bus = "getNewWire"(){dims = "128xi32"}: () -> (!hir.wire)
      hir.wire_write %0 to %C_bus at %tj offset %s3 :
      (!hir.int,!hir.wire)
      %tk_end=hir.unroll_for %k = 0 to 128 step 1 iter_time(%tk = %tj tstep 1){
        %i_delayed = hir.delay %i by %k at %ti: !hir.int // hoist from j-loop
        %k_int = hir.cast %k at %ti offset %k:!hir.static_int -> !hir.int
        %a = hir.mem_read %A[%i_delayed, %k_int] at %ti offset %k: !hir.mem_interface -> !hir.int// hoist from j-loop
        %a_delayed = hir.delay %a by %j at %ti offset %k : !hir.int
        %j_int = hir.cast %j at %tk:!hir.static_int -> !hir.int
        %b = hir.mem_read %B[%j_int,%k_int] at %tk : !hir.mem_interface -> !hir.int
        %ab = hir.call @mult(%a_delayed,%b) at %tk
        :(!hir.int,!hir.int)->(!hir.int)
        %c_prev = hir.wire_read %C_bus[%k_int] at %tk offset %s3 : !hir.wire ->
        !hir.int
        %c  = hir.call @add(%ab,%c_prev) at %tk offset %s3
        :(!hir.int,!hir.int)->(!hir.int)
        %k_int2 = hir.cast %k at %tk:!hir.static_int -> !hir.int
        %_1 = hir.cast %s1 at %tk : !hir.static_int -> !hir.int
        %kPlus1 = hir.add at %tk (%k_int2,%_1)
        hir.wire_write %c to %C_bus[%kPlus1,%c] at %tk offset %s4:
        (!hir.int,!hir.wire)
      }
      %s127 = hir.constant 127 : !hir.static_int
      %127  = hir.cast %s127 at %tk_end: !hir.static_int -> !hir.int
      %acc = hir.wire_read %C_bus[%127] at %tk_end offset %s3: !hir.wire ->
      !hir.int
      %j_int = hir.cast %j at %tk_end offset %s3 :!hir.static_int -> !hir.int
      hir.mem_write %acc to %C[%i,%j_int] at %tk_end offset %s3 :
      (!hir.int,!hir.mem_interface)
    }
  }
  hir.return
}
