hir.func @counter at %t (%rst : !hir.wire<1*i1,r>,
%load : !hir.wire<1*i1,r>,
%out : !hir.wire<1*i4,w>){
  %0 = hir.constant 0 : !hir.const<i32>
  %1 = hir.constant 1 : !hir.const<i32>
  %Cr,%Cw = hir.alloc() : !hir.memref<1*i32, packing=[], r>, !hir.memref<1*i32, packing=[], w>
  hir.while iter_time(%ti = %t ){
    hir.yield %1 at %ti offset %1
    %r = hir.wire_read %rst[%0] at %ti : !hir.wire<1*i32>[!hir.const<i32>] -> i32
    %l = hir.wire_read %load[%0] at %ti : !hir.wire<1*i32>[!hir.const<i32>] -> i32
    %c = hir.mem_read %Cr[%0] at %ti : !hir.memref<1*i32, r>[!hir.const<i32>] -> i32
    hir.wire_write %c to %out[%0] at %ti : (i32, !hir.wire<1*i32>[!hir.const<i32>])
    hir.if(%r){
      hir.mem_write %0 to %Cw[%0] at %ti  : (!hir.const<i32>, !hir.memref<1*i32, packing=[], w>[!hir.const<i32>])
    }else{
      hir.if(%l) at %ti{
        hir.mem_write %c_next to %Cw[%0] at %ti  : (i32, !hir.memref<1*i32, packing=[], w>[!hir.const<i32>])
      }else{
        %c_next = hir.add (%c, %1) : (i32, !hir.const<i32>) -> (i32)
        hir.mem_write %c_next to %Cw[%0] at %ti  : (i32, !hir.memref<1*i32, packing=[], w>[!hir.const<i32>])
      }
    }
  }
  hir.return
}
