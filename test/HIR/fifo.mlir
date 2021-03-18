// RUN: circt-opt %s
!ty_buffer_r = type !hir.memref<256*i32, r>
!ty_buffer_w = type !hir.memref<256*i32, w>
!ty_reg_r = type !hir.memref<1*i32, packing=[], r>
!ty_reg_w = type !hir.memref<1*i32, packing=[], w>



hir.func @fifo at %t(
%in_r : !ty_buffer_r,
%out_w : !ty_buffer_w){
  %0 = hir.constant 0
  %1 = hir.constant 1
  %16 = hir.constant 16
  %256 = hir.constant 256

  %mem_r,%mem_w = hir.alloc() : !ty_buffer_r, !ty_buffer_w
  %push_counter_r,%push_counter_w = hir.alloc()  : !ty_reg_r, !ty_reg_w
  %pop_counter_r,%pop_counter_w = hir.alloc() : !ty_reg_r , !ty_reg_w

  hir.mem_write %0 to %push_counter_w[%0] at %t 
  : (!hir.const, !ty_reg_w[!hir.const])
  hir.mem_write %0 to %pop_counter_w[%0] at %t 
  : (!hir.const, !ty_reg_w[!hir.const])

  hir.for %i : i32 = %0 : !hir.const to %256 : !hir.const 
  step %1:!hir.const iter_time(%ti = %t offset %16 ){
    %v =  hir.mem_read %in_r[%i] at %ti : !ty_buffer_r[i32] -> i32
    %wr_addr =  hir.mem_read %push_counter_r[%0] at %ti offset %1
    : !ty_reg_r[!hir.const] -> i32

    %rd_addr =  hir.mem_read %pop_counter_r[%0] at %ti offset %1 
    : !ty_reg_r[!hir.const] -> i32

    %wr_addr_next = hir.add(%wr_addr,%1) : (i32,!hir.const) -> (i32)
    %b1 = hir.neq(%wr_addr_next,%rd_addr) : (i32,i32) -> (i1)
    %ti1 = hir.delay %ti by %1 at %ti : !hir.time -> !hir.time
    hir.if(%b1) at %ti1{
      hir.mem_write %wr_addr_next to %push_counter_w[%0] at %ti 
      : (i32, !ty_reg_w[!hir.const])

      hir.mem_write %v to %mem_w[%wr_addr] at %ti
      : (i32, !ty_buffer_w[i32])
    }
    hir.yield at %ti offset %1
  }

  hir.for %i : i32 = %0 : !hir.const to %256 : !hir.const 
  step %1:!hir.const iter_time(%ti = %t offset %1 ){

    %wr_addr =  hir.mem_read %push_counter_r[%0] at %ti
    : !ty_reg_r[!hir.const] -> i32

    %rd_addr =  hir.mem_read %pop_counter_r[%0] at %ti
    : !ty_reg_r[!hir.const] -> i32

    %rd_addr_next = hir.add(%rd_addr,%1) : (i32,!hir.const) -> (i32)
    %b1 = hir.neq(%rd_addr,%wr_addr) : (i32,i32) -> (i1)
    hir.if(%b1) at %ti{
      hir.mem_write %rd_addr_next to %pop_counter_w[%0] at %ti
      : (i32, !ty_reg_w[!hir.const])
    }

    %v =  hir.mem_read %mem_r[%rd_addr] at %ti : !ty_buffer_r[i32] -> i32
    %i1 = hir.delay %i by %1 at %ti : i32 -> i32
    hir.mem_write %v to %out_w[%i1] at %ti offset %1 
    : (i32, !ty_buffer_w[i32])
    hir.yield at %ti offset %1
  }
  hir.return
}
