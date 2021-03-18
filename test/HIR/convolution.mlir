// RUN: circt-opt %s
hir.func @convolution at %t(
%A :!hir.memref<16*16*i32, r>,
%B :!hir.memref<16*16*i32, w>){

  %buff,%buffW = hir.alloc() :!hir.memref<3*16*i32,packing=[0], r>,
  !hir.memref<3*16*i32,packing=[0], w>
  %wndw,%wndwW = hir.alloc() :!hir.memref<3*3*i32,packing=[], r>,
  !hir.memref<3*3*i32,packing=[], w>

  %0 = hir.constant 0
  %1 = hir.constant 1
  %2 = hir.constant 2
  %3 = hir.constant 3
  %4 = hir.constant 4
  %16 = hir.constant 16

  //Read from input. Update line buffer. Input values to each row of window.
  hir.for %i : i32 = %0 : !hir.const to %16 : !hir.const 
  step %1:!hir.const iter_time(%ti = %t offset %1 ){
    %tf=hir.for %j : i32 = %0 : !hir.const to %16 : !hir.const 
    step %1:!hir.const iter_time(%tj = %ti offset %1 ){
      hir.yield at %tj offset %1
      %v =  hir.mem_read %A[%i, %j] at %tj 
      : !hir.memref<16*16*i32, r>[i32,i32] -> i32
      %v0 = hir.mem_read %buff[%0,%j] at %tj
      : !hir.memref<3*16*i32,packing=[0], r>[!hir.const,i32] -> i32
      %v1 = hir.mem_read %buff[%1,%j] at %tj
      : !hir.memref<3*16*i32,packing=[0], r>[!hir.const,i32] -> i32
      %v2 = hir.mem_read %buff[%2,%j] at %tj
      : !hir.memref<3*16*i32,packing=[0], r>[!hir.const,i32] -> i32

      %j1 = hir.delay %j by %1 at %tj :i32 -> i32
      hir.mem_write %v1 to %buffW[%0,%j1] at %tj offset %1 
      : (i32, !hir.memref<3*16*i32, packing=[0], w>[!hir.const,i32])
      hir.mem_write %v2 to %buffW[%1,%j1] at %tj offset %1 
      : (i32, !hir.memref<3*16*i32, packing=[0], w>[!hir.const,i32])
      hir.mem_write %v to %buffW[%2,%j1] at %tj offset %1 
      : (i32, !hir.memref<3*16*i32, packing=[0], w>[!hir.const,i32])

      hir.mem_write %v1 to %wndwW[%0,%0] at %tj offset %1
      : (i32, !hir.memref<3*3*i32, packing=[], w>[!hir.const,!hir.const])
      hir.mem_write %v2 to %wndwW[%1,%0] at %tj offset %1
      : (i32, !hir.memref<3*3*i32, packing=[], w>[!hir.const,!hir.const])
      hir.mem_write %v to %wndwW[%2,%0] at %tj offset %1
      : (i32, !hir.memref<3*3*i32, packing=[], w>[!hir.const,!hir.const])
    }
    hir.yield at %tf offset %1
  }

  hir.for %i : i32 = %0 : !hir.const to %16 : !hir.const 
  step %1:!hir.const iter_time(%ti = %t offset %1 ){
    %tf=hir.for %j : i32 = %0 : !hir.const to %16 : !hir.const 
    step %1:!hir.const iter_time(%tj = %ti offset %1 ){
      hir.yield at %tj offset %1
      hir.unroll_for %k1 = 0 to 3 step 1 iter_time(%tk1 = %tj){
        hir.yield at %tk1 
        hir.unroll_for %k2 = 0 to 2 step 1 iter_time(%tk2 = %tk1){
          hir.yield at %tk2
          %v = hir.mem_read %wndw[%k1,%k2] at %tk2 offset %1
          : !hir.memref<3*3*i32, packing=[], r>[!hir.const,!hir.const] ->
          i32
          %k2Plus1 = hir.add(%k2,%1) :(!hir.const,!hir.const) -> (!hir.const)
          hir.mem_write %v to %wndwW[%k1,%k2Plus1] at %tk2 offset %1
            :(i32, !hir.memref<3*3*i32, packing=[], w> [!hir.const,!hir.const])
        }
      }
    }
    hir.yield at %tf offset %1
  }

  hir.for %i : i32 = %0 : !hir.const to %16 : !hir.const 
  step %1:!hir.const iter_time(%ti = %t offset %1 ){
    %tf=hir.for %j : i32 = %0 : !hir.const to %16 : !hir.const 
    step %1:!hir.const iter_time(%tj = %ti offset %1 ){
      hir.yield at %tj offset %1
      %b1 = hir.gt(%i,%1) : (i32,!hir.const) -> (i1)
      %b2 = hir.gt(%j,%1) : (i32,!hir.const) -> (i1)
      %b3 = hir.and(%b1,%b2) : (i1,i1) -> (i1)
      hir.if(%b3) at %tj{
        %v = hir.call @weighted_average(%wndw) at %tj offset %2
          :(!hir.memref<3*3*i32, packing=[], r>) -> (i32 delay 2)
        %i4 = hir.delay %i by %4 at %tj : i32 -> i32
        %j4 = hir.delay %j by %4 at %tj : i32 -> i32
        hir.mem_write %v to %B[%i4,%j4] at %tj offset %4
          :(i32, !hir.memref<16*16*i32, w> [i32,i32])
      }      
    }
    hir.yield at %tf offset %1
  }
  hir.return
}

