#r_bram = {"rd"=1}
#r_reg = {"rd"=1}
#w_reg = {"wr"=1}
#w_bram = {"wr"=1}
// RUN: circt-opt %s
hir.func @convolution at %t(
%A :!hir.memref<16x16xi32, #r_bram>,
%B :!hir.memref<16x16xi32, #w_bram>){

  %buff,%buffW = hir.alloca("bram") :!hir.memref<3x16xi32,[1], #r_bram>,
  !hir.memref<3x16xi32,[1], #w_bram>
  %wndw,%wndwW = hir.alloca("bram") :!hir.memref<3x3xi32,[0,1], #r_reg>,
  !hir.memref<3x3xi32,[0,1], #w_reg>

  %0 = hir.constant (0) :!hir.const 
  %1 = hir.constant (1) :!hir.const
  %2 = hir.constant (2) :!hir.const
  %3 = hir.constant (3) :!hir.const
  %4 = hir.constant (4) :!hir.const
  %16 = hir.constant (16) : !hir.const

  //Read from input. Update line buffer. Input values to each row of window.
  hir.for %i : i32 = %0 : !hir.const to %16 : !hir.const 
  step %1:!hir.const iter_time(%ti = %t + %1 ){
    %tf=hir.for %j : i32 = %0 : !hir.const to %16 : !hir.const 
    step %1:!hir.const iter_time(%tj = %ti + %1 ){
      hir.yield at %tj + %1
      %v =  hir.load %A[%i, %j] at %tj 
      : !hir.memref<16x16xi32, #r_bram>[i32,i32] -> i32
      %v0 = hir.load %buff[%0,%j] at %tj
      : !hir.memref<3x16xi32,[1], #r_bram>[!hir.const,i32] -> i32
      %v1 = hir.load %buff[%1,%j] at %tj
      : !hir.memref<3x16xi32,[1], #r_bram>[!hir.const,i32] -> i32
      %v2 = hir.load %buff[%2,%j] at %tj
      : !hir.memref<3x16xi32,[1], #r_bram>[!hir.const,i32] -> i32

      %j1 = hir.delay %j by %1 at %tj :i32 -> i32
      hir.store %v1 to %buffW[%0,%j1] at %tj + %1 
      : (i32, !hir.memref<3x16xi32, [1], #w_bram>[!hir.const,i32])
      hir.store %v2 to %buffW[%1,%j1] at %tj + %1 
      : (i32, !hir.memref<3x16xi32, [1], #w_bram>[!hir.const,i32])
      hir.store %v to %buffW[%2,%j1] at %tj + %1 
      : (i32, !hir.memref<3x16xi32, [1], #w_bram>[!hir.const,i32])

      hir.store %v1 to %wndwW[%0,%0] at %tj + %1
      : (i32, !hir.memref<3x3xi32, [0,1], #w_reg>[!hir.const,!hir.const])
      hir.store %v2 to %wndwW[%1,%0] at %tj + %1
      : (i32, !hir.memref<3x3xi32, [0,1], #w_reg>[!hir.const,!hir.const])
      hir.store %v to %wndwW[%2,%0] at %tj + %1
      : (i32, !hir.memref<3x3xi32, [0,1], #w_reg>[!hir.const,!hir.const])
    }
    hir.yield at %tf + %1
  }

  hir.for %i : i32 = %0 : !hir.const to %16 : !hir.const 
  step %1:!hir.const iter_time(%ti = %t + %1 ){
    %tf=hir.for %j : i32 = %0 : !hir.const to %16 : !hir.const 
    step %1:!hir.const iter_time(%tj = %ti + %1 ){
      hir.yield at %tj + %1
      hir.unroll_for %k1 = 0 to 3 step 1 iter_time(%tk1 = %tj){
        hir.yield at %tk1 
        hir.unroll_for %k2 = 0 to 2 step 1 iter_time(%tk2 = %tk1){
          hir.yield at %tk2
          %v = hir.load %wndw[%k1,%k2] at %tk2 + %1
          : !hir.memref<3x3xi32, [0,1], #r_reg>[!hir.const,!hir.const] ->
          i32
          %k2Plus1 = hir.add(%k2,%1) :(!hir.const,!hir.const) -> (!hir.const)
          hir.store %v to %wndwW[%k1,%k2Plus1] at %tk2 + %1
            :(i32, !hir.memref<3x3xi32, [0,1], #w_reg> [!hir.const,!hir.const])
        }
      }
    }
    hir.yield at %tf + %1
  }

  hir.for %i : i32 = %0 : !hir.const to %16 : !hir.const 
  step %1:!hir.const iter_time(%ti = %t + %1 ){
    %tf=hir.for %j : i32 = %0 : !hir.const to %16 : !hir.const 
    step %1:!hir.const iter_time(%tj = %ti + %1 ){
      hir.yield at %tj + %1
      %b1 = hir.gt(%i,%1) : (i32,!hir.const) -> (i1)
      %b2 = hir.gt(%j,%1) : (i32,!hir.const) -> (i1)
      %b3 = hir.and(%b1,%b2) : (i1,i1) -> (i1)
      hir.if(%b3) at %tj{
        %v = hir.call @weighted_average(%wndw) at %tj + %2
          :!hir.func<(!hir.memref<3x3xi32, [0,1], #r_reg>) -> (i32 delay 2)>
        %i4 = hir.delay %i by %4 at %tj : i32 -> i32
        %j4 = hir.delay %j by %4 at %tj : i32 -> i32
        hir.store %v to %B[%i4,%j4] at %tj + %4
          :(i32, !hir.memref<16x16xi32, #w_bram> [i32,i32])
      }      
    }
    hir.yield at %tf + %1
  }
  hir.return
}

