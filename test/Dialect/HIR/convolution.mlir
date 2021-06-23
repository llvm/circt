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

  %0 =  constant 0 :index  
  %1 =  constant 1 :index
  %2 =  constant 2 :index
  %3 =  constant 3 :index
  %4 =  constant 4 :index
  %16 = constant 16:index

  //Read from input. Update line buffer. Input values to each row of window.
  hir.for %i : i4 = %0 : index to %16 : index 
  step %1:index iter_time(%ti = %t + %1 ){
    %tf=hir.for %j : i4 = %0 : index to %16 : index 
    step %1:index iter_time(%tj = %ti + %1 ){
      hir.yield at %tj + %1
      %v =  hir.load %A[%i, %j] at %tj 
      : !hir.memref<16x16xi32, #r_bram>
      %v0 = hir.load %buff[%0,%j] at %tj
      : !hir.memref<3x16xi32,[1], #r_bram>
      %v1 = hir.load %buff[%1,%j] at %tj
      : !hir.memref<3x16xi32,[1], #r_bram>
      %v2 = hir.load %buff[%2,%j] at %tj
      : !hir.memref<3x16xi32,[1], #r_bram>

      %j1 = hir.delay %j by %1 at %tj :i4 -> i4
      hir.store %v1 to %buffW[%0,%j1] at %tj + %1 
      : !hir.memref<3x16xi32, [1], #w_bram>
      hir.store %v2 to %buffW[%1,%j1] at %tj + %1 
      : !hir.memref<3x16xi32, [1], #w_bram>
      hir.store %v to %buffW[%2,%j1] at %tj + %1 
      : !hir.memref<3x16xi32, [1], #w_bram>

      hir.store %v1 to %wndwW[%0,%0] at %tj + %1
      : !hir.memref<3x3xi32, [0,1], #w_reg>
      hir.store %v2 to %wndwW[%1,%0] at %tj + %1
      : !hir.memref<3x3xi32, [0,1], #w_reg>
      hir.store %v to %wndwW[%2,%0] at %tj + %1
      : !hir.memref<3x3xi32, [0,1], #w_reg>
    }
    hir.yield at %tf + %1
  }

  hir.for %i : i32 = %0 : index to %16 : index 
  step %1:index iter_time(%ti = %t + %1 ){
    %tf=hir.for %j : i32 = %0 : index to %16 : index 
    step %1:index iter_time(%tj = %ti + %1 ){
      hir.yield at %tj + %1
      hir.unroll_for %k1 = 0 to 3 step 1 iter_time(%tk1 = %tj){
        hir.yield at %tk1 
        hir.unroll_for %k2 = 0 to 2 step 1 iter_time(%tk2 = %tk1){
          hir.yield at %tk2
          %v = hir.load %wndw[%k1,%k2] at %tk2 + %1
          : !hir.memref<3x3xi32, [0,1], #r_reg>
          %k2Plus1 = addi %k2, %1 : index
          hir.store %v to %wndwW[%k1,%k2Plus1] at %tk2 + %1
            :!hir.memref<3x3xi32, [0,1], #w_reg>
        }
      }
    }
    hir.yield at %tf + %1
  }

  hir.for %i : i4 = %0 : index to %16 : index 
  step %1:index iter_time(%ti = %t + %1 ){
    %tf=hir.for %j : i4 = %0 : index to %16 : index 
    step %1:index iter_time(%tj = %ti + %1 ){
      hir.yield at %tj + %1
      %c1_i4 = constant 1:i4
      %b1 = cmpi "ugt", %i, %c1_i4 : i4
      %b2 = cmpi "ugt", %j, %c1_i4 : i4
      %b3 = and %b1, %b2  : i1

      hir.if(%b3) at %tj{
        %v = hir.call @weighted_average(%wndw) at %tj + %2
          :!hir.func<(!hir.memref<3x3xi32, [0,1], #r_reg>) -> (i32 delay 2)>
        %i4 = hir.delay %i by %4 at %tj : i4 -> i4
        %j4 = hir.delay %j by %4 at %tj : i4 -> i4
        hir.store %v to %B[%i4,%j4] at %tj + %4
          :!hir.memref<16x16xi32, #w_bram>
      }      
    }
    hir.yield at %tf + %1
  }
  hir.return
}

