#s_axis = {"split"="s_axis_split"}
#r_bram = {"rd"=3}
#w_bram = {"wr"=1}
#r_reg = {"rd"=0}
#w_reg = {"wr"=1}
hir.func @testBus at %t(%a:i32 delay 2,
%b :!hir.memref<16x16x4x4xi32,[0,1], #r_bram>, 
%c : f32 delay 5){

  //%br_addr_send, %br_addr_recv= hir.alloca("bus") 
  //: tensor<4x!hir.bus<wr i1, wr i4, proto valid>>
  //  tensor<4x!hir.bus<rd i1, rd i4>>
    
  //%br_data_send, %br_data_recv= hir.alloca("bus") 
  //: tensor<4x!hir.bus<wr i32>>
  //  tensor<4x!hir.bus<rd i32>>

  //%bw_send, %bw_recv= hir.alloca("bus") 
  //: tensor<4x!hir.bus<wr i1, wr tuple <i4,  i32>, proto valid>>
  //  tensor<4x!hir.bus<rd i1, rd tuple <i4,  i32>>>

  //hir.call @bram(%br_addr_recv,%br_data_send,%bw_recv)
  //{"params"={"WIDTH"=32,"DEPTH"=16,"BANKS"=4}}

  
  %0 = constant   0  :index
  %1 = constant   1  :index
  %2 = constant   2  :index
  %3 = constant   3  :index
  %i1 = constant  11 :index
  %i1_i4 = constant 11  :i4
  %i2 = constant  12 :index
  %j1 = constant  2  :index
  %j1_i4 = constant  2  :i4
  %j2 = constant  3  :index

  hir.call @testBus(%a,%b,%c) at %t
    :!hir.func<(i32 delay 2, !hir.memref<16x16x4x4xi32,[0,1], #r_bram>
    , f32 delay 5) -> ()>

  %mr, %mw= hir.alloca("bram") 
  :!hir.memref<16x16x4x4xi32,[0,1], #r_bram>,
  !hir.memref<16x16x4x4xi32,[0,1], #w_bram>

  %x = hir.alloca("bus"):!hir.bus<rd i32>
  %y = hir.alloca("bus"):!hir.bus<rd i1, wr i1, wr f32, proto #s_axis>
  %z = hir.alloca("bus"):tensor<1x!hir.bus<wr f32>>
  //only tensor<bus> is allowed at the moment.
  //%u = hir.alloca("bus"):tuple<!hir.bus<wr i32, rd i1>, !hir.bus<i32>>

  %b1 = hir.split %x : !hir.bus<rd i32> -> !hir.bus<rd i32>
  %b2 = hir.split %z[%0]: tensor<1x!hir.bus<wr f32>> -> !hir.bus<wr f32>

  //hir.send %1 to %br_addr_send[%j][%0] at %t 
  //hir.send %i to %br_addr_send[%j][%1] at %t 
  //%v = hir.recv %br_data_recv[%j] at %t+1

  %v =  hir.load %mr[%i1_i4,%j1_i4, %i2,%j2] at %t 
  : !hir.memref<16x16x4x4xi32,[0,1],#r_bram>

  //hir.send %1 to %bw_send[%j][%0]
  //%iv = hir.tuple (%i,%v)
  //hir.send %iv to %bw_send[%j][%1]
  hir.store %v to %mw[%i1_i4,%j1_i4, %i2,%j2] at %t+%3
  : !hir.memref<16x16x4x4xi32,[0,1],#w_bram>


  %regr, %regw= hir.alloca("reg") 
  :!hir.memref<6xi32,[0], #r_reg>,
  !hir.memref<6xi32,[0], #w_reg>
  
  %v2 =  hir.load %regr[%i1] at %t 
  : !hir.memref<6xi32,[0],#r_reg>

  hir.store %v2 to %regw[%j1] at %t
  : !hir.memref<6xi32,[0],#w_reg>

  %f1 = index_cast %1 : index to i32
  %f2 = constant 2:i32

  %f4 = addi %i1,%i2: index
  %f5 = index_cast %f4 : index to i32

  //
  hir.store %f5 to %mw[%i1_i4,%j1_i4, %i2,%j2] at %t+%3
  : !hir.memref<16x16x4x4xi32,[0,1],#w_bram>

  hir.return
}
