// RUN: circt-opt %s
#bram_r = {"rd_latency"=1}
#reg_r = {"rd_latency"=0}
#bram_w = {"wr_latency"=1}
#reg_w = {"wr_latency"=1}

hir.func.extern @i32mult_dsp48 at %t (%a: i32, %b: i32) ->(%p: i32 delay 2)

hir.func @hir_convolution at %t(
%img :!hir.memref<8x8xi32> ports [#bram_r],
%kernel :!hir.memref<2x2xi32> ports [#bram_r],
%output : !hir.memref<8x8xi32>ports [#bram_w]){

  %c0_i4 = hw.constant  0:i4
  %c7_i4 = hw.constant  7:i4
  %c1_i4 = hw.constant  1:i4
  %c0_i2 = hw.constant  0:i2
  %c1_i2 = hw.constant  1:i2
  %c2_i2 = hw.constant  2:i2
  %c0_i32 = hw.constant  0:i32
  %0 = arith.constant  0:index
  %4 = arith.constant  4:index
  %2 = arith.constant  2:index
  %3 = arith.constant  3:index

  %true = hw.constant true
  //hir.if %true at time(%ti=%t) ->(i1){
  //  %tt = hw.constant 1:i1
  //  hir.yield (%tt):(i1)
  //}else{
  //  %ff = hw.constant 0:i1
  //  hir.yield (%ff): (i1)
  //}
  //hir.while %true iter_time(%ti = %t + 1 ){
  //  hir.probe %ti name "ti":!hir.time
  //  hir.next_iter condition %true at %ti + 10
  //}

  //hir.for %i : i4 = %c0_i4  to %c7_i4  step %c1_i4 iter_time(%ti = %t + 1 ){
  //  hir.probe %i name "i":i4
  //  hir.probe %ti name "ti":!hir.time
  //  %tj_end=hir.for %j : i4 =%c0_i4  to %c7_i4  step %c1_i4 iter_time(%tj = %ti + 1 ){
  //    hir.probe %j name "j":i4
  //    hir.probe %tj name "tj":!hir.time
  //    %ti1_end=hir.for %i1 : i2 = %c0_i2 to %c2_i2 step %c1_i2 iter_time(%ti1 = %tj + 1 ){
  //      hir.probe %i1 name "i1":i2
  //      hir.probe %ti1 name "ti1":!hir.time
  //      %tj1_end=hir.for %j1 : i2 = %c0_i2 to %c2_i2 step %c1_i2 iter_time(%tj1 = %ti1 + 1 ){
  //        %is_first_j1_iter = hir.is_first_iter:i1
  //        hir.probe %j1 name "j1":i2
  //        hir.probe %tj1 name "tj1":!hir.time
  //        hir.next_iter at %tj1 + 1
  //      }
  //      hir.probe %tj1_end name "tj1_end":!hir.time
  //      hir.next_iter at %tj1_end + 10
  //    }
  //    hir.probe %ti1_end name "ti1_end":!hir.time
  //    hir.next_iter at %ti1_end + 1
  //  }
  //  hir.probe %tj_end name "tj_end":!hir.time
  //  hir.next_iter at %tj_end + 1
  //}

  //hir.comment "debug end"

  %val = hir.alloca("reg") :!hir.memref<(bank 1)xi32> ports [#reg_r,#reg_w]

    //Read from input. Update line buffer. Input values to each row of window.
  %ti_end=hir.for %i : i4 = %c0_i4  to %c7_i4  step %c1_i4 iter_time(%ti = %t + 1 ){
    hir.probe %i name "i":i4
    hir.probe %ti name "ti":!hir.time
    %tj_end=hir.for %j : i4 =%c0_i4  to %c7_i4  step %c1_i4 iter_time(%tj = %ti + 1 ){

      hir.store %c0_i32 to %val[port 1][%0] at %tj + 3
      : !hir.memref<(bank 1)xi32>

      %ti1_end=hir.for %i1 : i2 = %c0_i2 to %c2_i2 step %c1_i2 iter_time(%ti1 = %tj + 1 ){
        hir.probe %i1 name "i1":i2
        hir.probe %ti1 name "ti1":!hir.time
        %tj1_end=hir.for %j1 : i2 = %c0_i2 to %c2_i2 step %c1_i2 iter_time(%tj1 = %ti1 + 1 ){
          %i1_i4 = comb.concat %c0_i2,%i1 : i2,i2
          %idx1_i4 = comb.add %i,%i1_i4  : i4
          %j1_i4 = comb.concat %c0_i2,%j1 : i2,i2
          %idx2_i4 = comb.add %j,%j1_i4  : i4
          %idx1 = comb.extract %idx1_i4 from 0:(i4)->(i3)
          %idx2 = comb.extract %idx2_i4 from 0:(i4)->(i3)
          hir.probe %idx1_i4 name "idx1_i4":i4
          hir.probe %idx2_i4 name "idx2_i4":i4
          hir.probe %tj1 name "tj1":!hir.time
          %v1 = hir.load %img[port 0][%idx1,%idx2] at %tj1  
          : !hir.memref<8x8xi32>
          %i1_i1 = comb.extract %i1 from 0:(i2)->(i1)
          %j1_i1 = comb.extract %j1 from 0:(i2)->(i1)
          %v2 = hir.load %kernel[port 0][%i1_i1,%j1_i1] at %tj1  
          : !hir.memref<2x2xi32>

          %mul = hir.call "i32mult_dsp48_inst" @i32mult_dsp48(%v1,%v2) at %tj1+1:
          !hir.func<(i32,i32) -> (i32 delay 2)>

          %v3 = hir.load %val[port 0][%0] at %tj1 + 3
          : !hir.memref<(bank 1)xi32>

          %res = comb.add %mul,%v3 :i32

          hir.store %res to %val[port 1][%0] at %tj1+3
          : !hir.memref<(bank 1)xi32>

          hir.next_iter at %tj1 + 1
        }
        hir.next_iter at %tj1_end + 1
      }

      %v = hir.load %val[port 0][%0] at %ti1_end + 3
      : !hir.memref<(bank 1)xi32>
      %i3 = hir.delay %i by 3 at %ti1_end : i4 
      %j3 = hir.delay %j by 3 at %ti1_end : i4 
      %i3_i3 = comb.extract %i3 from 0:(i4)->(i3)
      %j3_i3 = comb.extract %j3 from 0:(i4)->(i3)
      hir.store %v to %output[port 0][%i3_i3,%j3_i3] at %ti1_end+3
      : !hir.memref<8x8xi32>
      hir.next_iter at %ti1_end + 1
    }
    hir.next_iter at %tj_end + 1
  }
  hir.return
}

