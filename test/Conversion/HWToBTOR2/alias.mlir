// RUN: circt-opt %s --convert-hw-to-btor2 -o tmp.mlir | FileCheck %s  

module {
  //CHECK:  [[NID0:[0-9]+]] sort bitvec 1
  //CHECK:  [[NID1:[0-9]+]] input [[NID0]] reset
  //CHECK:  [[NID2:[0-9]+]] input [[NID0]] in
  hw.module @Alias(in %clock : !seq.clock, in %reset : i1, in %in : i1, out out : i32) {
  //CHECK:  [[NID3:[0-9]+]] sort bitvec 32
  //CHECK:  [[NID4:[0-9]+]] state [[NID3]] count

    //CHECK:  [[NID5:[0-9]+]] constd [[NID0]] 0
    %false = hw.constant false
    //CHECK:  [[NID6:[0-9]+]] constd [[NID3]] 0
    %c0_i32 = hw.constant 0 : i32
    //CHECK:  [[NID7:[0-9]+]] constd [[NID0]] -1
    %true = hw.constant true
    //CHECK:  [[NID8:[0-9]+]] sort bitvec 2
    //CHECK:  [[NID9:[0-9]+]] constd [[NID8]] -2
    %c-2_i2 = hw.constant -2 : i2
    %count = seq.compreg %6, %clock reset %reset, %c0_i32 : i32
    %b = hw.wire %2  : i32
    %d = hw.wire %b  : i32
    %c = hw.wire %5  : i32
    //CHECK:  [[NID10:[0-9]+]] sort bitvec 33
    //CHECK:  [[NID11:[0-9]+]] concat [[NID10]] [[NID5]] [[NID4]]
    %0 = comb.concat %false, %count : i1, i32
    //CHECK:  [[NID12:[0-9]+]] constd [[NID10]] 1
    %c1_i33 = hw.constant 1 : i33
    //CHECK:  [[NID13:[0-9]+]] sub [[NID10]] [[NID11]] [[NID12]]
    %1 = comb.sub bin %0, %c1_i33 : i33
    //CHECK:  [[NID14:[0-9]+]] slice [[NID3]] [[NID13]] 31 0
    %2 = comb.extract %1 from 0 : (i33) -> i32
    //CHECK:  [[NID15:[0-9]+]] concat [[NID10]] [[NID5]] [[NID14]]
    %3 = comb.concat %false, %d : i1, i32
    //CHECK:  [[NID16:[0-9]+]] constd [[NID10]] 2
    %c2_i33 = hw.constant 2 : i33
    //CHECK:  [[NID17:[0-9]+]] add [[NID10]] [[NID15]] [[NID16]]
    %4 = comb.add bin %3, %c2_i33 : i33
    //CHECK:  [[NID18:[0-9]+]] slice [[NID3]] [[NID17]] 31 0
    %5 = comb.extract %4 from 0 : (i33) -> i32
    //CHECK:  [[NID19:[0-9]+]] ite [[NID3]] [[NID2]] [[NID14]] [[NID18]]
    //CHECK:  [[NID20:[0-9]+]] ite [[NID3]] [[NID1]] [[NID6]] [[NID19]]
    %6 = comb.mux bin %in, %b, %c : i32
    //CHECK:  [[NID21:[0-9]+]] next [[NID3]] [[NID4]] [[NID20]]
    hw.output %count : i32
  }

}
