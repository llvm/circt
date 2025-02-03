// RUN: circt-opt %s --convert-hw-to-btor2 -o %t | FileCheck %s  

module {
  //CHECK:    [[NID0:[0-9]+]] sort bitvec 1
  //CHECK:    [[NID1:[0-9]+]] input [[NID0]] reset0
  //CHECK:    [[NID2:[0-9]+]] input [[NID0]] reset1
  //CHECK:    [[NID3:[0-9]+]] sort bitvec 32
  //CHECK:    [[NID4:[0-9]+]] input [[NID3]] in
  hw.module @MultipleResets(in %clock : !seq.clock, in %reset0 : i1, in %reset1 : i1, in %in : i32) {
    // Registers are all emitted before any other operation
    //CHECK:    [[NID5:[0-9]+]] state [[NID3]] reg0
    //CHECK:    [[NID6:[0-9]+]] state [[NID3]] reg1
    //CHECK:    [[NID7:[0-9]+]] state [[NID3]] reg2

    //CHECK:    [[NID8:[0-9]+]] constd [[NID3]] 0
    %c0_i32 = hw.constant 0 : i32

    %reg0 = seq.compreg %in, %clock reset %reset0, %c0_i32 : i32
    %reg1 = seq.compreg %in, %clock reset %reset1, %c0_i32 : i32

    //CHECK:    [[NID9:[0-9]+]] and [[NID0]] [[NID1]] [[NID2]]
    %reset_and = comb.and %reset0, %reset1 : i1

    %reg2 = seq.compreg %in, %clock reset %reset_and, %c0_i32 : i32

    // Register reset ITEs and next statements are emitted last
    //CHECK:    [[NID10:[0-9]+]] ite [[NID3]] [[NID1]] [[NID8]] [[NID4]]
    //CHECK:    [[NID11:[0-9]+]] next [[NID3]] [[NID5]] [[NID10]]
    //CHECK:    [[NID12:[0-9]+]] ite [[NID3]] [[NID2]] [[NID8]] [[NID4]]
    //CHECK:    [[NID13:[0-9]+]] next [[NID3]] [[NID6]] [[NID12]]
    //CHECK:    [[NID14:[0-9]+]] ite [[NID3]] [[NID9]] [[NID8]] [[NID4]]
    //CHECK:    [[NID15:[0-9]+]] next [[NID3]] [[NID7]] [[NID14]]

    hw.output
  }
}
