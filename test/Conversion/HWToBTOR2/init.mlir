// RUN: circt-opt %s --convert-hw-to-btor2 -o %t | FileCheck %s  

module {
  //CHECK:    [[NID0:[0-9]+]] sort bitvec 1
  //CHECK:    [[NID1:[0-9]+]] input [[NID0]] reset
  hw.module @test(in %clock : !seq.clock, in %reset : i1) {
    %0 = seq.from_clock %clock 
    //CHECK:    [[NID3:[0-9]+]] constd [[NID0]] 0
    // Register states get pregenerated
    //CHECK:    [[NID2:[0-9]+]] state [[NID0]] reg
    %false = hw.constant false
    //CHECK:    [[INIT:[0-9]+]] init [[NID0]] [[NID2]] [[NID3]]
    %init = seq.initial() {
      %false_0 = hw.constant false
      seq.yield %false_0 : i1
    } : () -> !seq.immutable<i1>
    //CHECK:    [[RESET:[0-9]+]] constd [[NID0]] 0
    %reg = seq.compreg %false, %clock reset %reset, %false initial %init : i1

    //CHECK:    [[NID4:[0-9]+]] eq [[NID0]] [[NID2]] [[RESET]]
    %10 = comb.icmp bin eq %reg, %false : i1

    sv.always posedge %0 {
        //CHECK:    [[NID5:[0-9]+]] not [[NID0]] [[NID4]]
        //CHECK:    [[NID6:[0-9]+]] bad [[NID5]]
        sv.assert %10, immediate
    }
    //CHECK:    [[NID7:[0-9]+]] ite [[NID0]] [[NID1]] [[RESET]] [[RESET]]
    //CHECK:    [[NID8:[0-9]+]] next [[NID0]] [[NID2]] [[NID7]]
  }

}
