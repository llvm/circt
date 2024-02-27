// RUN: circt-opt %s --convert-hw-to-btor2 -o tmp.mlir | FileCheck %s  

module {
  //CHECK:    [[NID0:[0-9]+]] sort bitvec 1
  //CHECK:    [[NID1:[0-9]+]] input [[NID0]] reset
  hw.module @test(in %clock : !seq.clock, in %reset : i1) {
    %0 = seq.from_clock %clock 
    //CHECK:    [[NID3:[0-9]+]] constd [[NID0]] 0
    %false = hw.constant false
    //CHECK:    [[NID2:[0-9]+]] state [[NID0]] reg
    //CHECK:    [[NID9:[0-9]+]] init [[NID0]] [[NID2]] [[NID3]]
    %reg = seq.compreg %false, %clock reset %reset, %false powerOn %false : i1  

    //CHECK:    [[NID4:[0-9]+]] eq [[NID0]] [[NID2]] [[NID3]]
    %10 = comb.icmp bin eq %reg, %false : i1

    sv.always posedge %0 {
        //CHECK:    [[NID5:[0-9]+]] not [[NID0]] [[NID4]]
        //CHECK:    [[NID6:[0-9]+]] bad [[NID5]]
        sv.assert %10, immediate
    }
    //CHECK:    [[NID7:[0-9]+]] ite [[NID0]] [[NID1]] [[NID3]] [[NID3]]
    //CHECK:    [[NID8:[0-9]+]] next [[NID0]] [[NID2]] [[NID7]]
  }

}
