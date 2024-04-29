// RUN: circt-opt %s --convert-hw-to-btor2 -o tmp.mlir | FileCheck %s  

module {
  //CHECK:    [[NID0:[0-9]+]] sort bitvec 1
  //CHECK:    [[NID1:[0-9]+]] input [[NID0]] reset
  hw.module @Counter(in %clock : !seq.clock, in %reset : i1) {
    //CHECK:    [[NID2:[0-9]+]] sort bitvec 32
    //CHECK:    [[NID3:[0-9]+]] state [[NID2]] count
    //CHECK:    [[NID4:[0-9]+]] constd [[NID2]] 43
    %c43_i32 = hw.constant 43 : i32
    //CHECK:    [[NID5:[0-9]+]] constd [[NID2]] 1
    %c1_i32 = hw.constant 1 : i32
    //CHECK:    [[NID6:[0-9]+]] constd [[NID2]] 42
    %c42_i32 = hw.constant 42 : i32
    //CHECK:    [[NID7:[0-9]+]] constd [[NID0]] -1
    %true = hw.constant true
    //CHECK:    [[NID8:[0-9]+]] constd [[NID2]] 0
    %c0_i32 = hw.constant 0 : i32
    %0 = seq.from_clock %clock
    %count = seq.firreg %3 clock %clock reset sync %reset, %c0_i32 {firrtl.random_init_start = 0 : ui64} : i32
    //CHECK:    [[NID9:[0-9]+]] eq [[NID0]] [[NID3]] [[NID6]]
    %1 = comb.icmp bin eq %count, %c42_i32 : i32
    //CHECK:    [[NID10:[0-9]+]] add [[NID2]] [[NID3]] [[NID5]]
    %2 = comb.add %count, %c1_i32 {sv.namehint = "_count_T"} : i32
    //CHECK:    [[NID11:[0-9]+]] ite [[NID2]] [[NID9]] [[NID8]] [[NID10]]
    %3 = comb.mux bin %1, %c0_i32, %2 : i32
    //CHECK:    [[NID12:[0-9]+]] ult [[NID0]] [[NID3]] [[NID4]]
    %4 = comb.icmp bin ult %count, %c43_i32 : i32
    //CHECK:    [[NID13:[0-9]+]] xor [[NID0]] [[NID1]] [[NID7]]
    %5 = comb.xor bin %reset, %true : i1

    //CHECK:    [[NID14:[0-9]+]] implies [[NID0]] [[NID13]] [[NID12]]
    //CHECK:    [[NID15:[0-9]+]] not [[NID0]] [[NID14]]
    //CHECK:    [[NID16:[0-9]+]] bad [[NID15]] 
    sv.always posedge %0 {
      sv.if %5 {
        sv.assert %4, immediate label "assert__assert"
      }
    }
    //CHECK:    [[NID17:[0-9]+]] ite [[NID2]] [[NID1]] [[NID8]] [[NID11]]
    //CHECK:    [[NID18:[0-9]+]] next [[NID2]] [[NID3]] [[NID17]]
    hw.output
  }
  om.class @Counter_Class(%basepath: !om.basepath) {
  }
}
