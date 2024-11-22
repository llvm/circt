// RUN: circt-opt %s --convert-hw-to-btor2 -o %t | FileCheck %s  

module {
  // CHECK:   [[NID0:[0-9]+]] sort bitvec 32
  // CHECK:   [[NID1:[0-9]+]] input [[NID0]] a
  hw.module @inc(in %a : i32, in %clk : !seq.clock, out pred : i1) {
    %0 = seq.from_clock %clk

    // CHECK:   [[BIGSORT:[0-9]+]] sort bitvec 100
    // CHECK:   [[BIGCONST:[0-9]+]] constd [[BIGSORT]] 111111111111111111111111111
    %bigConst = hw.constant 111111111111111111111111111 : i100

    // CHECK:   [[NID2:[0-9]+]] constd [[NID0]] 0
    %c0_i32 = hw.constant 0 : i32

    // CHECK:   [[NID3:[0-9]+]] sort bitvec 1
    // CHECK:   [[NID4:[0-9]+]] constd [[NID3]] 0
    %false = hw.constant false

    // CHECK:   [[NID5:[0-9]+]] constd [[NID3]] -1
    %true = hw.constant true

    // CHECK:   [[NID6:[0-9]+]] sort bitvec 33
    // CHECK:   [[NID7:[0-9]+]] concat [[NID6]] [[NID4]] [[NID1]]
    %1 = comb.concat %false, %a : i1, i32

    // CHECK:   [[NID8:[0-9]+]] constd [[NID6]] 1
    %c1_i33 = hw.constant 1 : i33

    // CHECK:   [[NID9:[0-9]+]] add [[NID6]] [[NID7]] [[NID8]]
    %2 = comb.add bin %1, %c1_i33 : i33

    // CHECK:   [[NID10:[0-9]+]] slice [[NID0]] [[NID9]] 31 0
    %3 = comb.extract %2 from 0 : (i33) -> i32

    // CHECK:   [[NID11:[0-9]+]] slice [[NID3]] [[NID9]] 16 16
    %4 = comb.extract %2 from 16 : (i33) -> i1

    // CHECK:   [[NID12:[0-9]+]] ugt [[NID3]] [[NID10]] 2
    %5 = comb.icmp bin ugt %3, %a : i32

    // CHECK:   [[NID13:[0-9]+]] implies [[NID3]] [[NID5]] [[NID12]]
    // CHECK:   [[NID14:[0-9]+]] not [[NID3]] [[NID13]]
    // CHECK:   [[NID15:[0-9]+]] bad [[NID14:[0-9]+]]
    sv.always posedge %0 {
      sv.if %true {
        sv.assert %5, immediate message "a + 1 should be greater than a"
      }
    }
    hw.output %5 : i1
  }
}

