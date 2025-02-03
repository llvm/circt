// RUN: circt-opt %s --convert-hw-to-btor2 -o %t | FileCheck %s  

module {
  // CHECK:   [[NID0:[0-9]+]] sort bitvec 32
  // CHECK:   [[NID1:[0-9]+]] input [[NID0]] a  
  // CHECK:   [[NID2:[0-9]+]] sort bitvec 1  
  // CHECK:   [[NID3:[0-9]+]] constd [[NID2]] 0
  // CHECK:   [[NID4:[0-9]+]] sort bitvec 33
  // CHECK:   [[NID5:[0-9]+]] concat [[NID4]] [[NID3]] [[NID1]]
  // CHECK:   [[NID6:[0-9]+]] constd [[NID4]] 1
  // CHECK:   [[NID7:[0-9]+]] add [[NID4]] [[NID5]] [[NID6]]
  // CHECK:   [[NID8:[0-9]+]] slice [[NID0]] [[NID7]] 31 0
  // CHECK:   [[NID9:[0-9]+]] ugt [[NID2]] [[NID8]] [[NID1]]
  // CHECK:   [[NID10:[0-9]+]] not [[NID2]] [[NID9]]
  // CHECK:   [[NID11:[0-9]+]] bad [[NID10]]
  // CHECK:   [[NID13:[0-9]+]] constd [[NID0]] 0
  hw.module @inc(in %a : i32, in %clk : !seq.clock, out pred : i1) {
    %0 = seq.from_clock %clk
    sv.always posedge %0 {
      sv.assert %4, immediate message "a + 1 should be greater than a"
    }
    %4 = comb.icmp ugt %3, %a : i32
    %3 = comb.extract %2 from 0 : (i33) -> i32
    %2 = comb.add %1, %c1_i33 : i33
    %c1_i33 = hw.constant 1 : i33
    %1 = comb.concat %false, %a : i1, i32
    %false = hw.constant false
    %c0_i32 = hw.constant 0 : i32
    hw.output %4 : i1
  }
}

