// RUN: circt-opt %s --convert-hw-to-btor2 -o %t | FileCheck %s  

module {

  // CHECK:   [[NID0:[0-9]+]] sort bitvec 32
  // CHECK:   [[NID1:[0-9]+]] input [[NID0]] a
  hw.module @foo(in %a : i32) {
    
    // CHECK:   [[NID2:[0-9]+]] constd [[NID0]] 0
    // CHECK:   [[NID3:[0-9]+]] sort bitvec 1
    // CHECK:   [[NID4:[0-9]+]] ugt [[NID3]] [[NID1]] [[NID2]]
    // CHECK:   [[NID5:[0-9]+]] not [[NID3]] [[NID4]]
    // CHECK:   [[NID6:[0-9]+]] bad [[NID5]] 
    %fstr = verif.format_verilog_string "Hi %x\0A" (%a) : i32
    verif.print %fstr
    
    %c0_i32 = hw.constant 0 : i32
    %cond = comb.icmp bin ugt %a, %c0_i32 : i32
    verif.assert %cond : i1
  }

}
