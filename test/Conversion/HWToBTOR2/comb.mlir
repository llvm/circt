// RUN: circt-opt %s --convert-hw-to-btor2 -o tmp.mlir | FileCheck %s  

module {
  // CHECK:   1 sort bitvec 32
  // CHECK:   2 input 1 a
  hw.module @inc(in %a : i32, in %clk : !seq.clock, out pred : i1) {
    %0 = seq.from_clock %clk

    // CHECK:   3 constd 1 0
    %c0_i32 = hw.constant 0 : i32

    // CHECK:   4 sort bitvec 1
    // CHECK:   5 constd 4 0
    %false = hw.constant false

    // CHECK:   6 constd 4 -1
    %true = hw.constant true
    %.pred.output = hw.wire %4  : i1
    %b = hw.wire %3  : i32

    // CHECK:   7 sort bitvec 33
    // CHECK:   8 concat 7 5 2
    %1 = comb.concat %false, %a : i1, i32

    // CHECK:   9 constd 7 1
    %c1_i33 = hw.constant 1 : i33

    // CHECK:   10 add 7 8 9
    %2 = comb.add bin %1, %c1_i33 : i33

    // CHECK:   11 slice 1 10 31 0
    %3 = comb.extract %2 from 0 : (i33) -> i32

    // CHECK:   12 ugt 4 11 2
    %4 = comb.icmp bin ugt %b, %a : i32

    // CHECK:   13 implies 4 6 12
    // CHECK:   14 not 4 13
    // CHECK:   15 bad 14
    sv.always posedge %0 {
      sv.if %true {
        sv.assert %.pred.output, immediate message "a + 1 should be greater than a"
      }
    }
    hw.output %.pred.output : i1
  }
}

