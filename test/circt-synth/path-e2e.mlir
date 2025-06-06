// RUN: circt-synth %s -output-longest-path=%t -top counter && cat %t | FileCheck %s
// CHECK-LABEL: # Analysis result for "counter"
// CHECK-NEXT: Found 135 closed paths
// CHECK-NEXT: Maximum path delay: 75 
hw.module @counter(in %a: i16, in %clk: !seq.clock, out result: i16) {
    %reg = seq.compreg %add, %clk : i16
    %add = comb.mul %reg, %a : i16
    hw.output %reg : i16
}
