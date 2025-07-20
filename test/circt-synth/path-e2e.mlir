// RUN: circt-synth %s -output-longest-path=- -top counter | FileCheck %s
// RUN: circt-synth %s -output-longest-path=- -top counter -output-longest-path-json | FileCheck %s --check-prefix JSON

// CHECK-LABEL: # Longest Path Analysis result for "counter"
// CHECK-NEXT: Found 288 paths
// CHECK-NEXT: Found 32 unique fanout points
// CHECK-NEXT: Maximum path delay: 48
// Don't test detailed reports as they are not stable.

// Make sure json is emitted.
// JSON: {"module_name":"counter","timing_levels":[
hw.module @counter(in %a: i16, in %clk: !seq.clock, out result: i16) {
    %reg = seq.compreg %add, %clk : i16
    %add = comb.mul %reg, %a : i16
    hw.output %reg : i16
}
