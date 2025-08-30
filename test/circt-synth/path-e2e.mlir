// RUN: circt-synth %s -output-longest-path=- -top counter | FileCheck %s --check-prefixes COMMON,AIG
// RUN: circt-synth %s -output-longest-path=- -top counter -lower-to-k-lut 6 | FileCheck %s --check-prefixes COMMON,LUT6
// RUN: circt-synth %s -output-longest-path=- -top counter -output-longest-path-json | FileCheck %s --check-prefix JSON

// COMMON-LABEL: # Longest Path Analysis result for "counter"
// COMMON-NEXT: Found 288 paths
// COMMON-NEXT: Found 32 unique fanout points
// AIG-NEXT: Maximum path delay: 42
// LUT6-NEXT: Maximum path delay: 7
// Don't test detailed reports as they are not stable.

// Make sure json is emitted.
// JSON: {"module_name":"counter","timing_levels":[
hw.module @counter(in %a: i16, in %clk: !seq.clock, out result: i16) {
    %reg = seq.compreg %add, %clk : i16
    %add = comb.mul %reg, %a : i16
    hw.output %reg : i16
}
