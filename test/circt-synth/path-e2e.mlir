// RUN: circt-synth %s -analysis-output=- -top counter --enable-sop-balancing | FileCheck %s --check-prefixes COMMON,AIG
// RUN: circt-synth %s -analysis-output=- -top counter -lower-to-k-lut 6 | FileCheck %s --check-prefixes COMMON,LUT6
// RUN: circt-synth %s -analysis-output=- -top test -analysis-output-format=json | FileCheck %s --check-prefix JSON

// COMMON-LABEL: # Longest Path Analysis result for "counter"
// AIG-NEXT: Found 175 paths
// LUT6-NEXT: Found 176 paths
// COMMON-NEXT: Found 32 unique end points
// AIG-NEXT: Maximum path delay: 26
// LUT6-NEXT: Maximum path delay: 6
// Don't test detailed reports as they are not stable.

hw.module @counter(in %a: i16, in %clk: !seq.clock, out result: i16) {
    %reg = seq.compreg %add, %clk : i16
    %add = comb.mul %reg, %a : i16
    hw.output %reg : i16
}

// Make sure json is emitted.
// JSON: {"module_name":"test","timing_levels":[
// COMMON-NOT: "test"
hw.module @test(in %a: i16, out result: i16) {
    hw.output %a : i16
}
