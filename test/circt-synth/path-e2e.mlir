// RUN: circt-synth %s -output-longest-path=- -top counter --enable-sop-balancing | FileCheck %s --check-prefixes COMMON,AIG
// RUN: circt-synth %s -output-longest-path=- -top counter --target-ir mig | FileCheck %s --check-prefixes COMMON,MIG
// RUN: circt-synth %s -output-longest-path=- -top counter -lower-to-k-lut 6 | FileCheck %s --check-prefixes COMMON,LUT6
// RUN: circt-synth %s -output-longest-path=- -top test -output-longest-path-json | FileCheck %s --check-prefix JSON

// COMMON-LABEL: # Longest Path Analysis result for "counter"
// COMMON-NEXT: Found 168 paths
// COMMON-NEXT: Found 32 unique end points
// AIG-NEXT: Maximum path delay: 27
// MIG-NEXT: Maximum path delay: 32
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
