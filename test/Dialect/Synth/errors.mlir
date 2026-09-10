// RUN: circt-opt %s --verify-diagnostics --split-input-file

hw.module @test(out result : i1) {
    // expected-error @+1 {{'synth.choice' op expected 1 or more operands, but found 0}}
    %0 = synth.choice : i1
    hw.output %0 : i1
}
