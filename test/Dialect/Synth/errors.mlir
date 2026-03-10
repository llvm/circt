// RUN: circt-opt %s --verify-diagnostics --split-input-file

hw.module @test(in %a : i1, in %b : i1, out result : i1) {
    // expected-error @+1 {{'synth.mig.maj_inv' op requires an odd number of operands}}
    %0 = synth.mig.maj_inv %a, %b : i1
    hw.output %0 : i1
}

// -----

hw.module @test(out result : i1) {
    // expected-error @+1 {{'synth.choice' op expected 1 or more operands, but found 0}}
    %0 = synth.choice : i1
    hw.output %0 : i1
}
