// RUN: circt-opt %s --verify-diagnostics

hw.module @test(in %a : i1, in %b : i1, out result : i1) {
    // expected-error @+1 {{'synth.mig.maj_inv' op requires an odd number of operands}}
    %0 = synth.mig.maj_inv %a, %b : i1
    hw.output %0 : i1
}
