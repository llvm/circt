// RUN: circt-opt --pass-pipeline='builtin.module(hw.module(synth-test-priority-cuts))' %s --split-input-file --verify-diagnostics

hw.module @test_too_many_operands(in %a : i1, in %b : i1, in %c : i1, out result : i1) {
    // expected-error @+1 {{Cut enumeration supports at most 2 operands, found: 3}}
    %and_three = synth.aig.and_inv %a, %b, %c : i1
    hw.output %and_three : i1
}

// -----

hw.module @test_multi_bit_result(in %a : i2, in %b : i2, out result : i2) {
    // expected-error @+1 {{Supported logic operations must have a single bit result type but found: 'i2'}}
    %and_multi = synth.aig.and_inv %a, %b : i2
    hw.output %and_multi : i2
}

// -----

// expected-error @+1 {{failed to sort operations topologically}}
hw.module @test_topo_sort(in %a : i1, out result : i1) {
    %c = synth.aig.and_inv %a, %d: i1
    %d = synth.aig.and_inv %a, %c : i1

    hw.output %c : i1
}
