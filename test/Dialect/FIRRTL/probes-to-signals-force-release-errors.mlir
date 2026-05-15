// RUN: circt-opt --firrtl-probes-to-signals --split-input-file --verify-diagnostics %s

// Test error on multiple different clocks
firrtl.circuit "MultipleClocks" {
  firrtl.module @MultipleClocks(in %clock1: !firrtl.clock, in %clock2: !firrtl.clock, in %cond: !firrtl.uint<1>, in %value: !firrtl.uint<4>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    
    firrtl.ref.force %clock1, %cond, %w_ref, %value : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>
    // expected-error @+1 {{multiple different clocks on force/release operations targeting the same probe not supported}}
    firrtl.ref.force %clock2, %cond, %w_ref, %value : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>
  }
}

// -----

// Test error on force_initial/release_initial only (no clock)
firrtl.circuit "NoClockError" {
  firrtl.module @NoClockError(in %cond: !firrtl.uint<1>, in %value: !firrtl.uint<4>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    
    // expected-error @+1 {{no clock found for force/release operations (only force_initial/release_initial present)}}
    firrtl.ref.force_initial %cond, %w_ref, %value : !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>, !firrtl.uint<4>
  }
}
