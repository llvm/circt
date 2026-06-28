// RUN: circt-translate --export-firrtl --firrtl-version=2.0.0 --verify-diagnostics --split-input-file %s

// The legacy FIRRTL < 3.0.0 register syntax ('reg ... with :') has no way to
// state the reset kind or polarity, so exporting an asynchronous or active-low
// reset to an old version must be rejected rather than silently emitted as a
// synchronous active-high reset. (At version >= 3.0.0 the 'regreset' inline
// qualifiers represent these directly and export succeeds.)
firrtl.circuit "AsyncRegResetExport" {
  firrtl.module @AsyncRegResetExport(in %clock: !firrtl.clock,
                                     in %reset: !firrtl.reset,
                                     in %in: !firrtl.uint<8>) {
    // expected-error @below {{cannot be exported to FIRRTL versions < 3.0.0: the legacy register syntax cannot represent an asynchronous or active-low reset}}
    %r = firrtl.regreset %clock, %reset, %in {clockEdge = 0 : i32, resetPolarity = 0 : i32, resetType = 1 : i32} : !firrtl.clock, !firrtl.reset, !firrtl.uint<8>, !firrtl.uint<8>
  }
}

// -----

firrtl.circuit "ActiveLowRegResetExport" {
  firrtl.module @ActiveLowRegResetExport(in %clock: !firrtl.clock,
                                         in %reset: !firrtl.reset,
                                         in %in: !firrtl.uint<8>) {
    // expected-error @below {{cannot be exported to FIRRTL versions < 3.0.0: the legacy register syntax cannot represent an asynchronous or active-low reset}}
    %r = firrtl.regreset %clock, %reset, %in {clockEdge = 0 : i32, resetPolarity = 1 : i32, resetType = 0 : i32} : !firrtl.clock, !firrtl.reset, !firrtl.uint<8>, !firrtl.uint<8>
  }
}
