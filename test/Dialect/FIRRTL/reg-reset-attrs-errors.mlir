// RUN: circt-opt %s -split-input-file -verify-diagnostics

// An `asyncreset`-typed reset must carry an explicit `resetType = AsyncReset`;
// otherwise it would lower as synchronous, silently changing the hardware.
firrtl.circuit "AsyncResetMissingAttr" {
firrtl.module @AsyncResetMissingAttr(in %clock: !firrtl.clock,
                                     in %reset: !firrtl.asyncreset,
                                     in %in: !firrtl.uint<8>) {
  // expected-error @+1 {{has an 'asyncreset'-typed reset but its 'resetType' attribute is not 'AsyncReset'}}
  %r = firrtl.regreset %clock, %reset, %in : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<8>, !firrtl.uint<8>
}
}

// -----

// An `asyncreset`-typed reset explicitly marked synchronous is rejected: the
// attribute would silently turn the asynchronous reset into a synchronous one.
firrtl.circuit "AsyncResetMarkedSync" {
firrtl.module @AsyncResetMarkedSync(in %clock: !firrtl.clock,
                                    in %reset: !firrtl.asyncreset,
                                    in %in: !firrtl.uint<8>) {
  // expected-error @+1 {{has an 'asyncreset'-typed reset but its 'resetType' attribute is not 'AsyncReset'}}
  %r = firrtl.regreset %clock, %reset, %in {resetType = 0 : i32} : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<8>, !firrtl.uint<8>
}
}

// -----

// Reset behavior attributes are not allowed on a reset-less register.
firrtl.circuit "RegWithResetType" {
firrtl.module @RegWithResetType(in %clock: !firrtl.clock) {
  // expected-error @+1 {{has reset attribute 'resetType', which is only valid on 'firrtl.regreset'}}
  %r = firrtl.reg %clock {resetType = 1 : i32} : !firrtl.clock, !firrtl.uint<8>
}
}

// -----

firrtl.circuit "RegWithResetEdge" {
firrtl.module @RegWithResetEdge(in %clock: !firrtl.clock) {
  // expected-error @+1 {{has reset attribute 'resetPolarity', which is only valid on 'firrtl.regreset'}}
  %r = firrtl.reg %clock {resetPolarity = 1 : i32} : !firrtl.clock, !firrtl.uint<8>
}
}
