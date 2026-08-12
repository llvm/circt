// RUN: circt-opt %s --verify-diagnostics --split-input-file

// expected-error @below {{failed to verify 'elementType': type supported for probe observation}}
hw.module @BadInOut(in %p: !probe.ref<!hw.inout<i1>>) {
}

// -----

// expected-error @below {{failed to verify 'elementType': type supported for probe observation}}
hw.module @UnsupportedSeqType(in %p: !probe.ref<!seq.immutable<i8>>) {
}

// -----

// expected-error @below {{probe refs are only supported on output ports}}
hw.module @InputRef(in %p: !probe.ref<i8>) {
}

// -----

// expected-error @below {{probe refs are only supported on output ports}}
hw.module @AggregateInputRef(in %p: !hw.struct<ref: !probe.ref<i8>>) {
}

// -----

hw.module @BadRead(in %in: i8) {
  %p = probe.send %in : i8
  // expected-error @below {{failed to verify that input and result types match}}
  %v = "probe.read"(%p) : (!probe.ref<i8>) -> i7
}

// -----

hw.module @BadRef(in %in: i8) {
  // expected-error @below {{failed to verify that input and ref types match}}
  %p = "probe.send"(%in) : (i8) -> !probe.ref<i7>
}

// -----

hw.module @ProbeInTriggered(in %in: i8, in %clock: i1) {
  hw.triggered posedge %clock(%in) : i8 {
    ^bb0(%arg: i8):
    // expected-error @below {{expects parent op 'hw.module'}}
    %p = probe.send %arg : i8
    %v = probe.read %p : <i8>
  }
}

// -----

hw.module @ProbeInNestedGraphRegion(in %in: i8) {
  sv.ordered {
    // expected-error @below {{expects parent op 'hw.module'}}
    %p = probe.send %in : i8
  }
}
