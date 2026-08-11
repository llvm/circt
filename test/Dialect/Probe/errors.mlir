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
