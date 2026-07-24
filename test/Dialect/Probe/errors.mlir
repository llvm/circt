// RUN: circt-opt %s --verify-diagnostics --split-input-file

// expected-error @below {{probe element type must be a non-inout type containing only HW value types or seq.clock leaves}}
hw.module @BadInOut(in %p: !probe.ref<!hw.inout<i1>>) {
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
  %forwarded, %p = probe.send %in : i8
  // expected-error @below {{failed to verify that input and result types match}}
  %v = "probe.read"(%p) : (!probe.ref<i8>) -> i7
}

// -----

hw.module @BadForwarded(in %in: i8) {
  // expected-error @below {{failed to verify that input and forwarded types match}}
  %forwarded, %p = "probe.send"(%in) : (i8) -> (i7, !probe.ref<i8>)
}

// -----

hw.module @BadRef(in %in: i8) {
  // expected-error @below {{failed to verify that input and ref types match}}
  %forwarded, %p = "probe.send"(%in) : (i8) -> (i8, !probe.ref<i7>)
}
