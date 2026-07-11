// RUN: circt-opt %s --verify-diagnostics --split-input-file

// expected-error @below {{probe element type must be a non-inout type containing only HW value types or seq.clock leaves}}
hw.module @BadInOut(in %p: !probe.ref<!hw.inout<i1>>) {
}

// -----

// expected-error @below {{input probe refs are not supported}}
hw.module @InputRef(in %p: !probe.ref<i8>) {
}

// -----

// expected-error @below {{input probe refs are not supported}}
hw.module @AggregateInputRef(in %p: !hw.struct<ref: !probe.ref<i8>>) {
}

// -----

hw.module @BadRead(in %in: i8) {
  %p = probe.send %in : i8
  // expected-error @below {{failed to verify that input and result types match}}
  %v = "probe.read"(%p) : (!probe.ref<i8>) -> i7
}

// -----

hw.module @BadSend(in %in: i8) {
  // expected-error @below {{failed to verify that input and result types match}}
  %p = "probe.send"(%in) : (i8) -> !probe.ref<i7>
}
