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

// -----

hw.module @SubfieldNonStruct(in %in: i8) {
  %p = probe.send %in : i8
  // expected-error @below {{input probe element type must be an hw.struct}}
  %q = probe.subfield %p["x"] : <i8> -> <i1>
}

// -----

hw.module @SubfieldMissing(in %in: !hw.struct<a: i1>) {
  %p = probe.send %in : !hw.struct<a: i1>
  // expected-error @below {{field 'x' not found}}
  %q = probe.subfield %p["x"] : <!hw.struct<a: i1>> -> <i1>
}

// -----

hw.module @SubindexNonArray(in %in: i8) {
  %p = probe.send %in : i8
  // expected-error @below {{operand #0 must be ref of an ArrayType}}
  %q = "probe.subindex"(%p) <{index = 0 : i32}> : (!probe.ref<i8>) -> !probe.ref<i8>
}

// -----

hw.module @SubindexOutOfBounds(in %in: !hw.array<2xi8>) {
  %p = probe.send %in : !hw.array<2xi8>
  // expected-error @below {{index 2 out of bounds}}
  %q = probe.subindex %p[2] : <!hw.array<2xi8>>
}

// -----

hw.module @BadCast(in %in: i8) {
  %p = probe.send %in : i8
  // expected-error @below {{input and result probe element types must have the same canonical HW type}}
  %q = probe.cast %p : <i8> -> <i7>
}
