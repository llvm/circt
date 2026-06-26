// RUN: circt-opt %s --verify-diagnostics --split-input-file

// expected-error @below {{probe element type must be a non-inout type containing only HW value types or seq.clock leaves}}
hw.module @BadInOut(in %p: !probe.ref<!hw.inout<i1>>) {
}

// -----

hw.module @BadRead(in %in: i8) {
  %p = probe.send %in : i8 -> !probe.ref<i8>
  // expected-error @below {{result type must match input probe element type}}
  %v = "probe.read"(%p) : (!probe.ref<i8>) -> i7
}

// -----

hw.module @SubfieldNonStruct(in %in: i8) {
  %p = probe.send %in : i8 -> !probe.ref<i8>
  // expected-error @below {{input probe element type must be an hw.struct}}
  %q = probe.subfield %p["x"] : !probe.ref<i8> -> !probe.ref<i1>
}

// -----

hw.module @SubfieldMissing(in %in: !hw.struct<a: i1>) {
  %p = probe.send %in : !hw.struct<a: i1> -> !probe.ref<!hw.struct<a: i1>>
  // expected-error @below {{field 'x' not found}}
  %q = probe.subfield %p["x"] : !probe.ref<!hw.struct<a: i1>> -> !probe.ref<i1>
}

// -----

hw.module @SubindexNonArray(in %in: i8) {
  %p = probe.send %in : i8 -> !probe.ref<i8>
  // expected-error @below {{input probe element type must be an hw.array}}
  %q = probe.subindex %p[0] : !probe.ref<i8> -> !probe.ref<i8>
}

// -----

hw.module @SubindexOutOfBounds(in %in: !hw.array<2xi8>) {
  %p = probe.send %in : !hw.array<2xi8> -> !probe.ref<!hw.array<2xi8>>
  // expected-error @below {{index 2 out of bounds}}
  %q = probe.subindex %p[2] : !probe.ref<!hw.array<2xi8>> -> !probe.ref<i8>
}

// -----

hw.module @BadCast(in %in: i8) {
  %p = probe.send %in : i8 -> !probe.ref<i8>
  // expected-error @below {{input and result probe element types must have the same canonical HW type}}
  %q = probe.cast %p : !probe.ref<i8> -> !probe.ref<i7>
}
