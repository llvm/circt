// RUN: circt-opt %s --arc-split-false-comb-loops | FileCheck %s

// An unrolled per-element update chain: the element-0 write uses the final
// chained value as its base, forming a whole-value SSA cycle in the graph
// region. Every element is unconditionally overwritten at its own stage, so
// the loop is false at element granularity.
// CHECK-LABEL: hw.module @falseArrayLoop
// CHECK-NOT: hw.array_inject
// CHECK: hw.output
hw.module @falseArrayLoop(in %sel : i1, in %a : i3, in %b : i3, out out : !hw.array<2xi3>) {
  %false = hw.constant false
  %true = hw.constant true
  %s0a = hw.array_inject %final[%false], %a : !hw.array<2xi3>, i1
  %s0b = hw.array_inject %final[%false], %b : !hw.array<2xi3>, i1
  %m0 = comb.mux %sel, %s0a, %s0b : !hw.array<2xi3>
  %s1a = hw.array_inject %m0[%true], %a : !hw.array<2xi3>, i1
  %s1b = hw.array_inject %m0[%true], %b : !hw.array<2xi3>, i1
  %final = comb.mux %sel, %s1a, %s1b : !hw.array<2xi3>
  hw.output %final : !hw.array<2xi3>
}

// Interlocking struct field updates: each inject reads the other's result,
// but each field is overwritten exactly once. False at field granularity.
// CHECK-LABEL: hw.module @falseStructLoop
// CHECK-NOT: hw.struct_inject
// CHECK: hw.struct_create
hw.module @falseStructLoop(in %a : i4, in %b : i8, out out : !hw.struct<x: i4, y: i8>) {
  %v0 = hw.struct_inject %v1["x"], %a : !hw.struct<x: i4, y: i8>
  %v1 = hw.struct_inject %v0["y"], %b : !hw.struct<x: i4, y: i8>
  hw.output %v1 : !hw.struct<x: i4, y: i8>
}

// Bitwise feedback where bit 1 of the concat depends on bit 0 through an
// extract: acyclic per bit.
// CHECK-LABEL: hw.module @falseBitLoop
// CHECK: [[AND:%.+]] = comb.and %a, %b : i1
// CHECK: hw.output [[AND]]
hw.module @falseBitLoop(in %a : i1, in %b : i1, out out : i1) {
  %w = comb.concat %dep, %a : i1, i1
  %bit0 = comb.extract %w from 0 : (i2) -> i1
  %dep = comb.and %bit0, %b : i1
  hw.output %dep : i1
}

// A nested aggregate (array of structs): the recursive leaf decomposition
// resolves the element- and field-level pass-throughs in one round.
// CHECK-LABEL: hw.module @falseNestedLoop
// CHECK-NOT: hw.array_inject
// CHECK-NOT: hw.struct_inject
hw.module @falseNestedLoop(in %sel : i1, in %a : i4, in %b : i8, out out : !hw.array<2xstruct<x: i4, y: i8>>) {
  %false = hw.constant false
  %true = hw.constant true
  %e1 = hw.array_get %final[%true] : !hw.array<2xstruct<x: i4, y: i8>>, i1
  %e1x = hw.struct_inject %e1["x"], %a : !hw.struct<x: i4, y: i8>
  %e1xy = hw.struct_inject %e1x["y"], %b : !hw.struct<x: i4, y: i8>
  %u0 = hw.array_inject %final[%false], %e1xy : !hw.array<2xstruct<x: i4, y: i8>>, i1
  %final = hw.array_inject %u0[%true], %e1xy : !hw.array<2xstruct<x: i4, y: i8>>, i1
  hw.output %final : !hw.array<2xstruct<x: i4, y: i8>>
}

// A genuine loop: element 0 feeds back into itself through the mux' false
// path. The pass must leave it alone (ConvertToArcs then reports it).
// CHECK-LABEL: hw.module @genuineElementLoop
// CHECK: hw.array_inject
hw.module @genuineElementLoop(in %sel : i1, in %a : i3, out out : !hw.array<2xi3>) {
  %false = hw.constant false
  %e0 = hw.array_get %w[%false] : !hw.array<2xi3>, i1
  %e0inc = comb.mux %sel, %a, %e0 : i3
  %w = hw.array_inject %w0[%false], %e0inc : !hw.array<2xi3>, i1
  %w0 = hw.array_inject %w[%false], %a : !hw.array<2xi3>, i1
  hw.output %w : !hw.array<2xi3>
}

// A genuine bitwise loop (bit 0 depends on itself): left alone.
// CHECK-LABEL: hw.module @genuineBitLoop
// CHECK: comb.concat
hw.module @genuineBitLoop(in %a : i1, out out : i2) {
  %bit0 = comb.extract %w from 0 : (i2) -> i1
  %dep = comb.and %bit0, %a : i1
  %w = comb.concat %a, %dep : i1, i1
  hw.output %w : i2
}
