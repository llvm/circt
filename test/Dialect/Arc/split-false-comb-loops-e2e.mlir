// RUN: circt-opt %s --arc-split-false-comb-loops --convert-to-arcs | FileCheck %s

// End-to-end: after splitting, ConvertToArcs' fan-in analysis must accept the
// previously cyclic modules (it would otherwise fail with "combinational loop
// detected").

// CHECK: arc.define
// CHECK: hw.module @falseArrayLoop
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

// CHECK: hw.module @falseBitLoop
hw.module @falseBitLoop(in %a : i1, in %b : i1, out out : i1) {
  %w = comb.concat %dep, %a : i1, i1
  %bit0 = comb.extract %w from 0 : (i2) -> i1
  %dep = comb.and %bit0, %b : i1
  hw.output %dep : i1
}

// CHECK: hw.module @falseNestedLoop
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
