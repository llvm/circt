// RUN: circt-opt --sv-check-var-drivers --split-input-file --verify-diagnostics %s | FileCheck %s

// NOTE: variables are currently declared with sv.reg and driven with
// sv.assign / sv.bpassign / sv.passign. Once the var/net migration lands,
// convert these to sv.var / sv.var_assign.

//===----------------------------------------------------------------------===//
// Legal
//===----------------------------------------------------------------------===//

// One continuous driver on a whole variable.
// CHECK-LABEL: hw.module @WholeVariable
hw.module @WholeVariable(in %a: i8) {
  %v = sv.reg : !hw.inout<i8>
  sv.assign %v, %a : i8
}

// -----

// Non-overlapping continuous slices of a packed vector.
// CHECK-LABEL: hw.module @DisjointSlices
hw.module @DisjointSlices(in %x: i4, in %y: i4) {
  %c0 = hw.constant 0 : i32
  %c4 = hw.constant 4 : i32
  %bus = sv.reg : !hw.inout<i8>
  %lo = sv.indexed_part_select_inout %bus[%c0 : 4] : !hw.inout<i8>, i32
  %hi = sv.indexed_part_select_inout %bus[%c4 : 4] : !hw.inout<i8>, i32
  sv.assign %lo, %x : i4
  sv.assign %hi, %y : i4
}

// -----

// Separate struct members.
// CHECK-LABEL: hw.module @DisjointFields
hw.module @DisjointFields(in %x: i8, in %y: i8) {
  %s = sv.reg : !hw.inout<struct<a: i8, b: i8>>
  %a = sv.struct_field_inout %s["a"] : !hw.inout<struct<a: i8, b: i8>>
  %b = sv.struct_field_inout %s["b"] : !hw.inout<struct<a: i8, b: i8>>
  sv.assign %a, %x : i8
  sv.assign %b, %y : i8
}

// -----

// Separate array elements.
// CHECK-LABEL: hw.module @DisjointElements
hw.module @DisjointElements(in %x: i8, in %y: i8) {
  %c0 = hw.constant 0 : i2
  %c1 = hw.constant 1 : i2
  %arr = sv.reg : !hw.inout<array<4xi8>>
  %e0 = sv.array_index_inout %arr[%c0] : !hw.inout<array<4xi8>>, i2
  %e1 = sv.array_index_inout %arr[%c1] : !hw.inout<array<4xi8>>, i2
  sv.assign %e0, %x : i8
  sv.assign %e1, %y : i8
}

// -----

// Any number of procedural writes to the same bits is legal.
// CHECK-LABEL: hw.module @ManyProcedural
hw.module @ManyProcedural(in %clk: i1, in %a: i8, in %b: i8) {
  %v = sv.reg : !hw.inout<i8>
  sv.always posedge %clk {
    sv.bpassign %v, %a : i8
    sv.passign %v, %b : i8
  }
  sv.initial {
    sv.bpassign %v, %a : i8
  }
}

// -----

// A continuous driver on one part and a procedural write to a disjoint part
// of the same vector: each bit is an independent element (SV Spec 6.5).
// CHECK-LABEL: hw.module @ContinuousAndProceduralDisjoint
hw.module @ContinuousAndProceduralDisjoint(in %clk: i1, in %x: i4, in %y: i4) {
  %c0 = hw.constant 0 : i32
  %c4 = hw.constant 4 : i32
  %bus = sv.reg : !hw.inout<i8>
  %lo = sv.indexed_part_select_inout %bus[%c0 : 4] : !hw.inout<i8>, i32
  %hi = sv.indexed_part_select_inout %bus[%c4 : 4] : !hw.inout<i8>, i32
  sv.assign %lo, %x : i4
  sv.always posedge %clk {
    sv.passign %hi, %y : i4
  }
}

// -----

// Multiple drivers on a net are legal; only variables are checked.
// CHECK-LABEL: hw.module @NetsAreNotChecked
hw.module @NetsAreNotChecked(in %a: i8, in %b: i8) {
  %w = sv.wire : !hw.inout<i8>
  sv.assign %w, %a : i8
  sv.assign %w, %b : i8
}

// -----

// The indexed part-select `base -: width` is a decrementing select.
// CHECK-LABEL: hw.module @DecrementingSelect
hw.module @DecrementingSelect(in %x: i4, in %y: i4) {
  %c3 = hw.constant 3 : i32
  %c7 = hw.constant 7 : i32
  %bus = sv.reg : !hw.inout<i8>
  %lo = sv.indexed_part_select_inout %bus[%c3 decrement : 4] : !hw.inout<i8>, i32
  %hi = sv.indexed_part_select_inout %bus[%c7 decrement : 4] : !hw.inout<i8>, i32
  sv.assign %lo, %x : i4
  sv.assign %hi, %y : i4
}

// -----

// Nested projection: struct field of an array element.
// CHECK-LABEL: hw.module @NestedProjections
hw.module @NestedProjections(in %x: i8, in %y: i8) {
  %c0 = hw.constant 0 : i1
  %c1 = hw.constant 1 : i1
  %arr = sv.reg : !hw.inout<array<2xstruct<a: i8>>>
  %e0 = sv.array_index_inout %arr[%c0] : !hw.inout<array<2xstruct<a: i8>>>, i1
  %e1 = sv.array_index_inout %arr[%c1] : !hw.inout<array<2xstruct<a: i8>>>, i1
  %f0 = sv.struct_field_inout %e0["a"] : !hw.inout<struct<a: i8>>
  %f1 = sv.struct_field_inout %e1["a"] : !hw.inout<struct<a: i8>>
  sv.assign %f0, %x : i8
  sv.assign %f1, %y : i8
}

// -----

//===----------------------------------------------------------------------===//
// Illegal
//===----------------------------------------------------------------------===//

hw.module @TwoWholeDrivers(in %a: i8, in %b: i8) {
  %v = sv.reg : !hw.inout<i8>
  // expected-note @below {{other driver is here}}
  sv.assign %v, %a : i8
  // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
  sv.assign %v, %b : i8
}

// -----

// The error is reported on the driver that sorts later by (start, end), here
// the wider slice.
hw.module @OverlappingSlices(in %x: i4, in %y: i3) {
  %c0 = hw.constant 0 : i32
  %bus = sv.reg : !hw.inout<i8>
  %s4 = sv.indexed_part_select_inout %bus[%c0 : 4] : !hw.inout<i8>, i32
  %s3 = sv.indexed_part_select_inout %bus[%c0 : 3] : !hw.inout<i8>, i32
  // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
  sv.assign %s4, %x : i4
  // expected-note @below {{other driver is here}}
  sv.assign %s3, %y : i3
}

// -----

// A whole-variable driver conflicts with a driver on any part of it.
hw.module @WholeAndPart(in %a: i8, in %x: i4) {
  %c0 = hw.constant 0 : i32
  %bus = sv.reg : !hw.inout<i8>
  %lo = sv.indexed_part_select_inout %bus[%c0 : 4] : !hw.inout<i8>, i32
  // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
  sv.assign %bus, %a : i8
  // expected-note @below {{other driver is here}}
  sv.assign %lo, %x : i4
}

// -----

hw.module @SameStructFieldTwice(in %x: i8, in %y: i8) {
  %s = sv.reg : !hw.inout<struct<a: i8, b: i8>>
  %a0 = sv.struct_field_inout %s["a"] : !hw.inout<struct<a: i8, b: i8>>
  %a1 = sv.struct_field_inout %s["a"] : !hw.inout<struct<a: i8, b: i8>>
  // expected-note @below {{other driver is here}}
  sv.assign %a0, %x : i8
  // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
  sv.assign %a1, %y : i8
}

// -----

hw.module @SameArrayElementTwice(in %x: i8, in %y: i8) {
  %c1 = hw.constant 1 : i2
  %arr = sv.reg : !hw.inout<array<4xi8>>
  %e0 = sv.array_index_inout %arr[%c1] : !hw.inout<array<4xi8>>, i2
  %e1 = sv.array_index_inout %arr[%c1] : !hw.inout<array<4xi8>>, i2
  // expected-note @below {{other driver is here}}
  sv.assign %e0, %x : i8
  // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
  sv.assign %e1, %y : i8
}

// -----

// A continuous driver mixed with a procedural write on the same bits.
hw.module @ContinuousAndProcedural(in %clk: i1, in %a: i8, in %b: i8) {
  %v = sv.reg : !hw.inout<i8>
  // expected-note @below {{other driver is here}}
  sv.assign %v, %a : i8
  sv.always posedge %clk {
    // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
    sv.passign %v, %b : i8
  }
}

// -----

// Only one bit in common is enough to conflict.
hw.module @ContinuousAndProceduralPartialOverlap(in %clk: i1, in %x: i4, in %y: i4) {
  %c0 = hw.constant 0 : i32
  %c3 = hw.constant 3 : i32
  %bus = sv.reg : !hw.inout<i8>
  %lo = sv.indexed_part_select_inout %bus[%c0 : 4] : !hw.inout<i8>, i32
  %mid = sv.indexed_part_select_inout %bus[%c3 : 4] : !hw.inout<i8>, i32
  // expected-note @below {{other driver is here}}
  sv.assign %lo, %x : i4
  sv.always posedge %clk {
    // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
    sv.passign %mid, %y : i4
  }
}

// -----

// A dynamic index makes the whole array the longest static prefix, so it
// conflicts with a continuous driver on any element (SV Spec 11.5.3).
hw.module @DynamicIndexClaimsWholeArray(in %clk: i1, in %i: i2, in %a: i8, in %b: i8) {
  %c0 = hw.constant 0 : i2
  %arr = sv.reg : !hw.inout<array<4xi8>>
  %fixed = sv.array_index_inout %arr[%c0] : !hw.inout<array<4xi8>>, i2
  %dyn = sv.array_index_inout %arr[%i] : !hw.inout<array<4xi8>>, i2
  // expected-note @below {{other driver is here}}
  sv.assign %fixed, %a : i8
  sv.always posedge %clk {
    // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
    sv.passign %dyn, %b : i8
  }
}

// -----

// Constants whose top bit is set (2 and 3 in i2) are valid unsigned indices,
// not negative ones, and must not be treated as dynamic.
hw.module @HighBitIndices(in %x: i8, in %y: i8) {
  %c2 = hw.constant 2 : i2
  %c3 = hw.constant 3 : i2
  %arr = sv.reg : !hw.inout<array<4xi8>>
  %e2 = sv.array_index_inout %arr[%c2] : !hw.inout<array<4xi8>>, i2
  %e3 = sv.array_index_inout %arr[%c3] : !hw.inout<array<4xi8>>, i2
  sv.assign %e2, %x : i8
  sv.assign %e3, %y : i8
}

// -----

//===----------------------------------------------------------------------===//
// Additional legal cases
//===----------------------------------------------------------------------===//

// sv.logic is a variable just like sv.reg.
// CHECK-LABEL: hw.module @LogicRoot
hw.module @LogicRoot(in %x: i4, in %y: i4) {
  %c0 = hw.constant 0 : i32
  %c4 = hw.constant 4 : i32
  %bus = sv.logic : !hw.inout<i8>
  %lo = sv.indexed_part_select_inout %bus[%c0 : 4] : !hw.inout<i8>, i32
  %hi = sv.indexed_part_select_inout %bus[%c4 : 4] : !hw.inout<i8>, i32
  sv.assign %lo, %x : i4
  sv.assign %hi, %y : i4
}

// -----

// A part-select inside a struct member: each bit of the member is independent.
// CHECK-LABEL: hw.module @PartSelectsInsideFields
hw.module @PartSelectsInsideFields(in %x: i4, in %y: i4, in %z: i8) {
  %c0 = hw.constant 0 : i32
  %c4 = hw.constant 4 : i32
  %s = sv.reg : !hw.inout<struct<a: i8, b: i8>>
  %a = sv.struct_field_inout %s["a"] : !hw.inout<struct<a: i8, b: i8>>
  %b = sv.struct_field_inout %s["b"] : !hw.inout<struct<a: i8, b: i8>>
  %alo = sv.indexed_part_select_inout %a[%c0 : 4] : !hw.inout<i8>, i32
  %ahi = sv.indexed_part_select_inout %a[%c4 : 4] : !hw.inout<i8>, i32
  sv.assign %alo, %x : i4
  sv.assign %ahi, %y : i4
  sv.assign %b, %z : i8
}

// -----

// Overlapping procedural writes are legal, even when they are partial.
// CHECK-LABEL: hw.module @OverlappingProceduralSlices
hw.module @OverlappingProceduralSlices(in %clk: i1, in %x: i4, in %y: i3) {
  %c0 = hw.constant 0 : i32
  %bus = sv.reg : !hw.inout<i8>
  %s4 = sv.indexed_part_select_inout %bus[%c0 : 4] : !hw.inout<i8>, i32
  %s3 = sv.indexed_part_select_inout %bus[%c0 : 3] : !hw.inout<i8>, i32
  sv.always posedge %clk {
    sv.passign %s4, %x : i4
    sv.passign %s3, %y : i3
  }
  sv.initial {
    sv.bpassign %s3, %y : i3
  }
}

// -----

// Drivers are grouped per variable: identical ranges on different variables
// never conflict.
// CHECK-LABEL: hw.module @IndependentVariables
hw.module @IndependentVariables(in %a: i8, in %b: i8) {
  %v = sv.reg : !hw.inout<i8>
  %w = sv.reg : !hw.inout<i8>
  %l = sv.logic : !hw.inout<i8>
  sv.assign %v, %a : i8
  sv.assign %w, %b : i8
  sv.assign %l, %a : i8
}

// -----

// A procedural write that stops short of a continuous driver does not conflict.
// CHECK-LABEL: hw.module @AdjacentContinuousAndProcedural
hw.module @AdjacentContinuousAndProcedural(in %clk: i1, in %x: i4, in %y: i2) {
  %c0 = hw.constant 0 : i32
  %c4 = hw.constant 4 : i32
  %c6 = hw.constant 6 : i32
  %bus = sv.reg : !hw.inout<i8>
  %p = sv.indexed_part_select_inout %bus[%c0 : 2] : !hw.inout<i8>, i32
  %c = sv.indexed_part_select_inout %bus[%c4 : 4] : !hw.inout<i8>, i32
  sv.always posedge %clk {
    sv.passign %p, %y : i2
  }
  sv.assign %c, %x : i4
}

// -----

//===----------------------------------------------------------------------===//
// Additional illegal cases
//===----------------------------------------------------------------------===//

// A dynamic part-select base makes the whole variable the longest static
// prefix.
hw.module @DynamicPartSelectClaimsWholeVariable(in %i: i32, in %x: i4, in %y: i4) {
  %c0 = hw.constant 0 : i32
  %bus = sv.reg : !hw.inout<i8>
  %fixed = sv.indexed_part_select_inout %bus[%c0 : 4] : !hw.inout<i8>, i32
  %dyn = sv.indexed_part_select_inout %bus[%i : 4] : !hw.inout<i8>, i32
  // expected-note @below {{other driver is here}}
  sv.assign %fixed, %x : i4
  // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
  sv.assign %dyn, %y : i4
}

// -----

// A part-select inside a struct member conflicts with the whole member.
hw.module @PartSelectAndWholeField(in %x: i4, in %y: i8) {
  %c0 = hw.constant 0 : i32
  %s = sv.reg : !hw.inout<struct<a: i8, b: i8>>
  %a = sv.struct_field_inout %s["a"] : !hw.inout<struct<a: i8, b: i8>>
  %alo = sv.indexed_part_select_inout %a[%c0 : 4] : !hw.inout<i8>, i32
  // expected-note @below {{other driver is here}}
  sv.assign %alo, %x : i4
  // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
  sv.assign %a, %y : i8
}

// -----

hw.module @LogicRootTwoDrivers(in %a: i8, in %b: i8) {
  %v = sv.logic : !hw.inout<i8>
  // expected-note @below {{other driver is here}}
  sv.assign %v, %a : i8
  // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
  sv.assign %v, %b : i8
}

// -----

// A conflict on one variable is not attributed to another variable.
hw.module @ErrorsStayWithTheirVariable(in %a: i8, in %b: i8) {
  %v = sv.reg : !hw.inout<i8>
  %w = sv.reg : !hw.inout<i8>
  // expected-note @below {{other driver is here}}
  sv.assign %v, %a : i8
  sv.assign %w, %a : i8
  // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
  sv.assign %v, %b : i8
}

// -----

// Same bits, procedural first: the continuous driver is the one reported.
hw.module @ProceduralThenContinuous(in %clk: i1, in %a: i8, in %b: i8) {
  %v = sv.reg : !hw.inout<i8>
  sv.always posedge %clk {
    // expected-note @below {{other driver is here}}
    sv.passign %v, %a : i8
  }
  // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
  sv.assign %v, %b : i8
}

// -----

// The kind of procedural block does not matter.
hw.module @ContinuousAndAlwaysComb(in %a: i8, in %b: i8) {
  %v = sv.reg : !hw.inout<i8>
  // expected-note @below {{other driver is here}}
  sv.assign %v, %a : i8
  sv.alwayscomb {
    // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
    sv.bpassign %v, %b : i8
  }
}

// -----

// A wide procedural driver must keep covering later, narrower procedural
// writes: the continuous driver still conflicts with the wide one.
hw.module @WideProceduralHidesNarrowOnes(in %clk: i1, in %a: i8, in %b: i1, in %c: i1) {
  %c2 = hw.constant 2 : i32
  %c5 = hw.constant 5 : i32
  %bus = sv.reg : !hw.inout<i8>
  %narrow = sv.indexed_part_select_inout %bus[%c2 : 1] : !hw.inout<i8>, i32
  %cont = sv.indexed_part_select_inout %bus[%c5 : 1] : !hw.inout<i8>, i32
  sv.always posedge %clk {
    // expected-note @below {{other driver is here}}
    sv.passign %bus, %a : i8
    sv.passign %narrow, %b : i1
  }
  // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
  sv.assign %cont, %c : i1
}

// -----

// Every procedural write under a wide continuous driver is reported.
hw.module @WideContinuousHidesNothing(in %clk: i1, in %a: i8, in %b: i1, in %c: i1) {
  %c2 = hw.constant 2 : i32
  %c5 = hw.constant 5 : i32
  %bus = sv.reg : !hw.inout<i8>
  %p2 = sv.indexed_part_select_inout %bus[%c2 : 1] : !hw.inout<i8>, i32
  %p5 = sv.indexed_part_select_inout %bus[%c5 : 1] : !hw.inout<i8>, i32
  // expected-note @below {{other driver is here}}
  sv.assign %bus, %a : i8
  sv.always posedge %clk {
    // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
    sv.passign %p2, %b : i1
    // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
    sv.passign %p5, %c : i1
  }
}

// -----

//===----------------------------------------------------------------------===//
// sv.var / sv.var_assign
//===----------------------------------------------------------------------===//

// One continuous driver on a whole variable, and independent variables.
// CHECK-LABEL: hw.module @VarWholeVariable
hw.module @VarWholeVariable(in %a: i8, in %b: i8) {
  %v = sv.var : !sv.var<i8>
  %w = sv.var : !sv.var<i8>
  sv.var_assign %v, %a : i8
  sv.var_assign %w, %b : i8
}

// -----

hw.module @VarTwoWholeDrivers(in %a: i8, in %b: i8) {
  %v = sv.var : !sv.var<i8>
  // expected-note @below {{other driver is here}}
  sv.var_assign %v, %a : i8
  // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
  sv.var_assign %v, %b : i8
}

// -----

// Identical values do not matter: the rule counts drivers.
hw.module @VarSameValueTwice(in %a: i1) {
  %v = sv.var : !sv.var<i1>
  // expected-note @below {{other driver is here}}
  sv.var_assign %v, %a : i1
  // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
  sv.var_assign %v, %a : i1
}

// -----

// sv.var_assign drivers and legacy sv.assign drivers on distinct variables
// do not interact.
// CHECK-LABEL: hw.module @VarAndLegacyRoots
hw.module @VarAndLegacyRoots(in %a: i8) {
  %v = sv.var : !sv.var<i8>
  %r = sv.reg : !hw.inout<i8>
  sv.var_assign %v, %a : i8
  sv.assign %r, %a : i8
}
