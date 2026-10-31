// RUN: circt-opt --sv-check-var-drivers --split-input-file --verify-diagnostics %s
//
//
// NOTE: variables are currently declared with sv.reg and driven with
// sv.assign, because the inout projection ops and the procedural assigns do
// not accept !sv.var yet. Once they do, convert these tests to sv.var and
// sv.var_assign.

// -----

hw.module @TwoDriversOfVar(in %a: i8, in %b: i8) {
  %v = sv.var : !sv.var<i8>
  // expected-note @below {{other driver is here}}
  sv.var_assign %v, %a : i8
  // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
  sv.var_assign %v, %b : i8
}

// -----

hw.module @ContinuousThenNonblocking(in %clk: i1, in %a: i8, in %b: i8) {
  %v = sv.reg : !hw.inout<i8>
  // expected-note @below {{other driver is here}}
  sv.assign %v, %a : i8
  sv.always posedge %clk {
    // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
    sv.passign %v, %b : i8
  }
}

// -----

hw.module @ContinuousThenBlocking(in %a: i8, in %b: i8) {
  %v = sv.reg : !hw.inout<i8>
  // expected-note @below {{other driver is here}}
  sv.assign %v, %a : i8
  sv.alwayscomb {
    // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
    sv.bpassign %v, %b : i8
  }
}

// -----

hw.module @NonblockingThenContinuous(in %clk: i1, in %a: i8, in %b: i8) {
  %v = sv.reg : !hw.inout<i8>
  sv.always posedge %clk {
    // expected-note @below {{other driver is here}}
    sv.passign %v, %a : i8
  }
  // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
  sv.assign %v, %b : i8
}

// -----

hw.module @SlicesSharingOneBit(in %x: i4, in %y: i4) {
  %c0 = hw.constant 0 : i32
  %c3 = hw.constant 3 : i32
  %bus = sv.reg : !hw.inout<i8>
  %lo = sv.indexed_part_select_inout %bus[%c0 : 4] : !hw.inout<i8>, i32
  %mid = sv.indexed_part_select_inout %bus[%c3 : 4] : !hw.inout<i8>, i32
  // expected-note @below {{other driver is here}}
  sv.assign %lo, %x : i4
  // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
  sv.assign %mid, %y : i4
}

// -----

hw.module @DecrementingSelectOverlap(in %x: i4, in %y: i4) {
  %c4 = hw.constant 4 : i32
  %c7 = hw.constant 7 : i32
  %bus = sv.reg : !hw.inout<i8>
  %dec = sv.indexed_part_select_inout %bus[%c7 decrement : 4] : !hw.inout<i8>, i32
  %inc = sv.indexed_part_select_inout %bus[%c4 : 4] : !hw.inout<i8>, i32
  // expected-note @below {{other driver is here}}
  sv.assign %dec, %x : i4
  // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
  sv.assign %inc, %y : i4
}

// -----

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

hw.module @SliceAndStructField(in %x: i4, in %y: i8) {
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

hw.module @ArrayPartSelectCountsElements(in %x: !hw.array<2xi8>, in %y: i8) {
  %c0 = hw.constant 0 : i2
  %c1 = hw.constant 1 : i2
  %arr = sv.reg : !hw.inout<array<4xi8>>
  %sel = sv.indexed_part_select_inout %arr[%c0 : 2] : !hw.inout<array<4xi8>>, i2
  %e1 = sv.array_index_inout %arr[%c1] : !hw.inout<array<4xi8>>, i2
  // expected-note @below {{other driver is here}}
  sv.assign %sel, %x : !hw.array<2xi8>
  // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
  sv.assign %e1, %y : i8
}

// -----

hw.module @SpecProceduralSliceOverlapsContinuousBit(in %clk: i1, in %x: i1, in %y: i5) {
  %c3 = hw.constant 3 : i32
  %abc = sv.reg : !hw.inout<struct<A: i8, B: i8, C: i8>>
  %A = sv.struct_field_inout %abc["A"] : !hw.inout<struct<A: i8, B: i8, C: i8>>
  %a3 = sv.indexed_part_select_inout %A[%c3 : 1] : !hw.inout<i8>, i32
  %a73 = sv.indexed_part_select_inout %A[%c3 : 5] : !hw.inout<i8>, i32
  // expected-note @below {{other driver is here}}
  sv.assign %a3, %x : i1
  sv.always posedge %clk {
    // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
    sv.passign %a73, %y : i5
  }
}

// -----

// `m[1][i]` writes row 1, which contains `m[1][0]`.
hw.module @DynamicIndexOverlapsRow(in %clk: i1, in %i: i3, in %a: i8, in %b: i8) {
  %c0 = hw.constant 0 : i3
  %c1 = hw.constant 1 : i3
  %m = sv.reg : !hw.inout<array<5xarray<5xi8>>>
  %row1 = sv.array_index_inout %m[%c1] : !hw.inout<array<5xarray<5xi8>>>, i3
  %dyn = sv.array_index_inout %row1[%i] : !hw.inout<array<5xi8>>, i3
  %fix = sv.array_index_inout %row1[%c0] : !hw.inout<array<5xi8>>, i3
  sv.always posedge %clk {
    // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
    sv.passign %dyn, %a : i8
  }
  // expected-note @below {{other driver is here}}
  sv.assign %fix, %b : i8
}

// -----

// `m[i][1]` writes all of m, which contains `m[2][0]`. The write starts at the
// first bit, so it sorts first and the continuous driver is reported.
hw.module @LeadingDynamicIndexWritesEverything(in %clk: i1, in %i: i3, in %a: i8, in %b: i8) {
  %c0 = hw.constant 0 : i3
  %c1 = hw.constant 1 : i3
  %c2 = hw.constant 2 : i3
  %m = sv.reg : !hw.inout<array<5xarray<5xi8>>>
  %rowi = sv.array_index_inout %m[%i] : !hw.inout<array<5xarray<5xi8>>>, i3
  %row2 = sv.array_index_inout %m[%c2] : !hw.inout<array<5xarray<5xi8>>>, i3
  %dyn = sv.array_index_inout %rowi[%c1] : !hw.inout<array<5xi8>>, i3
  %fix = sv.array_index_inout %row2[%c0] : !hw.inout<array<5xi8>>, i3
  sv.always posedge %clk {
    // expected-note @below {{other driver is here}}
    sv.passign %dyn, %a : i8
  }
  // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
  sv.assign %fix, %b : i8
}

// -----

// A dynamic index in the middle drops the selects below it: the prefix of
// `m[1][i][2]` is `m[1]`, which contains `m[1][0][0]`.
hw.module @MiddleDynamicIndexOverlapsRow(in %clk: i1, in %i: i3, in %a: i1, in %b: i1) {
  %c0 = hw.constant 0 : i3
  %c1 = hw.constant 1 : i3
  %c2 = hw.constant 2 : i3
  %m = sv.reg : !hw.inout<array<5xarray<5xarray<5xi1>>>>
  %row1 = sv.array_index_inout %m[%c1] : !hw.inout<array<5xarray<5xarray<5xi1>>>>, i3
  %rowi = sv.array_index_inout %row1[%i] : !hw.inout<array<5xarray<5xi1>>>, i3
  %dyn = sv.array_index_inout %rowi[%c2] : !hw.inout<array<5xi1>>, i3
  %row0 = sv.array_index_inout %row1[%c0] : !hw.inout<array<5xarray<5xi1>>>, i3
  %fix = sv.array_index_inout %row0[%c0] : !hw.inout<array<5xi1>>, i3
  sv.always posedge %clk {
    // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
    sv.passign %dyn, %a : i1
  }
  // expected-note @below {{other driver is here}}
  sv.assign %fix, %b : i1
}

// -----

// A part-select with a dynamic base writes the whole variable.
hw.module @DynamicSliceWritesEverything(in %i: i32, in %x: i4, in %y: i4) {
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

hw.module @WideProceduralBeforeNarrowOne(in %clk: i1, in %a: i8, in %b: i1, in %c: i1) {
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

hw.module @WideContinuousOverTwoProceduralOnes(in %clk: i1, in %a: i8, in %b: i1, in %c: i1) {
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

hw.module @LaterWiderDriverIsRemembered(in %clk: i1, in %a: i1, in %b: i8, in %c: i1) {
  %c0 = hw.constant 0 : i32
  %c5 = hw.constant 5 : i32
  %bus = sv.reg : !hw.inout<i8>
  %bit0 = sv.indexed_part_select_inout %bus[%c0 : 1] : !hw.inout<i8>, i32
  %bit5 = sv.indexed_part_select_inout %bus[%c5 : 1] : !hw.inout<i8>, i32
  sv.always posedge %clk {
    sv.passign %bit0, %a : i1
    // expected-note @below {{other driver is here}}
    sv.passign %bus, %b : i8
  }
  // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
  sv.assign %bit5, %c : i1
}

// -----

hw.module @LaterWiderContinuousDriverIsRemembered(in %clk: i1, in %x: i1, in %y: i7, in %p: i1) {
  %c0 = hw.constant 0 : i32
  %c1 = hw.constant 1 : i32
  %c4 = hw.constant 4 : i32
  %bus = sv.reg : !hw.inout<i8>
  %bit0 = sv.indexed_part_select_inout %bus[%c0 : 1] : !hw.inout<i8>, i32
  %upper = sv.indexed_part_select_inout %bus[%c1 : 7] : !hw.inout<i8>, i32
  %bit4 = sv.indexed_part_select_inout %bus[%c4 : 1] : !hw.inout<i8>, i32
  sv.assign %bit0, %x : i1
  // expected-note @below {{other driver is here}}
  sv.assign %upper, %y : i7
  sv.always posedge %clk {
    // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
    sv.passign %bit4, %p : i1
  }
}

// -----

hw.module @ConflictOfOneVariable(in %a: i8, in %b: i8) {
  %v = sv.reg : !hw.inout<i8>
  %w = sv.reg : !hw.inout<i8>
  // expected-note @below {{other driver is here}}
  sv.assign %v, %a : i8
  sv.assign %w, %a : i8
  // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
  sv.assign %v, %b : i8
}

// -----

hw.module @ConflictsOfTwoVariables(in %a: i8, in %b: i8) {
  %v = sv.reg : !hw.inout<i8>
  %w = sv.reg : !hw.inout<i8>
  // expected-note @below {{other driver is here}}
  sv.assign %v, %a : i8
  // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
  sv.assign %v, %b : i8
  // expected-note @below {{other driver is here}}
  sv.assign %w, %a : i8
  // expected-error @below {{multiple continuous and/or conflicting procedural drivers}}
  sv.assign %w, %b : i8
}
