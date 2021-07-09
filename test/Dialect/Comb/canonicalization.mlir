// RUN: circt-opt %s -simple-canonicalizer | FileCheck %s

// CHECK-LABEL: @narrowMux
hw.module @narrowMux(%a: i8, %b: i8, %c: i1) -> (%o: i4) {
// CHECK-NEXT: %0 = comb.extract %a from 1 : (i8) -> i4
// CHECK-NEXT: %1 = comb.extract %b from 1 : (i8) -> i4
// CHECK-NEXT: %2 = comb.mux %c, %0, %1 : i4
  %0 = comb.mux %c, %a, %b : i8
  %1 = comb.extract %0 from 1 : (i8) -> i4
  hw.output %1 : i4
}

// CHECK-LABEL: @notMux
hw.module @notMux(%a: i4, %b: i4, %c: i1) -> (%o: i4) {
// CHECK-NEXT: comb.mux %c, %b, %a : i4
  %c1 = hw.constant 1 : i1
  %0 = comb.xor %c, %c1 : i1
  %1 = comb.mux %0, %a, %b : i4
  hw.output %1 : i4
}

// CHECK-LABEL: @notNot
hw.module @notNot(%a: i1) -> (%o: i1) {
// CHECK-NEXT: hw.output %a
  %c1 = hw.constant 1 : i1
  %0 = comb.xor %a, %c1 : i1
  %1 = comb.xor %0, %c1 : i1
  hw.output %1 : i1
}


// CHECK-LABEL: @andCancel
hw.module @andCancel(%a: i4, %b : i4) -> (%o1: i4, %o2: i4) {
// CHECK-NEXT: hw.constant 0 : i4
// CHECK-NEXT: hw.output %c0_i4, %c0_i4 : i4, i4
  %c1 = hw.constant 15 : i4
  %anot = comb.xor %a, %c1 : i4
  %1 = comb.and %a, %anot : i4
  %2 = comb.and %b, %a, %b, %anot, %b : i4
  hw.output %1, %2 : i4, i4
}


// CHECK-LABEL: hw.module @andDedup1(%arg0: i7, %arg1: i7) -> (i7) {
hw.module @andDedup1(%arg0: i7, %arg1: i7) -> (i7) {
// CHECK-NEXT:    %0 = comb.and %arg0, %arg1 : i7
// CHECK-NEXT:    hw.output %0 : i7
  %0 = comb.and %arg0    : i7
  %1 = comb.and %0, %arg1: i7
  hw.output %1 : i7
}

// CHECK-LABEL: hw.module @andDedup2(%arg0: i7, %arg1: i7) -> (i7) {
hw.module @andDedup2(%arg0: i7, %arg1: i7) -> (i7) {
// CHECK-NEXT:    %0 = comb.and %arg0, %arg1 : i7
// CHECK-NEXT:    hw.output %0 : i7
  %0 = comb.and %arg0, %arg0: i7
  %1 = comb.and %0, %arg1: i7
  hw.output %1 : i7
}

// CHECK-LABEL: hw.module @andDedupLong(%arg0: i7, %arg1: i7, %arg2: i7) -> (i7) {
hw.module @andDedupLong(%arg0: i7, %arg1: i7, %arg2: i7) -> (i7) {
// CHECK-NEXT:    %0 = comb.and %arg0, %arg1, %arg2 : i7
// CHECK-NEXT:    hw.output %0 : i7
  %0 = comb.and %arg0, %arg1, %arg2, %arg0: i7
  hw.output %0 : i7
}

// CHECK-LABEL: @extractNested
hw.module @extractNested(%0: i5) -> (%o1 : i1) {
// Multiple layers of nested extract is a weak evidence that the cannonicalization
// operates recursively.
// CHECK-NEXT: %0 = comb.extract %arg0 from 4 : (i5) -> i1
  %1 = comb.extract %0 from 1 : (i5) -> i4
  %2 = comb.extract %1 from 2 : (i4) -> i2
  %3 = comb.extract %2 from 1 : (i2) -> i1
  hw.output %3 : i1
}

// CHECK-LABEL: @flattenMuxTrue
hw.module @flattenMuxTrue(%arg0: i1, %arg1: i8, %arg2: i8, %arg3: i8, %arg4 : i8) -> (%o1 : i8) {
// CHECK-NEXT:    [[RET:%[0-9]+]] = comb.mux %arg0, %arg1, %arg4
// CHECK-NEXT:    hw.output [[RET]]
  %0 = comb.mux %arg0, %arg1, %arg2 : i8
  %1 = comb.mux %arg0, %0   , %arg3 : i8
  %2 = comb.mux %arg0, %1   , %arg4 : i8
  hw.output %2 : i8
}

// CHECK-LABEL: @flattenMuxFalse
hw.module @flattenMuxFalse(%arg0: i1, %arg1: i8, %arg2: i8, %arg3: i8, %arg4 : i8) -> (%o1 : i8) {
// CHECK-NEXT:    [[RET:%[0-9]+]] = comb.mux %arg0, %arg4, %arg2
// CHECK-NEXT:    hw.output [[RET]]
  %0 = comb.mux %arg0, %arg1, %arg2 : i8
  %1 = comb.mux %arg0, %arg3, %0    : i8
  %2 = comb.mux %arg0, %arg4, %1    : i8
  hw.output %2 : i8
}

// CHECK-LABEL: @flattenMuxMixed
hw.module @flattenMuxMixed(%arg0: i1, %arg1: i8, %arg2: i8, %arg3: i8, %arg4 : i8) -> (%o1 : i8) {
// CHECK-NEXT:    [[RET:%[0-9]+]] = comb.mux %arg0, %arg1, %arg4
// CHECK-NEXT:    hw.output [[RET]]
  %0 = comb.mux %arg0, %arg1, %arg2 : i8
  %1 = comb.mux %arg0, %arg3, %arg4 : i8
  %2 = comb.mux %arg0, %0   , %1    : i8
  hw.output %2 : i8
}

// CHECK-LABEL: @flattenNotOnDifferentCond
hw.module @flattenNotOnDifferentCond(%arg0: i1, %arg1: i1, %arg2: i1, %arg3: i8, %arg4 : i8) -> (%o1 : i8) {
// CHECK-NEXT:    %0 = comb.mux %arg0, %arg3, %arg4 : i8
// CHECK-NEXT:    %1 = comb.mux %arg1, %0, %arg4 : i8
// CHECK-NEXT:    %2 = comb.mux %arg2, %1, %arg4 : i8
// CHECK-NEXT:    hw.output %2 : i8
  %0 = comb.mux %arg0, %arg3, %arg4 : i8
  %1 = comb.mux %arg1, %0,    %arg4 : i8
  %2 = comb.mux %arg2, %1,    %arg4 : i8
  hw.output %2 : i8
}
