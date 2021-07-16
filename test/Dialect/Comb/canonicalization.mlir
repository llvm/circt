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

// CHECK-LABEL: @subCst
hw.module @subCst(%a: i4) -> (%o1: i4) {
// CHECK-NEXT: %c-4_i4 = hw.constant -4 : i4
// CHECK-NEXT: %0 = comb.add %a, %c-4_i4 : i4
  %c1 = hw.constant 4 : i4
  %b = comb.sub %a, %c1 : i4
  hw.output %b : i4
}

// Validates that when there is a matching suffix, and prefix, both of them are removed
// appropriately, and strips of an unnecessary Cat where possible.
// CHECK-LABEL: hw.module @compareStrengthReductionRemoveSuffixAndPrefix
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.icmp uge %arg0, %arg1 : i9
// CHECK-NEXT:    hw.output [[RES]] : i1
hw.module @compareStrengthReductionRemoveSuffixAndPrefix(%arg0: i9, %arg1: i9) -> (%o : i1) {
  %0 = comb.concat %arg0, %arg0, %arg1: (i9, i9, i9) -> i18
  %1 = comb.concat %arg0, %arg1, %arg1: (i9, i9, i9) -> i18
  %2 = comb.icmp uge %0, %1 : i18
  hw.output %2 : i1
}

// Validates that comparison strength reduction will retain the concatenation operator
// when there is >1 elements left in one of them, and doens't spuriously remove all non-matching
// suffices
// CHECK-LABEL: hw.module @compareStrengthReductionRetainCat
// CHECK-NEXT:    [[ARG0:%[0-9]+]] = comb.concat %arg0, %arg1 : (i9, i9) -> i18
// CHECK-NEXT:    [[ARG1:%[0-9]+]] = comb.concat %arg1, %arg0 : (i9, i9) -> i18
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.icmp uge [[ARG0]], [[ARG1]] : i18
// CHECK-NEXT:    hw.output [[RES]] : i1
hw.module @compareStrengthReductionRetainCat(%arg0: i9, %arg1: i9) -> (%o : i1) {
  %0 = comb.concat %arg0, %arg0, %arg1 : (i9, i9, i9) -> i27
  %1 = comb.concat %arg0, %arg1, %arg0 : (i9, i9, i9) -> i27
  %2 = comb.icmp uge %0, %1 : i27
  hw.output %2 : i1
}

// Validates that narrowing signed comparisons without stripping the common suffix
// must not pad an additional sign bit.
// CHECK-LABEL: hw.module @compareStrengthSignedCommonSuffix
// CHECK-NEXT:    [[ARG0:%[0-9]+]] = comb.concat %arg0, %arg0 : (i9, i9) -> i18
// CHECK-NEXT:    [[ARG1:%[0-9]+]] = comb.concat %arg1, %arg1 : (i9, i9) -> i18
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.icmp sge [[ARG0]], [[ARG1]] : i18
// CHECK-NEXT:    hw.output [[RES]] : i1
hw.module @compareStrengthSignedCommonSuffix(%arg0: i9, %arg1: i9) -> (%o : i1) {
  %0 = comb.concat %arg0, %arg0, %arg1 : (i9, i9, i9) -> i27
  %1 = comb.concat %arg1, %arg1, %arg1 : (i9, i9, i9) -> i27
  %2 = comb.icmp sge %0, %1 : i27
  hw.output %2 : i1
}

// Validates that narrowing signed comparisons that strips of the common suffix
// must add the sign-bit.
// CHECK-LABEL: hw.module @compareStrengthSignedCommonPrefix
// CHECK-NEXT:    [[SIGNBIT:%[0-9]+]] = comb.extract %arg0 from 2 : (i3) -> i1
// CHECK-NEXT:    [[ARG1:%[0-9]+]] = comb.concat [[SIGNBIT]], %arg1 : (i1, i9) -> i10
// CHECK-NEXT:    [[ARG2:%[0-9]+]] = comb.concat [[SIGNBIT]], %arg2 : (i1, i9) -> i10
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.icmp sge [[ARG1]], [[ARG2]] : i10
// CHECK-NEXT:    hw.output [[RES]] : i1
hw.module @compareStrengthSignedCommonPrefix(%arg0 : i3, %arg1: i9, %arg2: i9) -> (%o : i1) {
  %0 = comb.concat %arg0, %arg1 : (i3, i9) -> i12
  %1 = comb.concat %arg0, %arg2 : (i3, i9) -> i12
  %2 = comb.icmp sge %0, %1 : i12
  hw.output %2 : i1
}

// Validates that narrowing signed comparisons that strips of the common suffix
// must add the sign-bit. The sign bit if the leading common element has a length of 1.
// CHECK-LABEL: hw.module @compareStrengthSignedCommonPrefixNoExtract
// CHECK-NEXT:    [[ARG1:%[0-9]+]] = comb.concat %arg0, %arg2 : (i1, i9) -> i10
// CHECK-NEXT:    [[ARG2:%[0-9]+]] = comb.concat %arg0, %arg3 : (i1, i9) -> i10
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.icmp sge [[ARG1]], [[ARG2]] : i10
// CHECK-NEXT:    hw.output [[RES]] : i1
hw.module @compareStrengthSignedCommonPrefixNoExtract(%arg0 : i1, %arg1 : i3, %arg2: i9, %arg3: i9) -> (%o : i1) {
  %0 = comb.concat %arg0, %arg1, %arg2 : (i1, i3, i9) -> i13
  %1 = comb.concat %arg0, %arg1, %arg3 : (i1, i3, i9) -> i13
  %2 = comb.icmp sge %0, %1 : i13
  hw.output %2 : i1
}

// Validates that cmp(concat(..), concat(...)) that should be simplified to true
// are indeed so.
// CHECK-LABEL: hw.module @compareConcatEliminationTrueCases
// CHECK-NEXT:    %true = hw.constant true
// CHECK-NEXT:    hw.output %true : i1
hw.module @compareConcatEliminationTrueCases(%arg0 : i4, %arg1: i9, %arg2: i7) -> (%o : i1) {
  %0 = comb.concat %arg0, %arg1, %arg2 : (i4, i9, i7) -> i20
  %1 = comb.concat %arg0, %arg1, %arg2 : (i4, i9, i7) -> i20
  %2 = comb.icmp sle %0, %1 : i20
  %3 = comb.icmp sge %0, %1 : i20
  %4 = comb.icmp ule %0, %1 : i20
  %5 = comb.icmp uge %0, %1 : i20
  %6 = comb.icmp  eq %0, %1 : i20
  %o = comb.and %2, %3, %4, %5, %6 : i1
  hw.output %o : i1
}

// Validates cases of cmp(concat(..), concat(...)) that should be simplified to false
// are indeed so.
// CHECK-LABEL: hw.module @compareConcatEliminationFalseCases
// CHECK-NEXT:    %false = hw.constant false
// CHECK-NEXT:    hw.output %false : i1
hw.module @compareConcatEliminationFalseCases(%arg0 : i4, %arg1: i9, %arg2: i7) -> (%o : i1) {
  %0 = comb.concat %arg0, %arg1, %arg2 : (i4, i9, i7) -> i20
  %1 = comb.concat %arg0, %arg1, %arg2 : (i4, i9, i7) -> i20
  %2 = comb.icmp slt %0, %1 : i20
  %3 = comb.icmp sgt %0, %1 : i20
  %4 = comb.icmp ult %0, %1 : i20
  %5 = comb.icmp ugt %0, %1 : i20
  %6 = comb.icmp  ne %0, %1 : i20
  %o = comb.or %2, %3, %4, %5, %6 : i1
  hw.output %o : i1
}

// Validates that extract(cat(a, b, c)) -> cat(b, c) when it aligns with the exact elements, or simply
// a when it is a single full element.
// CHECK-LABEL: hw.module @extractCatAlignWithExactElements
hw.module @extractCatAlignWithExactElements(%arg0: i8, %arg1: i9, %arg2: i10) -> (%o1 : i17, %o2: i19, %o3: i9, %o4: i10) {
  %0 = comb.concat %arg0, %arg1, %arg2 : (i8, i9, i10) -> i27

  // CHECK-NEXT:    [[R0:%.+]] = comb.concat %arg0, %arg1
  %1 = comb.extract %0 from 10 : (i27) -> i17

  // CHECK-NEXT:    [[R1:%.+]] = comb.concat %arg1, %arg2
  %2 = comb.extract %0 from 0 : (i27) -> i19
  %3 = comb.extract %0 from 10 : (i27) -> i9
  %4 = comb.extract %0 from 0 : (i27) -> i10

  // CHECK-NEXT:    hw.output [[R0]], [[R1]], %arg1, %arg2
  hw.output %1, %2, %3, %4 : i17, i19, i9, i10
}

// Validates that extract(cat(a, b, c)) -> cat(extract(b)) when it matches only on a single
// partial element
// CHECK-LABEL: hw.module @extractCatOnSinglePartialElement
hw.module @extractCatOnSinglePartialElement(%arg0: i8, %arg1: i9, %arg2: i10) -> (%o1 : i1, %o2: i1, %o3: i1, %o4: i1) {
  %0 = comb.concat %arg0, %arg1, %arg2 : (i8, i9, i10) -> i27

  // From the first bit position
  // CHECK-NEXT:    [[R0:%.+]] = comb.extract %arg2 from 0 : (i10) -> i1
  %1 = comb.extract %0 from 0 : (i27) -> i1

  // From the last bit position
  // CHECK-NEXT:    [[R1:%.+]] = comb.extract %arg2 from 9 : (i10) -> i1
  %2 = comb.extract %0 from 9 : (i27) -> i1

  // From some middling position
  // CHECK-NEXT:    [[R2:%.+]] = comb.extract %arg2 from 5 : (i10) -> i1
  %3 = comb.extract %0 from 5 : (i27) -> i1

  // From the first bit position on non-first element.
  // CHECK-NEXT:    [[R3:%.]] = comb.extract %arg1 from 0 : (i9) -> i1
  %4 = comb.extract %0 from 10 : (i27) -> i1

  // CHECK-NEXT:    hw.output [[R0]], [[R1]], [[R2]], [[R3]]
  hw.output %1, %2, %3, %4 : i1, i1, i1, i1
}

// Validates that extract(cat(a, b, c)) -> cat(extract(..), .., extract(..))
// containing a mix of full elements and extract elements.
// A few things to look out here:
// - extract is only inserted at elements that require it
// - no zero-elements introduced
// - the order of the elements are correct.
// CHECK-LABEL: hw.module @extractCatOnMultiplePartialElements
hw.module @extractCatOnMultiplePartialElements(%arg0: i8, %arg1: i9, %arg2: i10) -> (%o1 : i11, %o2 : i5) {
  %0 = comb.concat %arg0, %arg1, %arg2 : (i8, i9, i10) -> i27

  // Part of arg0, all of arg1, part of arg2
  // CHECK-NEXT: [[FROMARG2:%.+]] = comb.extract %arg2 from 9 : (i10) -> i1
  // CHECK-NEXT: [[FROMARG0:%.+]] = comb.extract %arg0 from 0 : (i8) -> i1
  // CHECK-NEXT: [[RESULT1:%.+]] = comb.concat [[FROMARG0]], %arg1, [[FROMARG2]] : (i1, i9, i1) -> i11
  %1 = comb.extract %0 from 9 : (i27) -> i11

  // Part of arg1 and part of arg2
  // CHECK-NEXT: [[FROMARG2:%.+]] = comb.extract %arg2 from 9 : (i10) -> i1
  // CHECK-NEXT: [[FROMARG1:%.+]] = comb.extract %arg1 from 0 : (i9) -> i4
  // CHECK-NEXT: [[RESULT2:%.+]] = comb.concat [[FROMARG1]], [[FROMARG2]] : (i4, i1) -> i5
  %2 = comb.extract %0 from 9 : (i27) -> i5

  // CHECK-NEXT: hw.output [[RESULT1:%.+]], [[RESULT2:%.+]]
  hw.output %1, %2 : i11, i5
}

// Validates that addition narrows the operand widths to the width of the
// single extract usage.
// CHECK-LABEL: hw.module @narrowAdditionSingleExtractUse
hw.module @narrowAdditionSingleExtractUse(%x: i8, %y: i8) -> (%z1: i6) {
  // CHECK-NEXT: [[RX:%.+]] = comb.extract %x from 0 : (i8) -> i6
  // CHECK-NEXT: [[RY:%.+]] = comb.extract %y from 0 : (i8) -> i6
  // CHECK-NEXT: [[RESULT:%.+]] = comb.add [[RX]], [[RY]] : i6
  // CHECK-NEXT: hw.output [[RESULT]]

  %false = hw.constant false
  %0 = comb.concat %false, %x : (i1, i8) -> i9
  %1 = comb.concat %false, %y : (i1, i8) -> i9
  %2 = comb.add %0, %1 : i9
  %3 = comb.extract %2 from 0 : (i9) -> i6
  hw.output %3 : i6
}

// Validates that addition narrows to the element itself without an extract
// where possible.
// CHECK-LABEL: hw.module @narrowAdditionToDirectAddition
hw.module @narrowAdditionToDirectAddition(%x: i8, %y: i8) -> (%z1: i8) {
  // CHECK-NEXT: [[RESULT:%.+]] = comb.add %x, %y : i8
  // CHECK-NEXT: hw.output [[RESULT]]

  %false = hw.constant false
  %0 = comb.concat %x, %x : (i8, i8) -> i16
  %1 = comb.concat %y, %y : (i8, i8) -> i16
  %2 = comb.add %0, %1 : i16
  %3 = comb.extract %2 from 0 : (i16) -> i8
  hw.output %3 : i8
}

// Validates that addition narrow to the widest extract
// CHECK-LABEL: hw.module @narrowAdditionToWidestExtract
hw.module @narrowAdditionToWidestExtract(%x: i8, %y: i8) -> (%z1: i3, %z2: i4) {
  // CHECK-NEXT: [[RX:%.+]] = comb.extract %x from 0 : (i8) -> i4
  // CHECK-NEXT: [[RY:%.+]] = comb.extract %y from 0 : (i8) -> i4
  // CHECK-NEXT: [[RESULT2:%.+]] = comb.add [[RX]], [[RY]] : i4
  // CHECK-NEXT: [[RESULT1:%.+]] = comb.extract [[RESULT2]] from 0 : (i4) -> i3
  // CHECK-NEXT: hw.output [[RESULT1]], [[RESULT2]]

  %0 = comb.concat %x, %x : (i8, i8) -> i9
  %1 = comb.concat %y, %y : (i8, i8) -> i9
  %2 = comb.add %0, %1 : i9
  %3 = comb.extract %2 from 0 : (i9) -> i3
  %4 = comb.extract %2 from 0 : (i9) -> i4
  hw.output %3, %4 : i3, i4
}

// Validates that addition narrow to the widest extract
// CHECK-LABEL: hw.module @narrowAdditionStripLeadingZero
hw.module @narrowAdditionStripLeadingZero(%x: i8, %y: i8) -> (%z: i8) {
  // CHECK-NEXT: [[RESULT:%.+]] = comb.add %x, %y : i8
  // CHECK-NEXT: hw.output [[RESULT]]

  %false = hw.constant false
  %0 = comb.concat %false, %x : (i1, i8) -> i9
  %1 = comb.concat %false, %y : (i1, i8) -> i9
  %2 = comb.add %0, %1 : i9
  %3 = comb.extract %2 from 0 : (i9) -> i8
  hw.output %3 : i8
}

// Validates that addition narrowing does not happen when the width of the
// largest use is as wide as the addition result itself.
// CHECK-LABEL: hw.module @narrowAdditionRetainOriginal
hw.module @narrowAdditionRetainOriginal(%x: i8, %y: i8) -> (%z0: i9, %z1: i8) {
  // CHECK-NEXT: false = hw.constant false
  // CHECK-NEXT: %0 = comb.concat %false, %x : (i1, i8) -> i9
  // CHECK-NEXT: %1 = comb.concat %false, %y : (i1, i8) -> i9
  // CHECK-NEXT: %2 = comb.add %0, %1 : i9
  // CHECK-NEXT: %3 = comb.extract %2 from 0 : (i9) -> i8
  // CHECK-NEXT: hw.output %2, %3 : i9, i8

  %false = hw.constant false
  %0 = comb.concat %false, %x : (i1, i8) -> i9
  %1 = comb.concat %false, %y : (i1, i8) -> i9
  %2 = comb.add %0, %1 : i9
  %3 = comb.extract %2 from 0 : (i9) -> i8
  hw.output %2, %3 : i9, i8
}

// Validates that addition narrowing retains the lower bits when not extracting from
// zero.
// CHECK-LABEL: hw.module @narrowAdditionExtractFromNoneZero
hw.module @narrowAdditionExtractFromNoneZero(%x: i8, %y: i8) -> (%z0: i4) {
  // CHECK-NEXT: [[RX:%.+]] = comb.extract %x from 0 : (i8) -> i5
  // CHECK-NEXT: [[RY:%.+]] = comb.extract %y from 0 : (i8) -> i5
  // CHECK-NEXT: [[ADD:%.+]] = comb.add [[RX]], [[RY]] : i5
  // CHECK-NEXT: [[RET:%.+]] = comb.extract [[ADD]] from 1 : (i5) -> i4
  // CHECK-NEXT: hw.output [[RET]]

  %0 = comb.add %x, %y : i8
  %1 = comb.extract %0 from 1 : (i8) -> i4
  hw.output %1 : i4
}

// Validates that subtraction narrowing retains the lower bits when not extracting from
// zero.
// CHECK-LABEL: hw.module @narrowSubExtractFromNoneZero
hw.module @narrowSubExtractFromNoneZero(%x: i8, %y: i8) -> (%z0: i4) {
  // CHECK-NEXT: [[RX:%.+]] = comb.extract %x from 0 : (i8) -> i5
  // CHECK-NEXT: [[RY:%.+]] = comb.extract %y from 0 : (i8) -> i5
  // CHECK-NEXT: [[ADD:%.+]] = comb.sub [[RX]], [[RY]] : i5
  // CHECK-NEXT: [[RET:%.+]] = comb.extract [[ADD]] from 1 : (i5) -> i4
  // CHECK-NEXT: hw.output [[RET]]

  %0 = comb.sub %x, %y : i8
  %1 = comb.extract %0 from 1 : (i8) -> i4
  hw.output %1 : i4
}

// Validates that bitwise operation does not retain the lower bit when extracting from
// non-zero.
// CHECK-LABEL: hw.module @narrowBitwiseOpsExtractFromNoneZero
hw.module @narrowBitwiseOpsExtractFromNoneZero(%a: i8, %b: i8, %c: i8, %d: i1) -> (%w: i4, %x: i4, %y: i4, %z: i4) {
  // CHECK-NEXT: [[RA:%.+]] = comb.extract %a from 1 : (i8) -> i4
  // CHECK-NEXT: [[RB:%.+]] = comb.extract %b from 1 : (i8) -> i4
  // CHECK-NEXT: [[RC:%.+]] = comb.extract %c from 1 : (i8) -> i4
  // CHECK-NEXT: [[AND:%.+]] = comb.and [[RA]], [[RB]], [[RC]] : i4
  %0 = comb.and %a, %b, %c : i8
  %1 = comb.extract %0 from 1 : (i8) -> i4

  // CHECK-NEXT: [[RA:%.+]] = comb.extract %a from 1 : (i8) -> i4
  // CHECK-NEXT: [[RB:%.+]] = comb.extract %b from 1 : (i8) -> i4
  // CHECK-NEXT: [[RC:%.+]] = comb.extract %c from 1 : (i8) -> i4
  // CHECK-NEXT: [[OR:%.+]] = comb.or [[RA]], [[RB]], [[RC]] : i4
  %2 = comb.or %a, %b, %c : i8
  %3 = comb.extract %2 from 1 : (i8) -> i4

  // CHECK-NEXT: [[RA:%.+]] = comb.extract %a from 1 : (i8) -> i4
  // CHECK-NEXT: [[RB:%.+]] = comb.extract %b from 1 : (i8) -> i4
  // CHECK-NEXT: [[RC:%.+]] = comb.extract %c from 1 : (i8) -> i4
  // CHECK-NEXT: [[XOR:%.+]] = comb.xor [[RA]], [[RB]], [[RC]] : i4
  %4 = comb.xor %a, %b, %c : i8
  %5 = comb.extract %4 from 1 : (i8) -> i4

  // CHECK-NEXT: [[RA:%.+]] = comb.extract %a from 1 : (i8) -> i4
  // CHECK-NEXT: [[RB:%.+]] = comb.extract %b from 1 : (i8) -> i4
  // CHECK-NEXT: [[MUX:%.+]] = comb.mux %d, [[RA]], [[RB]] : i4
  %6 = comb.mux %d, %a, %b : i8
  %7 = comb.extract %6 from 1 : (i8) -> i4

  // CHECK-NEXT: hw.output [[AND]], [[OR]], [[XOR]], [[MUX]]
  hw.output %1, %3, %5, %7 : i4, i4, i4, i4
}
