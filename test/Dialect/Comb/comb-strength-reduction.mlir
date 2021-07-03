// RUN: circt-opt %s -simple-canonicalizer -comb-strength-reduction | FileCheck %s

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
