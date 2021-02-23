// RUN: circt-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: rtl.module @extract_noop(%arg0: i3) -> (i3) {
// CHECK-NEXT:    rtl.output %arg0

rtl.module @extract_noop(%arg0: i3) -> (i3) {
  %x = comb.extract %arg0 from 0 : (i3) -> i3
  rtl.output %x : i3
}

// Constant Folding

// CHECK-LABEL: rtl.module @extract_cstfold() -> (i3) {
// CHECK-NEXT:    %c-3_i3 = comb.constant -3 : i3
// CHECK-NEXT:    rtl.output  %c-3_i3

rtl.module @extract_cstfold() -> (i3) {
  %c42_i12 = comb.constant 42 : i12
  %x = comb.extract %c42_i12 from 3 : (i12) -> i3
  rtl.output %x : i3
}

// CHECK-LABEL: rtl.module @and_cstfold(%arg0: i7) -> (i7) {
// CHECK-NEXT:    %c1_i7 = comb.constant 1 : i7
// CHECK-NEXT:    %0 = comb.and %arg0, %c1_i7 : i7
// CHECK-NEXT:    rtl.output %0 : i7

rtl.module @and_cstfold(%arg0: i7) -> (i7) {
  %c11_i7 = comb.constant 11 : i7
  %c5_i7 = comb.constant 5 : i7
  %0 = comb.and %arg0, %c11_i7, %c5_i7 : i7
  rtl.output %0 : i7
}

// CHECK-LABEL: rtl.module @or_cstfold(%arg0: i7) -> (i7) {
// CHECK-NEXT:    %c15_i7 = comb.constant 15 : i7
// CHECK-NEXT:    %0 = comb.or %arg0, %c15_i7 : i7
// CHECK-NEXT:    rtl.output %0 : i7

rtl.module @or_cstfold(%arg0: i7) -> (i7) {
  %c11_i7 = comb.constant 11 : i7
  %c5_i7 = comb.constant 5 : i7
  %0 = comb.or %arg0, %c11_i7, %c5_i7 : i7
  rtl.output %0 : i7
}

// CHECK-LABEL: rtl.module @xor_cstfold(%arg0: i7) -> (i7) {
// CHECK-NEXT:    %c14_i7 = comb.constant 14 : i7
// CHECK-NEXT:    %0 = comb.xor %arg0, %c14_i7 : i7
// CHECK-NEXT:    rtl.output %0 : i7

rtl.module @xor_cstfold(%arg0: i7) -> (i7) {
  %c11_i7 = comb.constant 11 : i7
  %c5_i7 = comb.constant 5 : i7
  %0 = comb.xor %arg0, %c11_i7, %c5_i7 : i7
  rtl.output %0 : i7
}

// CHECK-LABEL: rtl.module @merge_fold
// CHECK-NEXT:    %0 = comb.merge %arg0, %arg0, %arg1 : i7
// CHECK-NEXT:    rtl.output %arg0, %arg0, %0 : i7, i7, i7
rtl.module @merge_fold(%arg0: i7, %arg1: i7) -> (i7, i7, i7) {
  %a = comb.merge %arg0 : i7
  %b = comb.merge %arg0, %arg0, %arg0 : i7
  %c = comb.merge %arg0, %arg0, %arg1 : i7
  rtl.output %a, %b, %c: i7, i7, i7
}

// CHECK-LABEL: rtl.module @add_cstfold(%arg0: i7) -> (i7) {
// CHECK-NEXT:    %c15_i7 = comb.constant 15 : i7
// CHECK-NEXT:    %0 = comb.add %arg0, %c15_i7 : i7
// CHECK-NEXT:    rtl.output %0 : i7
rtl.module @add_cstfold(%arg0: i7) -> (i7) {
  %c10_i7 = comb.constant 10 : i7
  %c5_i7 = comb.constant 5 : i7
  %0 = comb.add %arg0, %c10_i7, %c5_i7 : i7
  rtl.output %0 : i7
}

// CHECK-LABEL: rtl.module @mul_cstfold(%arg0: i7) -> (i7) {
// CHECK-NEXT:    %c15_i7 = comb.constant 15 : i7
// CHECK-NEXT:    %0 = comb.mul %arg0, %c15_i7 : i7
// CHECK-NEXT:    rtl.output %0 : i7
rtl.module @mul_cstfold(%arg0: i7) -> (i7) {
  %c3_i7 = comb.constant 3 : i7
  %c5_i7 = comb.constant 5 : i7
  %0 = comb.mul %arg0, %c3_i7, %c5_i7 : i7
  rtl.output %0 : i7
}

// CHECK-LABEL: rtl.module @variadic_noop(%arg0: i11) -> (i11) {
// CHECK-NEXT:    rtl.output %arg0

rtl.module @variadic_noop(%arg0: i11) -> (i11) {
  %0 = comb.and %arg0 : i11
  %1 = comb.or  %0 : i11
  %2 = comb.xor %1 : i11
  %3 = comb.add %2 : i11
  %4 = comb.mul %3 : i11
  rtl.output %4 : i11
}

// CHECK-LABEL: rtl.module @and_annulment0(%arg0: i11, %arg1: i11) -> (i11) {
// CHECK-NEXT:    %c0_i11 = comb.constant 0 : i11
// CHECK-NEXT:    rtl.output %c0_i11

rtl.module @and_annulment0(%arg0: i11, %arg1: i11) -> (i11) {
  %c0_i11 = comb.constant 0 : i11
  %0 = comb.and %arg0, %arg1, %c0_i11 : i11
  rtl.output %0 : i11
}

// CHECK-LABEL: rtl.module @and_annulment1(%arg0: i7)
// CHECK-NEXT:    %c0_i7 = comb.constant 0 : i7
// CHECK-NEXT:    rtl.output %c0_i7

rtl.module @and_annulment1(%arg0: i7) -> (i7) {
  %c1_i7 = comb.constant 1 : i7
  %c2_i7 = comb.constant 2 : i7
  %c4_i7 = comb.constant 4 : i7
  %0 = comb.and %arg0, %c1_i7, %c2_i7, %c4_i7: i7
  rtl.output %0 : i7
}

// CHECK-LABEL: rtl.module @or_annulment0(%arg0: i11) -> (i11) {
// CHECK-NEXT:    %c-1_i11 = comb.constant -1 : i11
// CHECK-NEXT:    rtl.output %c-1_i11

rtl.module @or_annulment0(%arg0: i11) -> (i11) {
  %c-1_i11 = comb.constant -1 : i11
  %0 = comb.or %arg0, %arg0, %arg0, %c-1_i11 : i11
  rtl.output %0 : i11
}

// CHECK-LABEL: rtl.module @or_annulment1(%arg0: i3)
// CHECK-NEXT:    %c-1_i3 = comb.constant -1 : i3
// CHECK-NEXT:    rtl.output %c-1_i3

rtl.module @or_annulment1(%arg0: i3) -> (i3) {
  %c1_i3 = comb.constant 1 : i3
  %c2_i3 = comb.constant 2 : i3
  %c4_i3 = comb.constant 4 : i3
  %0 = comb.or %arg0, %c1_i3, %c2_i3, %c4_i3: i3
  rtl.output %0 : i3
}

// CHECK-LABEL: rtl.module @mul_annulment(%arg0: i11, %arg1: i11, %arg2: i11) -> (i11) {
// CHECK-NEXT:    %c0_i11 = comb.constant 0 : i11
// CHECK-NEXT:    rtl.output %c0_i11

rtl.module @mul_annulment(%arg0: i11, %arg1: i11, %arg2: i11) -> (i11) {
  %c0_i11 = comb.constant 0 : i11
  %0 = comb.mul %arg0, %c0_i11, %arg1 : i11
  rtl.output %0 : i11
}

// CHECK-LABEL: rtl.module @mul_overflow(%arg0: i2) -> (i2) {
// CHECK-NEXT:    %c0_i2 = comb.constant 0 : i2
// CHECK-NEXT:    rtl.output %c0_i2

rtl.module @mul_overflow(%arg0: i2) -> (i2) {
  %c2_i2 = comb.constant 2 : i2
  %0 = comb.mul %arg0, %c2_i2, %c2_i2 : i2
  rtl.output %0 : i2
}

// Flattening

// CHECK-LABEL: rtl.module @and_flatten_in_back(%arg0: i7, %arg1: i7, %arg2: i7) -> (i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.and %arg0, %arg1, %arg2 : i7
// CHECK-NEXT:    rtl.output [[RES]] : i7

rtl.module @and_flatten_in_back(%arg0: i7, %arg1: i7, %arg2: i7) -> (i7) {
  %and0 = comb.and %arg1, %arg2 : i7
  %0 = comb.and %arg0, %and0 : i7
  rtl.output %0 : i7
}

// CHECK-LABEL: rtl.module @and_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> (i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.and %arg0, %arg1, %arg2, %arg3 : i7
// CHECK-NEXT:    rtl.output [[RES]] : i7

rtl.module @and_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> (i7) {
  %and0 = comb.and %arg1, %arg2 : i7
  %0 = comb.and %arg0, %and0, %arg3 : i7
  rtl.output %0 : i7
}

// CHECK-LABEL: rtl.module @and_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> (i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.and %arg0, %arg1, %arg2 : i7
// CHECK-NEXT:    rtl.output [[RES]] : i7

rtl.module @and_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> (i7) {
  %and0 = comb.and %arg0, %arg1 : i7
  %0 = comb.and %and0, %arg2 : i7
  rtl.output %0 : i7
}

// CHECK-LABEL: rtl.module @or_flatten_in_back(%arg0: i7, %arg1: i7, %arg2: i7) -> (i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.or %arg0, %arg1, %arg2 : i7
// CHECK-NEXT:    rtl.output [[RES]] : i7

rtl.module @or_flatten_in_back(%arg0: i7, %arg1: i7, %arg2: i7) -> (i7) {
  %or0 = comb.or %arg1, %arg2 : i7
  %0 = comb.or %arg0, %or0 : i7
  rtl.output %0 : i7
}

// CHECK-LABEL: rtl.module @or_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> (i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.or %arg0, %arg1, %arg2, %arg3 : i7
// CHECK-NEXT:    rtl.output [[RES]] : i7

rtl.module @or_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> (i7) {
  %or0 = comb.or %arg1, %arg2 : i7
  %0 = comb.or %arg0, %or0, %arg3 : i7
  rtl.output %0 : i7
}

// CHECK-LABEL: rtl.module @or_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> (i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.or %arg0, %arg1, %arg2 : i7
// CHECK-NEXT:    rtl.output [[RES]] : i7

rtl.module @or_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> (i7) {
  %or0 = comb.or %arg0, %arg1 : i7
  %0 = comb.or %or0, %arg2 : i7
  rtl.output %0 : i7
}

// CHECK-LABEL: rtl.module @xor_flatten_in_back(%arg0: i7, %arg1: i7, %arg2: i7) -> (i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.xor %arg0, %arg1, %arg2 : i7
// CHECK-NEXT:    rtl.output [[RES]] : i7

rtl.module @xor_flatten_in_back(%arg0: i7, %arg1: i7, %arg2: i7) -> (i7) {
  %xor0 = comb.xor %arg1, %arg2 : i7
  %0 = comb.xor %arg0, %xor0 : i7
  rtl.output %0 : i7
}

// CHECK-LABEL: rtl.module @xor_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> (i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.xor %arg0, %arg1, %arg2, %arg3 : i7
// CHECK-NEXT:    rtl.output [[RES]] : i7

rtl.module @xor_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> (i7) {
  %xor0 = comb.xor %arg1, %arg2 : i7
  %0 = comb.xor %arg0, %xor0, %arg3 : i7
  rtl.output %0 : i7
}

// CHECK-LABEL: rtl.module @xor_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> (i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.xor %arg0, %arg1, %arg2 : i7
// CHECK-NEXT:    rtl.output [[RES]] : i7

rtl.module @xor_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> (i7) {
  %xor0 = comb.xor %arg0, %arg1 : i7
  %0 = comb.xor %xor0, %arg2 : i7
  rtl.output %0 : i7
}

rtl.module @add_flatten_in_back(%arg0: i7, %arg1: i7, %arg2: i7) -> (i7) {
  %add0 = comb.add %arg1, %arg2 : i7
  %0 = comb.add %arg0, %add0 : i7
  rtl.output %0 : i7
}

// CHECK-LABEL: rtl.module @add_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> (i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.add %arg0, %arg1, %arg2, %arg3 : i7
// CHECK-NEXT:    rtl.output [[RES]] : i7

rtl.module @add_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> (i7) {
  %add0 = comb.add %arg1, %arg2 : i7
  %0 = comb.add %arg0, %add0, %arg3 : i7
  rtl.output %0 : i7
}

// CHECK-LABEL: rtl.module @add_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> (i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.add %arg0, %arg1, %arg2 : i7
// CHECK-NEXT:    rtl.output [[RES]] : i7

rtl.module @add_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> (i7) {
  %add0 = comb.add %arg0, %arg1 : i7
  %0 = comb.add %add0, %arg2 : i7
  rtl.output %0 : i7
}

rtl.module @mul_flatten_in_back(%arg0: i7, %arg1: i7, %arg2: i7) -> (i7) {
  %mul0 = comb.mul %arg1, %arg2 : i7
  %0 = comb.mul %arg0, %mul0 : i7
  rtl.output %0 : i7
}

// CHECK-LABEL: rtl.module @mul_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> (i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.mul %arg0, %arg1, %arg2, %arg3 : i7
// CHECK-NEXT:    rtl.output [[RES]] : i7

rtl.module @mul_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> (i7) {
  %mul0 = comb.mul %arg1, %arg2 : i7
  %0 = comb.mul %arg0, %mul0, %arg3 : i7
  rtl.output %0 : i7
}

// CHECK-LABEL: rtl.module @mul_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> (i7) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.mul %arg0, %arg1, %arg2 : i7
// CHECK-NEXT:    rtl.output [[RES]] : i7

rtl.module @mul_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> (i7) {
  %mul0 = comb.mul %arg0, %arg1 : i7
  %0 = comb.mul %mul0, %arg2 : i7
  rtl.output %0 : i7
}

// Identities

// CHECK-LABEL: rtl.module @and_identity(%arg0: i11, %arg1: i11) -> (i11) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.and %arg0, %arg1
// CHECK-NEXT:    rtl.output [[RES]]

rtl.module @and_identity(%arg0: i11, %arg1: i11) -> (i11) {
  %c-1_i11 = comb.constant -1 : i11
  %0 = comb.and %c-1_i11, %arg0, %arg1, %c-1_i11 : i11
  rtl.output %0 : i11
}

// CHECK-LABEL: rtl.module @or_identity(%arg0: i11, %arg1: i11) -> (i11) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.or %arg0, %arg1
// CHECK-NEXT:    rtl.output [[RES]]

rtl.module @or_identity(%arg0: i11, %arg1: i11) -> (i11) {
  %c0_i11 = comb.constant 0 : i11
  %0 = comb.or %arg0, %c0_i11, %arg1 : i11
  rtl.output %0 : i11
}

// CHECK-LABEL: rtl.module @xor_identity(%arg0: i11, %arg1: i11) -> (i11) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.xor %arg1, %arg0
// CHECK-NEXT:    rtl.output [[RES]]

rtl.module @xor_identity(%arg0: i11, %arg1: i11) -> (i11) {
  %c0_i11 = comb.constant 0 : i11
  %0 = comb.xor %c0_i11, %arg1, %arg0 : i11
  rtl.output %0 : i11
}

// CHECK-LABEL: rtl.module @add_identity(%arg0: i11, %arg1: i11) -> (i11) {
// CHECK-NEXT:   [[RES:%[0-9]+]] = comb.add %arg0, %arg1
// CHECK-NEXT:    rtl.output [[RES]]

rtl.module @add_identity(%arg0: i11, %arg1: i11) -> (i11) {
  %c0_i11 = comb.constant 0 : i11
  %0 = comb.add %arg0, %c0_i11, %arg1 : i11
  rtl.output %0 : i11
}

// CHECK-LABEL: rtl.module @mul_identity(%arg0: i11, %arg1: i11) -> (i11) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.mul %arg0, %arg1
// CHECK-NEXT:    rtl.output [[RES]]

rtl.module @mul_identity(%arg0: i11, %arg1: i11) -> (i11) {
  %c1_i11 = comb.constant 1 : i11
  %0 = comb.mul %arg0, %c1_i11, %arg1 : i11
  rtl.output %0 : i11
}

// Idempotency

// CHECK-LABEL: rtl.module @and_idempotent(%arg0: i11, %arg1: i11) -> (i11) {
// CHECK-NEXT:    %c9_i11 = comb.constant 9 : i11
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.and %arg0, %arg1, %c9_i11
// CHECK-NEXT:    rtl.output [[RES]]

rtl.module @and_idempotent(%arg0: i11, %arg1 : i11) -> (i11) {
  %c9_i11 = comb.constant 9 : i11
  %0 = comb.and %arg0, %arg1, %c9_i11, %c9_i11 : i11
  rtl.output %0 : i11
}

// CHECK-LABEL: rtl.module @or_idempotent(%arg0: i11, %arg1: i11) -> (i11) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.or %arg0, %arg1
// CHECK-NEXT:    rtl.output [[RES]]

rtl.module @or_idempotent(%arg0: i11, %arg1 : i11) -> (i11) {
  %0 = comb.or %arg0, %arg1, %arg1, %arg1 : i11
  rtl.output %0 : i11
}

// CHECK-LABEL: rtl.module @xor_idempotent(%arg0: i11, %arg1: i11, %arg2: i11) -> (i11) {
// CHECK-NEXT:    [[RES:%[0-9]+]] = comb.xor %arg0, %arg1
// CHECK-NEXT:    rtl.output [[RES]]

rtl.module @xor_idempotent(%arg0: i11, %arg1: i11, %arg2: i11) -> (i11) {
  %0 = comb.xor %arg0, %arg1, %arg2, %arg2 : i11
  rtl.output %0 : i11
}

// CHECK-LABEL: rtl.module @xor_idempotent_two_arguments(%arg0: i11) -> (i11) {
// CHECK-NEXT:    %c0_i11 = comb.constant 0 : i11
// CHECK-NEXT:    rtl.output %c0_i11 : i11

rtl.module @xor_idempotent_two_arguments(%arg0: i11) -> (i11) {
  %c0_i11 = comb.constant 0 : i11
  %0 = comb.xor %arg0, %arg0 : i11
  rtl.output %0 : i11
}

// Add reduction to shift left and multiplication.

// CHECK-LABEL: rtl.module @add_reduction1(%arg0: i11, %arg1: i11) -> (i11) {
// CHECK-NEXT:    %c1_i11 = comb.constant 1 : i11
// CHECK-NEXT:   [[RES:%[0-9]+]] = comb.shl %arg1, %c1_i11
// CHECK-NEXT:    rtl.output [[RES]]

rtl.module @add_reduction1(%arg0: i11, %arg1: i11) -> (i11) {
  %c1_i11 = comb.constant 1 : i11
  %0 = comb.add %arg1, %arg1 : i11
  rtl.output %0 : i11
}

// CHECK-LABEL: rtl.module @add_reduction2(%arg0: i11, %arg1: i11) -> (i11) {
// CHECK-NEXT:    %c3_i11 = comb.constant 3 : i11
// CHECK-NEXT:   [[RES:%[0-9]+]] = comb.mul %arg1, %c3_i11
// CHECK-NEXT:    rtl.output [[RES]]

rtl.module @add_reduction2(%arg0: i11, %arg1: i11) -> (i11) {
  %c3_i11 = comb.constant 3 : i11
  %0 = comb.add %arg1, %arg1, %arg1 : i11
  rtl.output %0 : i11
}

// CHECK-LABEL: rtl.module @add_reduction3(%arg0: i11, %arg1: i11) -> (i11) {
// CHECK-NEXT:    %c3_i11 = comb.constant 3 : i11
// CHECK-NEXT:   [[RES:%[0-9]+]] = comb.shl %arg1, %c3_i11
// CHECK-NEXT:    rtl.output [[RES]]

rtl.module @add_reduction3(%arg0: i11, %arg1: i11) -> (i11) {
  %c3_i11 = comb.constant 3 : i11
  %c7_i11 = comb.constant 7 : i11
  %0 = comb.mul %arg1, %c7_i11 : i11
  %1 = comb.add %arg1, %0 : i11
  rtl.output %1 : i11
}

// Multiply reduction to shift left.

// CHECK-LABEL: rtl.module @multiply_reduction(%arg0: i11, %arg1: i11) -> (i11) {
// CHECK-NEXT:    %c1_i11 = comb.constant 1 : i11
// CHECK-NEXT:   [[RES:%[0-9]+]] = comb.shl %arg1, %c1_i11
// CHECK-NEXT:    rtl.output [[RES]]

rtl.module @multiply_reduction(%arg0: i11, %arg1: i11) -> (i11) {
  %c1_i11 = comb.constant 1 : i11
  %c2_i11 = comb.constant 2 : i11
  %0 = comb.mul %arg1, %c2_i11 : i11
  rtl.output %0 : i11
}

// CHECK-LABEL: rtl.module @sext_constant_folding() -> (i5) {
// CHECK-NEXT:  %c-8_i5 = comb.constant -8 : i5
// CHECK-NEXT:  rtl.output %c-8_i5 : i5

rtl.module @sext_constant_folding() -> (i5) {
  %c8_i4 = comb.constant 8 : i4
  %0 = comb.sext %c8_i4 : (i4) -> (i5)
  rtl.output %0 : i5
}

// CHECK-LABEL: rtl.module @xorr_constant_folding1() -> (i1) {
// CHECK-NEXT:  %true = comb.constant true
// CHECK-NEXT:  rtl.output %true : i1

rtl.module @xorr_constant_folding1() -> (i1) {
  %c4_i4 = comb.constant 4 : i4
  %0 = comb.xorr %c4_i4 : i4
  rtl.output %0 : i1
}

// CHECK-LABEL: rtl.module @xorr_constant_folding2() -> (i1) {
// CHECK-NEXT:  %false = comb.constant false
// CHECK-NEXT:  rtl.output %false : i1
rtl.module @xorr_constant_folding2() -> (i1) {
  %c15_i4 = comb.constant 15 : i4
  %0 = comb.xorr %c15_i4 : i4
  rtl.output %0 : i1
}

// CHECK-LABEL: rtl.module @concat_fold_0
// CHECK-NEXT:  %c120_i8 = comb.constant 120 : i8
rtl.module @concat_fold_0() -> (i8) {
  %c7_i4 = comb.constant 7 : i4
  %c4_i3 = comb.constant 4 : i3
  %false = comb.constant false
  %0 = comb.concat %c7_i4, %c4_i3, %false : (i4, i3, i1) -> (i8)
  rtl.output %0 : i8
}

// CHECK-LABEL: rtl.module @concat_fold_1
// CHECK-NEXT:  %0 = comb.concat %arg0, %arg1, %arg2
rtl.module @concat_fold_1(%arg0: i4, %arg1: i3, %arg2: i1) -> (i8) {
  %a = comb.concat %arg0, %arg1 : (i4, i3) -> (i7)
  %b = comb.concat %a, %arg2 : (i7, i1) -> (i8)
  rtl.output %b : i8
}

// CHECK-LABEL: rtl.module @concat_fold_2
rtl.module @concat_fold_2(%arg0: i3, %arg1: i1) -> (i9) {
  // CHECK-NEXT:  %0 = comb.extract %arg0 from 2 : (i3) -> i1
  %b = comb.sext %arg0 : (i3) -> (i8)
  // CHECK-NEXT:  = comb.concat %0, %0, %0, %0, %0, %arg0, %arg1 : (i1, i1, i1, i1, i1, i3, i1) -> i9
  %c = comb.concat %b, %arg1 : (i8, i1) -> (i9)
  rtl.output %c : i9
}

// CHECK-LABEL: rtl.module @concat_fold_3
// CHECK-NEXT:    %c60_i7 = comb.constant 60 : i7
// CHECK-NEXT:    %0 = comb.concat %c60_i7, %arg0 : (i7, i1) -> i8
rtl.module @concat_fold_3(%arg0: i1) -> (i8) {
  %c7_i4 = comb.constant 7 : i4
  %c4_i3 = comb.constant 4 : i3
  %0 = comb.concat %c7_i4, %c4_i3, %arg0 : (i4, i3, i1) -> (i8)
  rtl.output %0 : i8
}

// CHECK-LABEL: rtl.module @concat_fold_4
// CHECK-NEXT:   %0 = comb.sext %arg0 : (i3) -> i5
rtl.module @concat_fold_4(%arg0: i3) -> (i5) {
  %0 = comb.extract %arg0 from 2 : (i3) -> (i1)
  %1 = comb.concat %0, %0, %arg0 : (i1, i1, i3) -> (i5)
  rtl.output %1 : i5
}


// CHECK-LABEL: rtl.module @concat_fold_5
// CHECK-NEXT:   %0 = comb.concat %arg0, %arg1 : (i3, i3) -> i6
// CHECK-NEXT:   rtl.output %0, %arg0
rtl.module @concat_fold_5(%arg0: i3, %arg1: i3) -> (i6, i3) {
  %0 = comb.extract %arg0 from 2 : (i3) -> (i1)
  %1 = comb.extract %arg0 from 0 : (i3) -> i2
  %2 = comb.concat %0, %1, %arg1 : (i1, i2, i3) -> i6

  %3 = comb.concat %0, %1 : (i1, i2) -> i3
  rtl.output %2, %3 : i6, i3
}

// CHECK-LABEL: rtl.module @concat_fold6(%arg0: i5, %arg1: i3) -> (i4) {
// CHECK-NEXT: %0 = comb.extract %arg0 from 1 : (i5) -> i4
// CHECK-NEXT: rtl.output %0 : i4
rtl.module @concat_fold6(%arg0: i5, %arg1: i3) -> (i4) {
  %0 = comb.extract %arg0 from 3 : (i5) -> i2
  %1 = comb.extract %arg0 from 1 : (i5) -> i2
  %2 = comb.concat %0, %1 : (i2, i2) -> i4
  rtl.output %2 : i4
}

// CHECK-LABEL: rtl.module @wire0()
// CHECK-NEXT:    rtl.output
rtl.module @wire0() {
  %w = sv.wire : !rtl.inout<i1>
  rtl.output
}

// CHECK-LABEL: rtl.module @wire1()
// CHECK-NEXT:    rtl.output
rtl.module @wire1() {
  %w = sv.wire : !rtl.inout<i1>
  %0 = sv.read_inout %w : !rtl.inout<i1>
  rtl.output
}

// CHECK-LABEL: rtl.module @wire2()
// CHECK-NEXT:    rtl.output
rtl.module @wire2() {
  %c = comb.constant 1 : i1
  %w = sv.wire : !rtl.inout<i1>
  sv.connect %w, %c : i1
  rtl.output
}

// CHECK-LABEL: rtl.module @wire3()
// CHECK-NEXT:    rtl.output
rtl.module @wire3() {
  %c = comb.constant 1 : i1
  %w = sv.wire : !rtl.inout<i1>
  %0 = sv.read_inout %w : !rtl.inout<i1>
  sv.connect %w, %c :i1
  rtl.output
}

// CHECK-LABEL: rtl.module @wire4() -> (i1)
// CHECK-NEXT:    %true = comb.constant true
// CHECK-NEXT:    %w = sv.wire : !rtl.inout<i1>
// CHECK-NEXT:    %0 = sv.read_inout %w : !rtl.inout<i1>
// CHECK-NEXT:    sv.connect %w, %true : i1
// CHECK-NEXT:    rtl.output %0
rtl.module @wire4() -> (i1) {
  %true = comb.constant true
  %w = sv.wire : !rtl.inout<i1>
  %0 = sv.read_inout %w : !rtl.inout<i1>
  sv.connect %w, %true : i1
  rtl.output %0 : i1
}

// CHECK-LABEL: rtl.module @mux_fold0(%arg0: i3, %arg1: i3)
// CHECK-NEXT:    rtl.output %arg0 : i3
rtl.module @mux_fold0(%arg0: i3, %arg1: i3) -> (i3) {
  %c1_i1 = comb.constant 1 : i1
  %0 = comb.mux %c1_i1, %arg0, %arg1 : i3
  rtl.output %0 : i3
}

// CHECK-LABEL: rtl.module @mux_fold1(%arg0: i1, %arg1: i3)
// CHECK-NEXT:    rtl.output %arg1 : i3
rtl.module @mux_fold1(%arg0: i1, %arg1: i3) -> (i3) {
  %0 = comb.mux %arg0, %arg1, %arg1 : i3
  rtl.output %0 : i3
}

// CHECK-LABEL: rtl.module @icmp_fold_constants() -> (i1) {
// CHECK-NEXT:    %false = comb.constant false
// CHECK-NEXT:    rtl.output %false : i1
rtl.module @icmp_fold_constants() -> (i1) {
  %c2_i2 = comb.constant 2 : i2
  %c3_i2 = comb.constant 3 : i2
  %0 = comb.icmp uge %c2_i2, %c3_i2 : i2
  rtl.output %0 : i1
}

// CHECK-LABEL: rtl.module @icmp_fold_same_operands(%arg0: i2) -> (i1) {
// CHECK-NEXT:    %false = comb.constant false
// CHECK-NEXT:    rtl.output %false : i1
rtl.module @icmp_fold_same_operands(%arg0: i2) -> (i1) {
  %0 = comb.icmp ugt %arg0, %arg0 : i2
  rtl.output %0 : i1
}

// CHECK-LABEL: rtl.module @icmp_fold_constant_rhs0(%arg0: i2) -> (i1) {
// CHECK-NEXT:    %false = comb.constant false
// CHECK-NEXT:    rtl.output %false : i1
rtl.module @icmp_fold_constant_rhs0(%arg0: i2) -> (i1) {
  %c3_i2 = comb.constant 3 : i2
  %0 = comb.icmp ugt %arg0, %c3_i2 : i2
  rtl.output %0 : i1
}

// CHECK-LABEL: rtl.module @icmp_fold_constant_rhs1(%arg0: i2) -> (i1) {
// CHECK-NEXT:    %false = comb.constant false
// CHECK-NEXT:    rtl.output %false : i1
rtl.module @icmp_fold_constant_rhs1(%arg0: i2) -> (i1) {
  %c-2_i2 = comb.constant -2 : i2
  %0 = comb.icmp slt %arg0, %c-2_i2 : i2
  rtl.output %0 : i1
}

// CHECK-LABEL: rtl.module @icmp_fold_constant_lhs0(%arg0: i2) -> (i1) {
// CHECK-NEXT:    %true = comb.constant true
// CHECK-NEXT:    rtl.output %true : i1
rtl.module @icmp_fold_constant_lhs0(%arg0: i2) -> (i1) {
  %c3_i2 = comb.constant 3 : i2
  %0 = comb.icmp uge %c3_i2, %arg0 : i2
  rtl.output %0 : i1
}

// CHECK-LABEL: rtl.module @icmp_fold_constant_lhs1(%arg0: i2) -> (i1) {
// CHECK-NEXT:    %true = comb.constant true
// CHECK-NEXT:    rtl.output %true : i1
rtl.module @icmp_fold_constant_lhs1(%arg0: i2) -> (i1) {
  %c-2_i2 = comb.constant -2 : i2
  %0 = comb.icmp sle %c-2_i2, %arg0 : i2
  rtl.output %0 : i1
}

// CHECK-LABEL: rtl.module @icmp_canonicalize0(%arg0: i2) -> (i1) {
// CHECK-NEXT:    %c-1_i2 = comb.constant -1 : i2
// CHECK-NEXT:    %0 = comb.icmp sgt %arg0, %c-1_i2 : i2
// CHECK-NEXT:    rtl.output %0 : i1
rtl.module @icmp_canonicalize0(%arg0: i2) -> (i1) {
  %c-1_i2 = comb.constant -1 : i2
  %0 = comb.icmp slt %c-1_i2, %arg0 : i2
  rtl.output %0 : i1
}

// CHECK-LABEL: rtl.module @icmp_canonicalize_ne(%arg0: i2) -> (i1) {
// CHECK-NEXT:    %c-2_i2 = comb.constant -2 : i2
// CHECK-NEXT:    %0 = comb.icmp ne %arg0, %c-2_i2 : i2
// CHECK-NEXT:    rtl.output %0 : i1
rtl.module @icmp_canonicalize_ne(%arg0: i2) -> (i1) {
  %c-2_i2 = comb.constant -2 : i2
  %0 = comb.icmp slt %c-2_i2, %arg0 : i2
  rtl.output %0 : i1
}

// CHECK-LABEL: rtl.module @icmp_canonicalize_eq(%arg0: i2) -> (i1) {
// CHECK-NEXT:    %c-2_i2 = comb.constant -2 : i2
// CHECK-NEXT:    %0 = comb.icmp eq %arg0, %c-2_i2 : i2
// CHECK-NEXT:    rtl.output %0 : i1
rtl.module @icmp_canonicalize_eq(%arg0: i2) -> (i1) {
  %c-1_i2 = comb.constant -1 : i2
  %0 = comb.icmp slt %arg0, %c-1_i2: i2
  rtl.output %0 : i1
}

// CHECK-LABEL: rtl.module @icmp_canonicalize_sgt(%arg0: i2) -> (i1) {
// CHECK-NEXT:    %c-1_i2 = comb.constant -1 : i2
// CHECK-NEXT:    %0 = comb.icmp sgt %arg0, %c-1_i2 : i2
// CHECK-NEXT:    rtl.output %0 : i1
rtl.module @icmp_canonicalize_sgt(%arg0: i2) -> (i1) {
  %c0_i2 = comb.constant 0 : i2
  %0 = comb.icmp sle %c0_i2, %arg0 : i2
  rtl.output %0 : i1
}

// CHECK-LABEL: rtl.module @shl_fold1() -> (i12) {
// CHECK-NEXT:   %c84_i12 = comb.constant 84 : i12
// CHECK-NEXT: rtl.output %c84_i12 : i12
rtl.module @shl_fold1() -> (i12) {
  %c42_i12 = comb.constant 42 : i12
  %c1_i12 = comb.constant 1 : i12
  %0 = comb.shl %c42_i12, %c1_i12 : i12
  rtl.output %0 : i12
}

// CHECK-LABEL: rtl.module @shl_fold2() -> (i12) {
// CHECK-NEXT:   %c0_i12 = comb.constant 0 : i12
// CHECK-NEXT: rtl.output %c0_i12 : i12
rtl.module @shl_fold2() -> (i12) {
  %c1_i12 = comb.constant 1 : i12
  %c10_i12 = comb.constant 12 : i12
  %0 = comb.shl %c1_i12, %c10_i12 : i12
  rtl.output %0 : i12
}

// CHECK-LABEL: rtl.module @shru_fold1() -> (i12) {
// CHECK-NEXT:   %c21_i12 = comb.constant 21 : i12
// CHECK-NEXT: rtl.output %c21_i12 : i12
rtl.module @shru_fold1() -> (i12) {
  %c42_i12 = comb.constant 42 : i12
  %c1_i12 = comb.constant 1 : i12
  %0 = comb.shru %c42_i12, %c1_i12 : i12
  rtl.output %0 : i12
}

// CHECK-LABEL: rtl.module @shru_fold2() -> (i12) {
// CHECK-NEXT:   %c2047_i12 = comb.constant 2047 : i12
// CHECK-NEXT: rtl.output %c2047_i12 : i12
rtl.module @shru_fold2() -> (i12) {
  %c-1_i12 = comb.constant -1 : i12
  %c1_i12 = comb.constant 1 : i12
  %0 = comb.shru %c-1_i12, %c1_i12 : i12
  rtl.output %0 : i12
}

// CHECK-LABEL: rtl.module @shrs_fold1() -> (i12) {
// CHECK-NEXT:   %c21_i12 = comb.constant 21 : i12
// CHECK-NEXT: rtl.output %c21_i12 : i12
rtl.module @shrs_fold1() -> (i12) {
  %c42_i12 = comb.constant 42 : i12
  %c1_i12 = comb.constant 1 : i12
  %0 = comb.shrs %c42_i12, %c1_i12 : i12
  rtl.output %0 : i12
}

// CHECK-LABEL: rtl.module @shrs_fold2() -> (i12) {
// CHECK-NEXT:   %c-3_i12 = comb.constant -3 : i12
// CHECK-NEXT: rtl.output %c-3_i12 : i12
rtl.module @shrs_fold2() -> (i12) {
  %c-5_i12 = comb.constant -5 : i12
  %c10_i12 = comb.constant 1 : i12
  %0 = comb.shrs %c-5_i12, %c10_i12 : i12
  rtl.output %0 : i12
}
