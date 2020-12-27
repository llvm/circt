// RUN: circt-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: func @extract_noop(%arg0: i3) -> i3 {
// CHECK-NEXT:    return %arg0

func @extract_noop(%arg0: i3) -> i3 {
  %x = rtl.extract %arg0 from 0 : (i3) -> i3
  return %x : i3
}

// Constant Folding

// CHECK-LABEL: func @extract_cstfold() -> i3 {
// CHECK-NEXT:    %c-3_i3 = rtl.constant(-3 : i3)
// CHECK-NEXT:    return  %c-3_i3

func @extract_cstfold() -> i3 {
  %c42_i12 = rtl.constant(42 : i12) : i12
  %x = rtl.extract %c42_i12 from 3 : (i12) -> i3
  return %x : i3
}

// CHECK-LABEL: func @and_cstfold(%arg0: i7) -> i7 {
// CHECK-NEXT:    %c1_i7 = rtl.constant(1 : i7)
// CHECK-NEXT:    %0 = rtl.and %arg0, %c1_i7 : i7
// CHECK-NEXT:    return %0 : i7

func @and_cstfold(%arg0: i7) -> i7 {
  %c11_i7 = rtl.constant(11 : i7) : i7
  %c5_i7 = rtl.constant(5 : i7) : i7
  %0 = rtl.and %arg0, %c11_i7, %c5_i7 : i7
  return %0 : i7
}

// CHECK-LABEL: func @or_cstfold(%arg0: i7) -> i7 {
// CHECK-NEXT:    %c15_i7 = rtl.constant(15 : i7)
// CHECK-NEXT:    %0 = rtl.or %arg0, %c15_i7 : i7
// CHECK-NEXT:    return %0 : i7

func @or_cstfold(%arg0: i7) -> i7 {
  %c11_i7 = rtl.constant(11 : i7) : i7
  %c5_i7 = rtl.constant(5 : i7) : i7
  %0 = rtl.or %arg0, %c11_i7, %c5_i7 : i7
  return %0 : i7
}

// CHECK-LABEL: func @xor_cstfold(%arg0: i7) -> i7 {
// CHECK-NEXT:    %c14_i7 = rtl.constant(14 : i7)
// CHECK-NEXT:    %0 = rtl.xor %arg0, %c14_i7 : i7
// CHECK-NEXT:    return %0 : i7

func @xor_cstfold(%arg0: i7) -> i7 {
  %c11_i7 = rtl.constant(11 : i7) : i7
  %c5_i7 = rtl.constant(5 : i7) : i7
  %0 = rtl.xor %arg0, %c11_i7, %c5_i7 : i7
  return %0 : i7
}

// CHECK-LABEL: func @merge_fold
// CHECK-NEXT:    %0 = rtl.merge %arg0, %arg0, %arg1 : i7
// CHECK-NEXT:    return %arg0, %arg0, %0 : i7, i7, i7
func @merge_fold(%arg0: i7, %arg1: i7) -> (i7, i7, i7) {
  %a = rtl.merge %arg0 : i7
  %b = rtl.merge %arg0, %arg0, %arg0 : i7
  %c = rtl.merge %arg0, %arg0, %arg1 : i7
  return %a, %b, %c: i7, i7, i7
}

// CHECK-LABEL: func @add_cstfold(%arg0: i7) -> i7 {
// CHECK-NEXT:    %c15_i7 = rtl.constant(15 : i7)
// CHECK-NEXT:    %0 = rtl.add %arg0, %c15_i7 : i7
// CHECK-NEXT:    return %0 : i7
func @add_cstfold(%arg0: i7) -> i7 {
  %c10_i7 = rtl.constant(10 : i7) : i7
  %c5_i7 = rtl.constant(5 : i7) : i7
  %0 = rtl.add %arg0, %c10_i7, %c5_i7 : i7
  return %0 : i7
}

// CHECK-LABEL: func @mul_cstfold(%arg0: i7) -> i7 {
// CHECK-NEXT:    %c15_i7 = rtl.constant(15 : i7)
// CHECK-NEXT:    %0 = rtl.mul %arg0, %c15_i7 : i7
// CHECK-NEXT:    return %0 : i7
func @mul_cstfold(%arg0: i7) -> i7 {
  %c3_i7 = rtl.constant(3 : i7) : i7
  %c5_i7 = rtl.constant(5 : i7) : i7
  %0 = rtl.mul %arg0, %c3_i7, %c5_i7 : i7
  return %0 : i7
}

// CHECK-LABEL: func @variadic_noop(%arg0: i11) -> i11 {
// CHECK-NEXT:    return %arg0

func @variadic_noop(%arg0: i11) -> i11 {
  %0 = rtl.and %arg0 : i11
  %1 = rtl.or  %0 : i11
  %2 = rtl.xor %1 : i11
  %3 = rtl.add %2 : i11
  %4 = rtl.mul %3 : i11
  return %4 : i11
}

// CHECK-LABEL: func @and_annulment0(%arg0: i11, %arg1: i11) -> i11 {
// CHECK-NEXT:    %c0_i11 = rtl.constant(0 : i11)
// CHECK-NEXT:    return %c0_i11

func @and_annulment0(%arg0: i11, %arg1: i11) -> i11 {
  %c0_i11 = rtl.constant(0 : i11) : i11
  %0 = rtl.and %arg0, %arg1, %c0_i11 : i11
  return %0 : i11
}

// CHECK-LABEL: func @and_annulment1(%arg0: i7)
// CHECK-NEXT:    %c0_i7 = rtl.constant(0 : i7)
// CHECK-NEXT:    return %c0_i7

func @and_annulment1(%arg0: i7) -> i7 {
  %c1_i7 = rtl.constant(1 : i7) : i7
  %c2_i7 = rtl.constant(2 : i7) : i7
  %c4_i7 = rtl.constant(4 : i7) : i7
  %0 = rtl.and %arg0, %c1_i7, %c2_i7, %c4_i7: i7
  return %0 : i7
}

// CHECK-LABEL: func @or_annulment0(%arg0: i11) -> i11 {
// CHECK-NEXT:    %c-1_i11 = rtl.constant(-1 : i11)
// CHECK-NEXT:    return %c-1_i11

func @or_annulment0(%arg0: i11) -> i11 {
  %c-1_i11 = rtl.constant(-1 : i11) : i11
  %0 = rtl.or %arg0, %arg0, %arg0, %c-1_i11 : i11
  return %0 : i11
}

// CHECK-LABEL: func @or_annulment1(%arg0: i3)
// CHECK-NEXT:    %c-1_i3 = rtl.constant(-1 : i3)
// CHECK-NEXT:    return %c-1_i3

func @or_annulment1(%arg0: i3) -> i3 {
  %c1_i3 = rtl.constant(1 : i3) : i3
  %c2_i3 = rtl.constant(2 : i3) : i3
  %c4_i3 = rtl.constant(4 : i3) : i3
  %0 = rtl.or %arg0, %c1_i3, %c2_i3, %c4_i3: i3
  return %0 : i3
}

// CHECK-LABEL: func @mul_annulment(%arg0: i11, %arg1: i11, %arg2: i11) -> i11 {
// CHECK-NEXT:    %c0_i11 = rtl.constant(0 : i11) : i11
// CHECK-NEXT:    return %c0_i11

func @mul_annulment(%arg0: i11, %arg1: i11, %arg2: i11) -> i11 {
  %c0_i11 = rtl.constant(0 : i11) : i11
  %0 = rtl.mul %arg0, %c0_i11, %arg1 : i11
  return %0 : i11
}

// CHECK-LABEL: func @mul_overflow(%arg0: i2) -> i2 {
// CHECK-NEXT:    %c0_i2 = rtl.constant(0 : i2) : i2
// CHECK-NEXT:    return %c0_i2

func @mul_overflow(%arg0: i2) -> i2 {
  %c2_i2 = rtl.constant(2 : i2) : i2
  %0 = rtl.mul %arg0, %c2_i2, %c2_i2 : i2
  return %0 : i2
}

// Flattening

// CHECK-LABEL: func @and_flatten_in_back(%arg0: i7, %arg1: i7, %arg2: i7) -> i7 {
// CHECK-NEXT:    [[RES:%[0-9]+]] = rtl.and %arg0, %arg1, %arg2 : i7
// CHECK-NEXT:    return [[RES]] : i7

func @and_flatten_in_back(%arg0: i7, %arg1: i7, %arg2: i7) -> i7 {
  %and0 = rtl.and %arg1, %arg2 : i7
  %0 = rtl.and %arg0, %and0 : i7
  return %0 : i7
}

// CHECK-LABEL: func @and_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> i7 {
// CHECK-NEXT:    [[RES:%[0-9]+]] = rtl.and %arg0, %arg1, %arg2, %arg3 : i7
// CHECK-NEXT:    return [[RES]] : i7

func @and_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> i7 {
  %and0 = rtl.and %arg1, %arg2 : i7
  %0 = rtl.and %arg0, %and0, %arg3 : i7
  return %0 : i7
}

// CHECK-LABEL: func @and_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> i7 {
// CHECK-NEXT:    [[RES:%[0-9]+]] = rtl.and %arg0, %arg1, %arg2 : i7
// CHECK-NEXT:    return [[RES]] : i7

func @and_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> i7 {
  %and0 = rtl.and %arg0, %arg1 : i7
  %0 = rtl.and %and0, %arg2 : i7
  return %0 : i7
}

// CHECK-LABEL: func @or_flatten_in_back(%arg0: i7, %arg1: i7, %arg2: i7) -> i7 {
// CHECK-NEXT:    [[RES:%[0-9]+]] = rtl.or %arg0, %arg1, %arg2 : i7
// CHECK-NEXT:    return [[RES]] : i7

func @or_flatten_in_back(%arg0: i7, %arg1: i7, %arg2: i7) -> i7 {
  %or0 = rtl.or %arg1, %arg2 : i7
  %0 = rtl.or %arg0, %or0 : i7
  return %0 : i7
}

// CHECK-LABEL: func @or_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> i7 {
// CHECK-NEXT:    [[RES:%[0-9]+]] = rtl.or %arg0, %arg1, %arg2, %arg3 : i7
// CHECK-NEXT:    return [[RES]] : i7

func @or_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> i7 {
  %or0 = rtl.or %arg1, %arg2 : i7
  %0 = rtl.or %arg0, %or0, %arg3 : i7
  return %0 : i7
}

// CHECK-LABEL: func @or_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> i7 {
// CHECK-NEXT:    [[RES:%[0-9]+]] = rtl.or %arg0, %arg1, %arg2 : i7
// CHECK-NEXT:    return [[RES]] : i7

func @or_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> i7 {
  %or0 = rtl.or %arg0, %arg1 : i7
  %0 = rtl.or %or0, %arg2 : i7
  return %0 : i7
}

// CHECK-LABEL: func @xor_flatten_in_back(%arg0: i7, %arg1: i7, %arg2: i7) -> i7 {
// CHECK-NEXT:    [[RES:%[0-9]+]] = rtl.xor %arg0, %arg1, %arg2 : i7
// CHECK-NEXT:    return [[RES]] : i7

func @xor_flatten_in_back(%arg0: i7, %arg1: i7, %arg2: i7) -> i7 {
  %xor0 = rtl.xor %arg1, %arg2 : i7
  %0 = rtl.xor %arg0, %xor0 : i7
  return %0 : i7
}

// CHECK-LABEL: func @xor_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> i7 {
// CHECK-NEXT:    [[RES:%[0-9]+]] = rtl.xor %arg0, %arg1, %arg2, %arg3 : i7
// CHECK-NEXT:    return [[RES]] : i7

func @xor_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> i7 {
  %xor0 = rtl.xor %arg1, %arg2 : i7
  %0 = rtl.xor %arg0, %xor0, %arg3 : i7
  return %0 : i7
}

// CHECK-LABEL: func @xor_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> i7 {
// CHECK-NEXT:    [[RES:%[0-9]+]] = rtl.xor %arg0, %arg1, %arg2 : i7
// CHECK-NEXT:    return [[RES]] : i7

func @xor_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> i7 {
  %xor0 = rtl.xor %arg0, %arg1 : i7
  %0 = rtl.xor %xor0, %arg2 : i7
  return %0 : i7
}

func @add_flatten_in_back(%arg0: i7, %arg1: i7, %arg2: i7) -> i7 {
  %add0 = rtl.add %arg1, %arg2 : i7
  %0 = rtl.add %arg0, %add0 : i7
  return %0 : i7
}

// CHECK-LABEL: func @add_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> i7 {
// CHECK-NEXT:    [[RES:%[0-9]+]] = rtl.add %arg0, %arg1, %arg2, %arg3 : i7
// CHECK-NEXT:    return [[RES]] : i7

func @add_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> i7 {
  %add0 = rtl.add %arg1, %arg2 : i7
  %0 = rtl.add %arg0, %add0, %arg3 : i7
  return %0 : i7
}

// CHECK-LABEL: func @add_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> i7 {
// CHECK-NEXT:    [[RES:%[0-9]+]] = rtl.add %arg0, %arg1, %arg2 : i7
// CHECK-NEXT:    return [[RES]] : i7

func @add_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> i7 {
  %add0 = rtl.add %arg0, %arg1 : i7
  %0 = rtl.add %add0, %arg2 : i7
  return %0 : i7
}

func @mul_flatten_in_back(%arg0: i7, %arg1: i7, %arg2: i7) -> i7 {
  %mul0 = rtl.mul %arg1, %arg2 : i7
  %0 = rtl.mul %arg0, %mul0 : i7
  return %0 : i7
}

// CHECK-LABEL: func @mul_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> i7 {
// CHECK-NEXT:    [[RES:%[0-9]+]] = rtl.mul %arg0, %arg1, %arg2, %arg3 : i7
// CHECK-NEXT:    return [[RES]] : i7

func @mul_flatten_in_middle(%arg0: i7, %arg1: i7, %arg2: i7, %arg3: i7) -> i7 {
  %mul0 = rtl.mul %arg1, %arg2 : i7
  %0 = rtl.mul %arg0, %mul0, %arg3 : i7
  return %0 : i7
}

// CHECK-LABEL: func @mul_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> i7 {
// CHECK-NEXT:    [[RES:%[0-9]+]] = rtl.mul %arg0, %arg1, %arg2 : i7
// CHECK-NEXT:    return [[RES]] : i7

func @mul_flatten_in_front(%arg0: i7, %arg1: i7, %arg2: i7) -> i7 {
  %mul0 = rtl.mul %arg0, %arg1 : i7
  %0 = rtl.mul %mul0, %arg2 : i7
  return %0 : i7
}

// Identities

// CHECK-LABEL: func @and_identity(%arg0: i11, %arg1: i11) -> i11 {
// CHECK-NEXT:    [[RES:%[0-9]+]] = rtl.and %arg0, %arg1
// CHECK-NEXT:    return [[RES]]

func @and_identity(%arg0: i11, %arg1: i11) -> i11 {
  %c-1_i11 = rtl.constant(-1 : i11) : i11
  %0 = rtl.and %c-1_i11, %arg0, %arg1, %c-1_i11 : i11
  return %0 : i11
}

// CHECK-LABEL: func @or_identity(%arg0: i11, %arg1: i11) -> i11 {
// CHECK-NEXT:    [[RES:%[0-9]+]] = rtl.or %arg0, %arg1
// CHECK-NEXT:    return [[RES]]

func @or_identity(%arg0: i11, %arg1: i11) -> i11 {
  %c0_i11 = rtl.constant(0 : i11) : i11
  %0 = rtl.or %arg0, %c0_i11, %arg1 : i11
  return %0 : i11
}

// CHECK-LABEL: func @xor_identity(%arg0: i11, %arg1: i11) -> i11 {
// CHECK-NEXT:    [[RES:%[0-9]+]] = rtl.xor %arg1, %arg0
// CHECK-NEXT:    return [[RES]]

func @xor_identity(%arg0: i11, %arg1: i11) -> i11 {
  %c0_i11 = rtl.constant(0 : i11) : i11
  %0 = rtl.xor %c0_i11, %arg1, %arg0 : i11
  return %0 : i11
}

// CHECK-LABEL: func @add_identity(%arg0: i11, %arg1: i11) -> i11 {
// CHECK-NEXT:   [[RES:%[0-9]+]] = rtl.add %arg0, %arg1
// CHECK-NEXT:    return [[RES]]

func @add_identity(%arg0: i11, %arg1: i11) -> i11 {
  %c0_i11 = rtl.constant(0 : i11) : i11
  %0 = rtl.add %arg0, %c0_i11, %arg1 : i11
  return %0 : i11
}

// CHECK-LABEL: func @mul_identity(%arg0: i11, %arg1: i11) -> i11 {
// CHECK-NEXT:    [[RES:%[0-9]+]] = rtl.mul %arg0, %arg1
// CHECK-NEXT:    return [[RES]]

func @mul_identity(%arg0: i11, %arg1: i11) -> i11 {
  %c1_i11 = rtl.constant(1 : i11) : i11
  %0 = rtl.mul %arg0, %c1_i11, %arg1 : i11
  return %0 : i11
}

// Idempotency

// CHECK-LABEL: func @and_idempotent(%arg0: i11, %arg1: i11) -> i11 {
// CHECK-NEXT:    %c9_i11 = rtl.constant(9 : i11) : i11
// CHECK-NEXT:    [[RES:%[0-9]+]] = rtl.and %arg0, %arg1, %c9_i11
// CHECK-NEXT:    return [[RES]]

func @and_idempotent(%arg0: i11, %arg1 : i11) -> i11 {
  %c9_i11 = rtl.constant(9 : i11) : i11
  %0 = rtl.and %arg0, %arg1, %c9_i11, %c9_i11 : i11
  return %0 : i11
}

// CHECK-LABEL: func @or_idempotent(%arg0: i11, %arg1: i11) -> i11 {
// CHECK-NEXT:    [[RES:%[0-9]+]] = rtl.or %arg0, %arg1
// CHECK-NEXT:    return [[RES]]

func @or_idempotent(%arg0: i11, %arg1 : i11) -> i11 {
  %0 = rtl.or %arg0, %arg1, %arg1, %arg1 : i11
  return %0 : i11
}

// CHECK-LABEL: func @xor_idempotent(%arg0: i11, %arg1: i11, %arg2: i11) -> i11 {
// CHECK-NEXT:    [[RES:%[0-9]+]] = rtl.xor %arg0, %arg1
// CHECK-NEXT:    return [[RES]]

func @xor_idempotent(%arg0: i11, %arg1: i11, %arg2: i11) -> i11 {
  %0 = rtl.xor %arg0, %arg1, %arg2, %arg2 : i11
  return %0 : i11
}

// CHECK-LABEL: func @xor_idempotent_two_arguments(%arg0: i11) -> i11 {
// CHECK-NEXT:    %c0_i11 = rtl.constant(0 : i11) : i11
// CHECK-NEXT:    return %c0_i11 : i11

func @xor_idempotent_two_arguments(%arg0: i11) -> i11 {
  %c0_i11 = rtl.constant(0 : i11) : i11
  %0 = rtl.xor %arg0, %arg0 : i11
  return %0 : i11
}

// Add reduction to shift left and multiplication.

// CHECK-LABEL: func @add_reduction1(%arg0: i11, %arg1: i11) -> i11 {
// CHECK-NEXT:    %c1_i11 = rtl.constant(1 : i11) : i11
// CHECK-NEXT:   [[RES:%[0-9]+]] = rtl.shl %arg1, %c1_i11
// CHECK-NEXT:    return [[RES]]

func @add_reduction1(%arg0: i11, %arg1: i11) -> i11 {
  %c1_i11 = rtl.constant(1 : i11) : i11
  %0 = rtl.add %arg1, %arg1 : i11
  return %0 : i11
}

// CHECK-LABEL: func @add_reduction2(%arg0: i11, %arg1: i11) -> i11 {
// CHECK-NEXT:    %c3_i11 = rtl.constant(3 : i11) : i11
// CHECK-NEXT:   [[RES:%[0-9]+]] = rtl.mul %arg1, %c3_i11
// CHECK-NEXT:    return [[RES]]

func @add_reduction2(%arg0: i11, %arg1: i11) -> i11 {
  %c3_i11 = rtl.constant(3 : i11) : i11
  %0 = rtl.add %arg1, %arg1, %arg1 : i11
  return %0 : i11
}

// CHECK-LABEL: func @add_reduction3(%arg0: i11, %arg1: i11) -> i11 {
// CHECK-NEXT:    %c3_i11 = rtl.constant(3 : i11) : i11
// CHECK-NEXT:   [[RES:%[0-9]+]] = rtl.shl %arg1, %c3_i11
// CHECK-NEXT:    return [[RES]]

func @add_reduction3(%arg0: i11, %arg1: i11) -> i11 {
  %c3_i11 = rtl.constant(3 : i11) : i11
  %c7_i11 = rtl.constant(7 : i11) : i11
  %0 = rtl.mul %arg1, %c7_i11 : i11
  %1 = rtl.add %arg1, %0 : i11
  return %1 : i11
}

// Multiply reduction to shift left.

// CHECK-LABEL: func @multiply_reduction(%arg0: i11, %arg1: i11) -> i11 {
// CHECK-NEXT:    %c1_i11 = rtl.constant(1 : i11) : i11
// CHECK-NEXT:   [[RES:%[0-9]+]] = rtl.shl %arg1, %c1_i11
// CHECK-NEXT:    return [[RES]]

func @multiply_reduction(%arg0: i11, %arg1: i11) -> i11 {
  %c1_i11 = rtl.constant(1 : i11) : i11
  %c2_i11 = rtl.constant(2 : i11) : i11
  %0 = rtl.mul %arg1, %c2_i11 : i11
  return %0 : i11
}

// CHECK-LABEL: func @zext_constant_folding() -> i5 {
// CHECK-NEXT:  %c8_i5 = rtl.constant(8 : i5) : i5
// CHECK-NEXT:  return %c8_i5 : i5

func @zext_constant_folding() -> i5 {
  %c8_i4 = rtl.constant(8 : i4) : i4
  %0 = rtl.zext %c8_i4 : (i4) -> i5
  return %0 : i5
}

// CHECK-LABEL: func @sext_constant_folding() -> i5 {
// CHECK-NEXT:  %c-8_i5 = rtl.constant(-8 : i5) : i5
// CHECK-NEXT:  return %c-8_i5 : i5

func @sext_constant_folding() -> i5 {
  %c8_i4 = rtl.constant(8 : i4) : i4
  %0 = rtl.sext %c8_i4 : (i4) -> i5
  return %0 : i5
}
