// RUN: circt-opt %s -canonicalize | FileCheck %s

// This test checks the canonicalization of the LLHD arithmetic and bitwise 
// operations. For each operation it checks:
//   * changing the order of the operands of commutative operations s.t. the 
//     constant one is on the right
//   * folding patterns
//   * result of the folding if all operands are constant
// Furthermore it checks the hoisting of llhd constants to the entry block


// CHECK-LABEL: @check_neg
func @check_neg() -> i32 {
    // CHECK-NEXT: %[[C0:.*]] = llhd.const -5 : i32
    %0 = llhd.const 5 : i32
    %2 = llhd.neg %0 : i32
    // CHECK-NEXT: return %[[C0]] : i32
    return %2 : i32
}

// CHECK-LABEL: @check_smod
// CHECK-SAME: %[[IN:.*]]: i32
func @check_smod(%in : i32) -> (i32, i32, i32, i32, i32, i32, i32) {
    // CHECK-NEXT: %[[FOUR:.*]] = llhd.const 4 : i32
    // CHECK-NEXT: %[[ONE:.*]] = llhd.const 1 : i32
    // CHECK-NEXT: %[[NEGONE:.*]] = llhd.const -1 : i32
    // CHECK-NEXT: %[[NEGFOUR:.*]] = llhd.const -4 : i32
    // CHECK-NEXT: %[[ZERO:.*]] = llhd.const 0 : i32
    %0 = llhd.const 5 : i32
    %1 = llhd.const -5 : i32
    %2 = llhd.const 9 : i32
    %3 = llhd.const -9 : i32
    %4 = llhd.const 1 : i32
    %5 = llhd.const 0 : i32
    // check correct calculation of smod if lhs and rhs are positive
    %6 = llhd.smod %2, %0 : i32
    // check correct calculation of smod if lhs is negative and rhs is positive
    %7 = llhd.smod %3, %0 : i32
    // check correct calculation of smod if lhs is positive and rhs is negative
    %8 = llhd.smod %2, %1 : i32
    // check correct calculation of smod if lhs and rhs are negative
    %9 = llhd.smod %3, %1 : i32
    // check correct pattern replacement if rhs is one
    %10 = llhd.smod %in, %4 : i32
    // check correct pattern replacement if lhs is zero
    %11 = llhd.smod %5, %in : i32
    // check correct pattern replacement if lhs and rhs are the same register
    %12 = llhd.smod %in, %in : i32
    // CHECK-NEXT: return %[[FOUR]], %[[ONE]], %[[NEGONE]], %[[NEGFOUR]], %[[ZERO]], %[[ZERO]], %[[ZERO]] : i32, i32, i32, i32, i32, i32, i32 
    return %6, %7, %8, %9, %10, %11, %12 : i32, i32, i32, i32, i32, i32, i32 
}

// CHECK-LABEL: @check_not
func @check_not() -> i32 {
    // CHECK-NEXT: %[[C0:.*]] = llhd.const -2 : i32
    %0 = llhd.const 1 : i32
    %2 = llhd.not %0 : i32
    // CHECK-NEXT: return %[[C0]] : i32
    return %2 : i32
}

// CHECK-LABEL: @check_and
// CHECK-SAME: %[[IN:.*]]: i32
func @check_and(%in : i32) -> (i32, i32, i32, i32) {
    // CHECK-NEXT: %[[ZERO:.*]] = llhd.const 0 : i32
    // CHECK-NEXT: %[[TWO:.*]] = llhd.const 2 : i32
    %0 = llhd.const 2 : i32
    %1 = llhd.const -1 : i32
    %2 = llhd.const 0 : i32
    // check correct calculation of smod if lhs and rhs are positive
    %3 = llhd.and %1, %0 : i32
    // check correct pattern replacement if one operand is zero
    %4 = llhd.and %2, %in : i32
    // check correct pattern replacement if one operand is all ones
    %5 = llhd.and %1, %in : i32
    // check correct pattern replacement if lhs and rhs are the same register
    %6 = llhd.and %in, %in : i32
    // CHECK-NEXT: return %[[TWO]], %[[ZERO]], %[[IN]], %[[IN]] : i32, i32, i32, i32
    return %3, %4, %5, %6 : i32, i32, i32, i32
}

// CHECK-LABEL: @check_or
// CHECK-SAME: %[[IN:.*]]: i32
func @check_or(%in : i32) -> (i32, i32, i32, i32) {
    // CHECK-NEXT: %[[NEGONE:.*]] = llhd.const -1 : i32
    // CHECK-NEXT: %[[TWO:.*]] = llhd.const 2 : i32
    %0 = llhd.const 2 : i32
    %1 = llhd.const -1 : i32
    %2 = llhd.const 0 : i32
    // check correct calculation of smod if lhs and rhs are positive
    %3 = llhd.or %2, %0 : i32
    // check corre2t pattern replacement if one operand is zero
    %4 = llhd.or %2, %in : i32
    // check correct pattern replacement if one operand is all ones
    %5 = llhd.or %1, %in : i32
    // check correct pattern replacement if lhs and rhs are the same register
    %6 = llhd.or %in, %in : i32
    // CHECK-NEXT: return %[[TWO]], %[[IN]], %[[NEGONE]], %[[IN]] : i32, i32, i32, i32
    return %3, %4, %5, %6 : i32, i32, i32, i32
}

// CHECK-LABEL: @check_xor
// CHECK-SAME: %[[IN:.*]]: i32
func @check_xor(%in : i32) -> (i32, i32, i32) {
    // CHECK-NEXT: %[[TWO:.*]] = llhd.const 2 : i32
    // CHECK-NEXT: %[[ZERO:.*]] = llhd.const 0 : i32
    %0 = llhd.const 2 : i32
    %2 = llhd.const 0 : i32
    // check correct calculation of smod if lhs and rhs are positive
    %3 = llhd.xor %2, %0 : i32
    // check correct pattern replacement if one operand is zero
    %4 = llhd.xor %2, %in : i32
    // check correct pattern replacement if lhs and rhs are the same register
    %6 = llhd.xor %in, %in : i32
    // CHECK-NEXT: return %[[TWO]], %[[IN]], %[[ZERO]] : i32, i32, i32
    return %3, %4, %6 : i32, i32, i32
}

// CHECK-LABEL: @check_shl
// CHECK-SAME: %[[BASE:.*]]: i4
// CHECK-SAME: %[[HIDDEN:.*]]: i8
func @check_shl(%base : i4, %hidden : i8) -> (i4, i4) {
    // CHECK-NEXT: %[[NEGSEVEN:.*]] = llhd.const -7 : i4
    %0 = llhd.const 2 : i4
    %1 = llhd.const 4 : i4
    %2 = llhd.const 0 : i4
    // check correct calculation of smod if lhs and rhs are positive
    %3 = llhd.shl %0, %1, %0 : (i4, i4, i4) -> i4
    // check correct pattern replacement if amount is constant zero
    %4 = llhd.shl %base, %hidden, %2 : (i4, i8, i4) -> i4
    // CHECK-NEXT: return %[[NEGSEVEN]], %[[BASE]] : i4, i4
    return %3, %4 : i4, i4
}

// CHECK-LABEL: @check_shr
// CHECK-SAME: %[[BASE:.*]]: i4
// CHECK-SAME: %[[HIDDEN:.*]]: i8
func @check_shr(%base : i4, %hidden : i8) -> (i4, i4) {
    // CHECK-NEXT: %[[NEGSEVEN:.*]] = llhd.const -7 : i4
    %0 = llhd.const 2 : i4
    %1 = llhd.const 4 : i4
    %2 = llhd.const 0 : i4
    // check correct calculation of smod if lhs and rhs are positive
    %3 = llhd.shr %1, %0, %0 : (i4, i4, i4) -> i4
    // check correct pattern replacement if amount is constant zero
    %4 = llhd.shr %base, %hidden, %2 : (i4, i8, i4) -> i4
    // CHECK-NEXT: return %[[NEGSEVEN]], %[[BASE]] : i4, i4
    return %3, %4 : i4, i4
}

// CHECK-LABEL: @const_hoisting
// CHECK-SAME: %[[SIG:.*]]: !llhd.sig<i32>
func @const_hoisting(%sig : !llhd.sig<i32>) {
    // CHECK-NEXT: %[[C0:.*]] = llhd.const -1 : i32
    // CHECK-NEXT: %[[TIME:.*]] = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
    // CHECK-NEXT: br ^[[BB:.*]]
    br ^bb1
// CHECK-NEXT: ^[[BB]]
^bb1:
    %0 = llhd.const -1 : i32
    %1 = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
    // CHECK-NEXT: llhd.drv %[[SIG]], %[[C0]] after %[[TIME]] : !llhd.sig<i32>
    llhd.drv %sig, %0 after %1 : !llhd.sig<i32>
    br ^bb1
}
