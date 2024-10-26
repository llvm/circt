// RUN: circt-opt %s --pass-pipeline="builtin.module(hw.module(convert-comb-to-aig{keep-bitwise-logical-ops=true}))" | FileCheck %s
// RUN: circt-opt %s --convert-comb-to-aig

// CHECK-LABEL: @add
hw.module @add(in %lhs: i2, in %rhs: i2, out out: i2) {
  // CHECK:      %[[lhs0:.*]] = comb.extract %lhs from 0 : (i2) -> i1
  // CHECK-NEXT: %[[lhs1:.*]] = comb.extract %lhs from 1 : (i2) -> i1
  // CHECK-NEXT: %[[rhs0:.*]] = comb.extract %rhs from 0 : (i2) -> i1
  // CHECK-NEXT: %[[rhs1:.*]] = comb.extract %rhs from 1 : (i2) -> i1
  // CHECK-NEXT: %[[sum0:.*]] = comb.xor bin %[[lhs0]], %[[rhs0]] : i1
  // CHECK-NEXT: %[[carry0:.*]] = comb.and bin %[[lhs0]], %[[rhs0]] : i1
  // CHECK-NEXT: %[[sum1:.*]] = comb.xor bin %[[lhs1]], %[[rhs1]], %[[carry0]] : i1
  // CHECK-NEXT: %[[concat:.*]] = comb.concat %[[sum1]], %[[sum0]] : i1, i1
  // CHECK-NEXT: hw.output %[[concat]] : i2
  %0 = comb.add %lhs, %rhs : i2
  hw.output %0 : i2
}

hw.module @sub(in %lhs: i2, in %rhs: i2, out out: i2) {
  %0 = comb.sub %lhs, %rhs : i2
  hw.output %0 : i2
}

// CHECK-LABEL: @icmp
hw.module @icmp(in %lhs: i2, in %rhs: i2, out out_eq: i1, out out_ne: i1,
                out out_le: i1, out out_lt: i1, out out_ge: i1, out out_gt: i1) {
  %0 = comb.icmp eq %lhs, %rhs : i2
  %1 = comb.icmp ne %lhs, %rhs : i2
  %2 = comb.icmp ule %lhs, %rhs : i2
  %3 = comb.icmp ult %lhs, %rhs : i2
  %4 = comb.icmp uge %lhs, %rhs : i2
  %5 = comb.icmp ugt %lhs, %rhs : i2

  hw.output %0, %1, %2, %3, %4, %5 : i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: @icmp_eq
hw.module @icmp_eq(in %lhs: i2, in %rhs: i2, out out_eq: i1) {
  // CHECK-NEXT: %[[xor:.*]] = comb.xor %lhs, %rhs : i2
  // CHECK-NEXT: %[[eq0:.*]] = comb.extract %[[xor]] from 0 : (i2) -> i1
  // CHECK-NEXT: %[[eq1:.*]] = comb.extract %[[xor]] from 1 : (i2) -> i1
  // CHECK-NEXT: %[[reduce:.*]] = aig.and_inv not %[[eq0]], not %[[eq1]] : i1
  // CHECK-NEXT: hw.output %[[reduce]] : i1
  %0 = comb.icmp eq %lhs, %rhs : i2
  hw.output %0 : i1
}

// CHECK-LABEL: @icmp_ne
hw.module @icmp_ne(in %lhs: i2, in %rhs: i2, out out_ne: i1) {
  // CHECK-NEXT: %[[xor:.*]] = comb.xor %lhs, %rhs : i2
  // CHECK-NEXT: %[[eq0:.*]] = comb.extract %[[xor]] from 0 : (i2) -> i1
  // CHECK-NEXT: %[[eq1:.*]] = comb.extract %[[xor]] from 1 : (i2) -> i1
  // CHECK-NEXT: %[[reduce:.*]] = comb.or bin %[[eq0]], %[[eq1]] : i1
  // CHECK-NEXT: hw.output %[[reduce]] : i1
  %0 = comb.icmp ne %lhs, %rhs : i2
  hw.output %0 : i1
}

// CHECK-LABEL: @icmp_ule
hw.module @icmp_ule(in %lhs: i2, in %rhs: i2, out out_ule: i1) {
  // CHECK-NEXT: %[[lhs0:.*]] = comb.extract %lhs from 0 : (i2) -> i1
  // CHECK-NEXT: %[[lhs1:.*]] = comb.extract %lhs from 1 : (i2) -> i1
  // CHECK-NEXT: %[[rhs0:.*]] = comb.extract %rhs from 0 : (i2) -> i1
  // CHECK-NEXT: %[[rhs1:.*]] = comb.extract %rhs from 1 : (i2) -> i1
  // CHECK-NEXT: %[[eq1:.*]] = comb.xor %[[lhs1]], %[[rhs1]] : i1
  // CHECK-NEXT: %[[lt1:.*]] = aig.and_inv not %[[lhs1]], %[[rhs1]] : i1
  // CHECK-NEXT: %[[eq0:.*]] = comb.xor %[[lhs0]], %[[rhs0]] : i1
  // CHECK-NEXT: %[[lt0:.*]] = aig.and_inv not %[[lhs0]], %[[rhs0]] : i1
  // CHECK-NEXT: %[[le0:.*]] = aig.and_inv not %[[eq0]], not %[[lt0]] : i1
  // CHECK-NEXT: %[[le:.*]] = aig.and_inv not %[[eq1]], not %[[lt1]], not %[[le0]] : i1
  // CHECK-NEXT: %[[result:.*]] = aig.and_inv not %[[le]] : i1
  // CHECK-NEXT: hw.output %[[result]] : i1
  %0 = comb.icmp ule %lhs, %rhs : i2
  hw.output %0 : i1
}

hw.module @shl3(in %lhs: i3, in %rhs: i3, out out: i3) {
  // Pre-condition for LEC.
  %c3 = hw.constant 3 : i3
  %in_bound = comb.icmp ult %rhs, %c3 : i3
  verif.assume %in_bound : i1

  %0 = comb.shl %lhs, %rhs : i3
  hw.output %0 : i3
}

hw.module @div_mod_u(in %lhs: i4, out out: i4, out out_mod: i4) {
  %c4_i4 = hw.constant 4 : i4
  %0 = comb.divu %lhs, %c4_i4 : i4
  %1 = comb.modu %lhs, %c4_i4 : i4
  hw.output %0, %1 : i4, i4
}

hw.module @mul(in %lhs: i4, in %rhs: i4, out out: i4) {
  %0 = comb.mul %lhs, %rhs : i4
  hw.output %0 : i4
}

hw.module @parity(in %input: i4, out out: i1) {
  %0 = comb.parity %input : i4
  hw.output %0 : i1
}

hw.module @icmp_signed_compare(in %lhs: i2, in %rhs: i2, out out_ugt: i1, out out_uge: i1, out out_ult: i1, out out_ule: i1) {
  %ugt = comb.icmp sgt %lhs, %rhs : i2
  // CHECK: ee
  %uge = hw.constant false
  %ult = hw.constant false
  %ule = hw.constant false
  // %uge = comb.icmp sge %lhs, %rhs : i2
  // %ult = comb.icmp slt %lhs, %rhs : i2
  // %ule = comb.icmp sle %lhs, %rhs : i2
  hw.output %ugt, %uge, %ult, %ule : i1, i1, i1, i1
}
