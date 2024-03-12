// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: func @bitvectors
func.func @bitvectors() {
  // CHECK: %c5_bv32 = smt.bv.constant #smt.bv<5> : !smt.bv<32> {smt.some_attr}
  %c5_bv32 = smt.bv.constant #smt.bv<5> : !smt.bv<32> {smt.some_attr}
  // CHECK: %c92_bv8 = smt.bv.constant #smt.bv<92> : !smt.bv<8> {smt.some_attr}
  %c92_bv8 = smt.bv.constant #smt.bv<0x5c> : !smt.bv<8> {smt.some_attr}
  // CHECK: %c-1_bv8 = smt.bv.constant #smt.bv<-1> : !smt.bv<8>
  %c-1_bv8 = smt.bv.constant #smt.bv<-1> : !smt.bv<8>

  // CHECK: [[C0:%.+]] = smt.bv.constant #smt.bv<0> : !smt.bv<32>
  %c = smt.bv.constant #smt.bv<0> : !smt.bv<32>

  // CHECK: %{{.*}} = smt.bv.neg [[C0]] {smt.some_attr} : !smt.bv<32>
  %0 = smt.bv.neg %c {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.add [[C0]], [[C0]] {smt.some_attr} : !smt.bv<32>
  %1 = smt.bv.add %c, %c {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.mul [[C0]], [[C0]] {smt.some_attr} : !smt.bv<32>
  %3 = smt.bv.mul %c, %c {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.urem [[C0]], [[C0]] {smt.some_attr} : !smt.bv<32>
  %4 = smt.bv.urem %c, %c {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.srem [[C0]], [[C0]] {smt.some_attr} : !smt.bv<32>
  %5 = smt.bv.srem %c, %c {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.smod [[C0]], [[C0]] {smt.some_attr} : !smt.bv<32>
  %7 = smt.bv.smod %c, %c {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.shl [[C0]], [[C0]] {smt.some_attr} : !smt.bv<32>
  %8 = smt.bv.shl %c, %c {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.lshr [[C0]], [[C0]] {smt.some_attr} : !smt.bv<32>
  %9 = smt.bv.lshr %c, %c {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.ashr [[C0]], [[C0]] {smt.some_attr} : !smt.bv<32>
  %10 = smt.bv.ashr %c, %c {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.udiv [[C0]], [[C0]] {smt.some_attr} : !smt.bv<32>
  %11 = smt.bv.udiv %c, %c {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.sdiv [[C0]], [[C0]] {smt.some_attr} : !smt.bv<32>
  %12 = smt.bv.sdiv %c, %c {smt.some_attr} : !smt.bv<32>

  // CHECK: %{{.*}} = smt.bv.not [[C0]] {smt.some_attr} : !smt.bv<32>
  %13 = smt.bv.not %c {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.and [[C0]], [[C0]] {smt.some_attr} : !smt.bv<32>
  %14 = smt.bv.and %c, %c {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.or [[C0]], [[C0]] {smt.some_attr} : !smt.bv<32>
  %15 = smt.bv.or %c, %c {smt.some_attr} : !smt.bv<32>
  // CHECK: %{{.*}} = smt.bv.xor [[C0]], [[C0]] {smt.some_attr} : !smt.bv<32>
  %16 = smt.bv.xor %c, %c {smt.some_attr} : !smt.bv<32>

  return
}
