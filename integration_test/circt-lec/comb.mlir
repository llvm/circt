// REQUIRES: libz3
// REQUIRES: circt-lec-jit

hw.module @basic(in %in: i1, out out: i1) {
  hw.output %in : i1
}

hw.module @not(in %in: i1, out out: i1) {
  %true = hw.constant true
  %out = comb.xor bin %in, %true : i1
  hw.output %out : i1
}

// comb.add
//  RUN: circt-lec %s -c1=adder -c2=completeAdder --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_ADD
//  COMB_ADD: c1 == c2

hw.module @adder(in %in1: i2, in %in2: i2, out out: i2) {
  %sum = comb.add bin %in1, %in2 : i2
  hw.output %sum : i2
}

hw.module @halfAdder(in %in1: i1, in %in2: i1, out carry: i1, out sum: i1) {
  %sum = comb.xor bin %in1, %in2 : i1
  %carry = comb.and bin %in1, %in2 : i1
  hw.output %carry, %sum: i1, i1
}

hw.module @completeAdder(in %in1: i2, in %in2 : i2, out out: i2) {
  %in1_0 = comb.extract %in1 from 0 : (i2) -> i1
  %in1_1 = comb.extract %in1 from 1 : (i2) -> i1
  %in2_0 = comb.extract %in2 from 0 : (i2) -> i1
  %in2_1 = comb.extract %in2 from 1 : (i2) -> i1
  %c1, %s1 = hw.instance "h1" @halfAdder(in1: %in1_0: i1, in2: %in2_0: i1) -> (carry: i1, sum: i1)
  %c2, %s2 = hw.instance "h2" @halfAdder(in1: %in1_1: i1, in2: %in2_1: i1) -> (carry: i1, sum: i1)
  %c3, %s3 = hw.instance "h3" @halfAdder(in1: %s2: i1, in2: %c1: i1) -> (carry: i1, sum: i1)
  %fullsum = comb.concat %s3, %s1 : i1, i1
  hw.output %fullsum : i2
}

// comb.and
//  RUN: circt-lec %s -c1=and -c2=decomposedAnd --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_AND
//  COMB_AND: c1 == c2

hw.module @and(in %in1: i1, in %in2: i1, out out: i1) {
  %out = comb.and bin %in1, %in2 : i1
  hw.output %out : i1
}

hw.module @decomposedAnd(in %in1: i1, in %in2: i1, out out: i1) {
  %not_in1 = hw.instance "n_in1" @not(in: %in1: i1) -> (out: i1)
  %not_in2 = hw.instance "n_in2" @not(in: %in2: i1) -> (out: i1)
  %not_and = comb.or bin %not_in1, %not_in2 : i1
  %and = hw.instance "and" @not(in: %not_and: i1) -> (out: i1)
  hw.output %and : i1
}

// comb.concat
// TODO

// comb.divs
// RUN: circt-lec %s -c1=divs_unsafe -c2=divs_unsafe --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_DIVS_UNSAFE
// RUN: circt-lec %s -c1=divs -c2=divs --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_DIVS
// COMB_DIVS_UNSAFE: c1 != c2
// COMB_DIVS: c1 == c2

hw.module @divs_unsafe(in %in1: i32, in %in2: i32, out out: i32) {
  %0 = comb.divs %in1, %in2 : i32
  hw.output %0 : i32
}

hw.module @divs(in %in1: i32, in %in2: i32, out out: i32) {
  %0 = hw.constant 0 : i32
  %1 = comb.icmp eq %in2, %0 : i32
  %2 = comb.divs %in1, %in2 : i32
  %3 = comb.mux %1, %0, %2 : i32
  hw.output %3 : i32
}

// comb.divu
// RUN: circt-lec %s -c1=divu_unsafe -c2=divu_unsafe --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_DIVU_UNSAFE
// RUN: circt-lec %s -c1=divu -c2=divu --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_DIVU
// COMB_DIVU_UNSAFE: c1 != c2
// COMB_DIVU: c1 == c2

hw.module @divu_unsafe(in %in1: i32, in %in2: i32, out out: i32) {
  %0 = comb.divu %in1, %in2 : i32
  hw.output %0 : i32
}

hw.module @divu(in %in1: i32, in %in2: i32, out out: i32) {
  %0 = hw.constant 0 : i32
  %1 = comb.icmp eq %in2, %0 : i32
  %2 = comb.divu %in1, %in2 : i32
  %3 = comb.mux %1, %0, %2 : i32
  hw.output %3 : i32
}

// comb.extract
// TODO

// comb.icmp
//  RUN: circt-lec %s -c1=eqInv -c2=constFalse --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_ICMPEQ
//  COMB_ICMPEQ: c1 == c2
hw.module @eqInv(in %a: i8, out eq: i1) {
  %ones = hw.constant -1 : i8
  %inv_a = comb.xor bin %a, %ones : i8
  %eq = comb.icmp bin eq %a, %inv_a : i8
  hw.output %eq : i1
}

hw.module @constFalse(in %a: i8, out eq: i1) {
  %eq = hw.constant false
  hw.output %eq : i1
}

// TODO: Other icmp predicates

// comb.mods
// RUN: circt-lec %s -c1=mods_unsafe -c2=mods_unsafe --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_MODS_UNSAFE
// RUN: circt-lec %s -c1=mods -c2=mods --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_MODS
// COMB_MODS_UNSAFE: c1 != c2
// COMB_MODS: c1 == c2

hw.module @mods_unsafe(in %in1: i32, in %in2: i32, out out: i32) {
  %0 = comb.mods %in1, %in2 : i32
  hw.output %0 : i32
}

hw.module @mods(in %in1: i32, in %in2: i32, out out: i32) {
  %0 = hw.constant 0 : i32
  %1 = comb.icmp eq %in2, %0 : i32
  %2 = comb.mods %in1, %in2 : i32
  %3 = comb.mux %1, %0, %2 : i32
  hw.output %3 : i32
}

// comb.modu
// RUN: circt-lec %s -c1=modu_unsafe -c2=modu_unsafe --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_MODU_UNSAFE
// RUN: circt-lec %s -c1=modu -c2=modu --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_MODU
// COMB_MODU_UNSAFE: c1 != c2
// COMB_MODU: c1 == c2

hw.module @modu_unsafe(in %in1: i32, in %in2: i32, out out: i32) {
  %0 = comb.modu %in1, %in2 : i32
  hw.output %0 : i32
}

hw.module @modu(in %in1: i32, in %in2: i32, out out: i32) {
  %0 = hw.constant 0 : i32
  %1 = comb.icmp eq %in2, %0 : i32
  %2 = comb.modu %in1, %in2 : i32
  %3 = comb.mux %1, %0, %2 : i32
  hw.output %3 : i32
}

// comb.mul
//  RUN: circt-lec %s -c1=mulBy2 -c2=addTwice --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_MUL
//  COMB_MUL: c1 == c2

hw.module @mulBy2(in %in: i2, out out: i2) {
  %two = hw.constant 2 : i2
  %res = comb.mul bin %in, %two : i2
  hw.output %res : i2
}

hw.module @addTwice(in %in: i2, out out: i2) {
  %res = comb.add bin %in, %in : i2
  hw.output %res : i2
}

// comb.mux
//  RUN: circt-lec %s -c1=mux -c2=decomposedMux --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_MUX
//  COMB_MUX: c1 == c2

hw.module @mux(in %cond: i1, in %tvalue: i8, in %fvalue: i8, out out: i8) {
  %res = comb.mux bin %cond, %tvalue, %fvalue : i8
  hw.output %res : i8
}

hw.module @decomposedMux(in %cond: i1, in %tvalue: i8, in %fvalue: i8, out out: i8) {
  %cond_bar = hw.instance "n" @not(in: %cond: i1) -> (out: i1)
  %lead_0 = hw.constant 0 : i7
  %c_t = comb.concat %lead_0, %cond : i7, i1
  %c_f = comb.concat %lead_0, %cond_bar : i7, i1
  %t = comb.mul bin %tvalue, %c_t : i8
  %f = comb.mul bin %fvalue, %c_f : i8
  %res = comb.add bin %t, %f : i8
  hw.output %res : i8
}

// comb.or
//  RUN: circt-lec %s -c1=or -c2=decomposedOr --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_OR
//  COMB_OR: c1 == c2

hw.module @or(in %in1: i1, in %in2: i1, out out: i1) {
  %out = comb.or bin %in1, %in2 : i1
  hw.output %out : i1
}

hw.module @decomposedOr(in %in1: i1, in %in2: i1, out out: i1) {
  %not_in1 = hw.instance "n_in1" @not(in: %in1: i1) -> (out: i1)
  %not_in2 = hw.instance "n_in2" @not(in: %in2: i1) -> (out: i1)
  %not_or = comb.and bin %not_in1, %not_in2 : i1
  %or = hw.instance "or" @not(in: %not_or: i1) -> (out: i1)
  hw.output %or : i1
}

// comb.parity
//  RUN: circt-lec %s -c1=parity -c2=decomposedParity --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_PARITY
//  COMB_PARITY: c1 == c2

hw.module @parity(in %in: i8, out out: i1) {
  %res = comb.parity bin %in : i8
  hw.output %res : i1
}

hw.module @decomposedParity(in %in: i8, out out: i1) {
  %b0 = comb.extract %in from 0 : (i8) -> i1
  %b1 = comb.extract %in from 1 : (i8) -> i1
  %b2 = comb.extract %in from 2 : (i8) -> i1
  %b3 = comb.extract %in from 3 : (i8) -> i1
  %b4 = comb.extract %in from 4 : (i8) -> i1
  %b5 = comb.extract %in from 5 : (i8) -> i1
  %b6 = comb.extract %in from 6 : (i8) -> i1
  %b7 = comb.extract %in from 7 : (i8) -> i1
  %res = comb.xor bin %b0, %b1, %b2, %b3, %b4, %b5, %b6, %b7 : i1
  hw.output %res : i1
}

// comb.replicate
//  RUN: circt-lec %s -c1=replicate -c2=decomposedReplicate --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_REPLICATE
//  COMB_REPLICATE: c1 == c2

hw.module @replicate(in %in: i2, out out: i8) {
  %res = comb.replicate %in : (i2) -> i8
  hw.output %res : i8
}

hw.module @decomposedReplicate(in %in: i2, out out: i8) {
  %res = comb.concat %in, %in, %in, %in : i2, i2, i2, i2
  hw.output %res : i8
}

// comb.shl
//  RUN: circt-lec %s -c1=shl -c2=decomposedShl --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_SHL
//  COMB_SHL: c1 == c2

hw.module @shl(in %in1: i2, in %in2: i2, out out: i2) {
  %res = comb.shl bin %in1, %in2 : i2
  hw.output %res : i2
}

hw.module @decomposedShl(in %in1: i2, in %in2: i2, out out: i2) {
  %zero = hw.constant 0 : i2
  %one = hw.constant 1 : i2
  %two = hw.constant 2 : i2
  // first possible shift
  %cond1 = comb.icmp bin ugt %in2, %zero : i2
  %mul1 = comb.mux bin %cond1, %two, %one : i2
  %shl1 = comb.mul bin %in1, %mul1 : i2
  // avoid subtraction underflow
  %cond1_1 = comb.icmp bin eq %in2, %zero : i2
  %sub1 = comb.mux bin %cond1_1, %zero, %one : i2
  %in2_2 = comb.sub bin %in2, %sub1 : i2
  // second possible shift
  %cond2 = comb.icmp bin ugt %in2_2, %zero : i2
  %mul2 = comb.mux bin %cond2, %two, %one : i2
  %shl2 = comb.mul bin %shl1, %mul2 : i2
  hw.output %shl2 : i2
}

// comb.shrs
// TODO

// comb.shru
// TODO

// comb.sub
//  RUN: circt-lec %s -c1=subtractor -c2=completeSubtractor --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_SUB
//  COMB_SUB: c1 == c2

hw.module @subtractor(in %in1: i8, in %in2: i8, out out: i8) {
  %diff = comb.sub bin %in1, %in2 : i8
  hw.output %diff : i8
}

hw.module @halfSubtractor(in %in1: i1, in %in2: i1, out borrow: i1, out diff: i1) {
  %diff = comb.xor bin %in1, %in2 : i1
  %not_in1 = hw.instance "n_in1" @not(in: %in1: i1) -> (out: i1)
  %borrow = comb.and bin %not_in1, %in2 : i1
  hw.output %borrow, %diff: i1, i1
}

hw.module @fullSubtractor(in %in1: i1, in %in2: i1, in %b_in: i1, out borrow: i1, out diff: i1) {
  %b1, %d1 = hw.instance "s1" @halfSubtractor(in1: %in1: i1, in2: %in2: i1) -> (borrow: i1, diff: i1)
  %b2, %d_out = hw.instance "s2" @halfSubtractor(in1: %d1: i1, in2: %b_in: i1) -> (borrow: i1, diff: i1)
  %b_out = comb.or bin %b1, %b2 : i1
  hw.output %b_out, %d_out: i1, i1
}

hw.module @completeSubtractor(in %in1: i8, in %in2 : i8, out out: i8) {
  %in1_0 = comb.extract %in1 from 0 : (i8) -> i1
  %in1_1 = comb.extract %in1 from 1 : (i8) -> i1
  %in1_2 = comb.extract %in1 from 2 : (i8) -> i1
  %in1_3 = comb.extract %in1 from 3 : (i8) -> i1
  %in1_4 = comb.extract %in1 from 4 : (i8) -> i1
  %in1_5 = comb.extract %in1 from 5 : (i8) -> i1
  %in1_6 = comb.extract %in1 from 6 : (i8) -> i1
  %in1_7 = comb.extract %in1 from 7 : (i8) -> i1
  %in2_0 = comb.extract %in2 from 0 : (i8) -> i1
  %in2_1 = comb.extract %in2 from 1 : (i8) -> i1
  %in2_2 = comb.extract %in2 from 2 : (i8) -> i1
  %in2_3 = comb.extract %in2 from 3 : (i8) -> i1
  %in2_4 = comb.extract %in2 from 4 : (i8) -> i1
  %in2_5 = comb.extract %in2 from 5 : (i8) -> i1
  %in2_6 = comb.extract %in2 from 6 : (i8) -> i1
  %in2_7 = comb.extract %in2 from 7 : (i8) -> i1
  %b0, %d0 = hw.instance "s0" @halfSubtractor(in1: %in1_0: i1, in2: %in2_0: i1) -> (borrow: i1, diff: i1)
  %b1, %d1 = hw.instance "s1" @fullSubtractor(in1: %in1_1: i1, in2: %in2_1: i1, b_in: %b0: i1) -> (borrow: i1, diff: i1)
  %b2, %d2 = hw.instance "s2" @fullSubtractor(in1: %in1_2: i1, in2: %in2_2: i1, b_in: %b1: i1) -> (borrow: i1, diff: i1)
  %b3, %d3 = hw.instance "s3" @fullSubtractor(in1: %in1_3: i1, in2: %in2_3: i1, b_in: %b2: i1) -> (borrow: i1, diff: i1)
  %b4, %d4 = hw.instance "s4" @fullSubtractor(in1: %in1_4: i1, in2: %in2_4: i1, b_in: %b3: i1) -> (borrow: i1, diff: i1)
  %b5, %d5 = hw.instance "s5" @fullSubtractor(in1: %in1_5: i1, in2: %in2_5: i1, b_in: %b4: i1) -> (borrow: i1, diff: i1)
  %b6, %d6 = hw.instance "s6" @fullSubtractor(in1: %in1_6: i1, in2: %in2_6: i1, b_in: %b5: i1) -> (borrow: i1, diff: i1)
  %b7, %d7 = hw.instance "s7" @fullSubtractor(in1: %in1_7: i1, in2: %in2_7: i1, b_in: %b6: i1) -> (borrow: i1, diff: i1)
  %diff = comb.concat %d7, %d6, %d5, %d4, %d3, %d2, %d1, %d0 : i1, i1, i1, i1, i1, i1, i1, i1
  hw.output %diff : i8
}

// comb.xor
// TODO
