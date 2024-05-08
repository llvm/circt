// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK-LABEL: moore.module @Foo
moore.module @Foo {
  // CHECK: moore.instance "foo" @Foo
  moore.instance "foo" @Foo
  // CHECK: %v1 = moore.variable : !moore.bit
  %v1 = moore.variable : !moore.bit
  %v2 = moore.variable : !moore.bit
  // CHECK: [[TMP:%.+]] = moore.variable name "v1" %v2 : !moore.bit
  moore.variable name "v1" %v2 : !moore.bit

  // CHECK: %w0 = moore.net wire : !moore.logic
  %w0 = moore.net wire : !moore.logic
  // CHECK: %w1 = moore.net wire %w0 : !moore.logic
  %w1 = moore.net wire %w0 : !moore.logic
  // CHECK: %w2 = moore.net uwire %w0 : !moore.logic
  %w2 = moore.net uwire %w0 : !moore.logic
  // CHECK: %w3 = moore.net tri %w0 : !moore.logic
  %w3 = moore.net tri %w0 : !moore.logic
  // CHECK: %w4 = moore.net triand %w0 : !moore.logic
  %w4 = moore.net triand %w0 : !moore.logic
  // CHECK: %w5 = moore.net trior %w0 : !moore.logic
  %w5 = moore.net trior %w0 : !moore.logic
  // CHECK: %w6 = moore.net wand %w0 : !moore.logic
  %w6 = moore.net wand %w0 : !moore.logic
  // CHECK: %w7 = moore.net wor %w0 : !moore.logic
  %w7 = moore.net wor %w0 : !moore.logic
  // CHECK: %w8 = moore.net trireg %w0 : !moore.logic
  %w8 = moore.net trireg %w0 : !moore.logic
  // CHECK: %w9 = moore.net tri0 %w0 : !moore.logic
  %w9 = moore.net tri0 %w0 : !moore.logic
  // CHECK: %w10 = moore.net tri1 %w0 : !moore.logic
  %w10 = moore.net tri1 %w0 : !moore.logic
  // CHECK: %w11 = moore.net supply0 : !moore.logic
  %w11 = moore.net supply0 : !moore.logic
  // CHECK: %w12 = moore.net supply1 : !moore.logic
  %w12 = moore.net supply1 : !moore.logic  

  // CHECK: moore.procedure initial {
  // CHECK: moore.procedure final {
  // CHECK: moore.procedure always {
  // CHECK: moore.procedure always_comb {
  // CHECK: moore.procedure always_latch {
  // CHECK: moore.procedure always_ff {
  moore.procedure initial {}
  moore.procedure final {}
  moore.procedure always {}
  moore.procedure always_comb {}
  moore.procedure always_latch {}
  moore.procedure always_ff {}

  // CHECK: moore.assign %v1, %v2 : !moore.bit
  moore.assign %v1, %v2 : !moore.bit

  moore.procedure always {
    // CHECK: moore.blocking_assign %v1, %v2 : !moore.bit
    moore.blocking_assign %v1, %v2 : !moore.bit
    // CHECK: moore.nonblocking_assign %v1, %v2 : !moore.bit
    moore.nonblocking_assign %v1, %v2 : !moore.bit
    // CHECK: %a = moore.variable  : !moore.int
    %a = moore.variable  : !moore.int
  }
}

// CHECK-LABEL: moore.module @Bar
moore.module @Bar {
}

// CHECK-LABEL: moore.module @Expressions
moore.module @Expressions {
  %b1 = moore.variable : !moore.bit
  %l1 = moore.variable : !moore.logic
  %b5 = moore.variable : !moore.packed<range<bit, 4:0>>
  %int = moore.variable : !moore.int
  %int2 = moore.variable : !moore.int
  %integer = moore.variable : !moore.integer
  %integer2 = moore.variable : !moore.integer
  %arr = moore.variable : !moore.unpacked<range<range<packed<range<bit, 7:0>>, 0:3>, 0:1>>

  // CHECK: moore.constant 0 : !moore.int
  moore.constant 0 : !moore.int
  // CHECK: moore.constant -2 : !moore.packed<range<bit, 1:0>>
  moore.constant 2 : !moore.packed<range<bit, 1:0>>
  // CHECK: moore.constant -2 : !moore.packed<range<bit<signed>, 1:0>>
  moore.constant -2 : !moore.packed<range<bit<signed>, 1:0>>

  // CHECK: moore.conversion %b5 : !moore.packed<range<bit, 4:0>> -> !moore.packed<range<logic, 4:0>>
  moore.conversion %b5 : !moore.packed<range<bit, 4:0>> -> !moore.packed<range<logic, 4:0>>

  // CHECK: moore.neg %int : !moore.int
  moore.neg %int : !moore.int
  // CHECK: moore.not %int : !moore.int
  moore.not %int : !moore.int

  // CHECK: moore.reduce_and %int : !moore.int -> !moore.bit
  moore.reduce_and %int : !moore.int -> !moore.bit
  // CHECK: moore.reduce_or %int : !moore.int -> !moore.bit
  moore.reduce_or %int : !moore.int -> !moore.bit
  // CHECK: moore.reduce_xor %int : !moore.int -> !moore.bit
  moore.reduce_xor %int : !moore.int -> !moore.bit
  // CHECK: moore.reduce_xor %integer : !moore.integer -> !moore.logic
  moore.reduce_xor %integer : !moore.integer -> !moore.logic

  // CHECK: moore.bool_cast %int : !moore.int -> !moore.bit
  moore.bool_cast %int : !moore.int -> !moore.bit
  // CHECK: moore.bool_cast %integer : !moore.integer -> !moore.logic
  moore.bool_cast %integer : !moore.integer -> !moore.logic

  // CHECK: moore.add %int, %int2 : !moore.int
  moore.add %int, %int2 : !moore.int
  // CHECK: moore.sub %int, %int2 : !moore.int
  moore.sub %int, %int2 : !moore.int
  // CHECK: moore.mul %int, %int2 : !moore.int
  moore.mul %int, %int2 : !moore.int
  // CHECK: moore.div %int, %int2 : !moore.int
  moore.div %int, %int2 : !moore.int
  // CHECK: moore.mod %int, %int2 : !moore.int
  moore.mod %int, %int2 : !moore.int

  // CHECK: moore.and %int, %int2 : !moore.int
  moore.and %int, %int2 : !moore.int
  // CHECK: moore.or %int, %int2 : !moore.int
  moore.or %int, %int2 : !moore.int
  // CHECK: moore.xor %int, %int2 : !moore.int
  moore.xor %int, %int2 : !moore.int

  // CHECK: moore.shl %l1, %b1 : !moore.logic, !moore.bit
  moore.shl %l1, %b1 : !moore.logic, !moore.bit
  // CHECK: moore.shr %l1, %b1 : !moore.logic, !moore.bit
  moore.shr %l1, %b1 : !moore.logic, !moore.bit
  // CHECK: moore.ashr %b5, %b1 : !moore.packed<range<bit, 4:0>>, !moore.bit
  moore.ashr %b5, %b1 : !moore.packed<range<bit, 4:0>>, !moore.bit

  // CHECK: moore.eq %int, %int2 : !moore.int -> !moore.bit
  moore.eq %int, %int2 : !moore.int -> !moore.bit
  // CHECK: moore.ne %int, %int2 : !moore.int -> !moore.bit
  moore.ne %int, %int2 : !moore.int -> !moore.bit
  // CHECK: moore.ne %integer, %integer2 : !moore.integer -> !moore.logic
  moore.ne %integer, %integer2 : !moore.integer -> !moore.logic
  // CHECK: moore.case_eq %int, %int2 : !moore.int
  moore.case_eq %int, %int2 : !moore.int
  // CHECK: moore.case_ne %int, %int2 : !moore.int
  moore.case_ne %int, %int2 : !moore.int
  // CHECK: moore.wildcard_eq %int, %int2 : !moore.int -> !moore.bit
  moore.wildcard_eq %int, %int2 : !moore.int -> !moore.bit
  // CHECK: moore.wildcard_ne %int, %int2 : !moore.int -> !moore.bit
  moore.wildcard_ne %int, %int2 : !moore.int -> !moore.bit
  // CHECK: moore.wildcard_ne %integer, %integer2 : !moore.integer -> !moore.logic
  moore.wildcard_ne %integer, %integer2 : !moore.integer -> !moore.logic

  // CHECK: moore.lt %int, %int2 : !moore.int -> !moore.bit
  moore.lt %int, %int2 : !moore.int -> !moore.bit
  // CHECK: moore.le %int, %int2 : !moore.int -> !moore.bit
  moore.le %int, %int2 : !moore.int -> !moore.bit
  // CHECK: moore.gt %int, %int2 : !moore.int -> !moore.bit
  moore.gt %int, %int2 : !moore.int -> !moore.bit
  // CHECK: moore.ge %int, %int2 : !moore.int -> !moore.bit
  moore.ge %int, %int2 : !moore.int -> !moore.bit
  // CHECK: moore.ge %integer, %integer2 : !moore.integer -> !moore.logic
  moore.ge %integer, %integer2 : !moore.integer -> !moore.logic

  // CHECK: moore.concat %b1 : (!moore.bit) -> !moore.packed<range<bit, 0:0>>
  moore.concat %b1 : (!moore.bit) -> !moore.packed<range<bit, 0:0>>
  // CHECK: moore.concat %b5, %b1 : (!moore.packed<range<bit, 4:0>>, !moore.bit) -> !moore.packed<range<bit, 5:0>>
  moore.concat %b5, %b1 : (!moore.packed<range<bit, 4:0>>, !moore.bit) -> !moore.packed<range<bit, 5:0>>
  // CHECK: moore.concat %l1, %l1, %l1 : (!moore.logic, !moore.logic, !moore.logic) -> !moore.packed<range<logic, 2:0>>
  moore.concat %l1, %l1, %l1 : (!moore.logic, !moore.logic, !moore.logic) -> !moore.packed<range<logic, 2:0>>
  // CHECK: [[VAL:%.*]] = moore.concat %b1 : (!moore.bit) -> !moore.packed<range<bit, 0:0>>
  // CHECK: moore.replicate [[VAL]] : (!moore.packed<range<bit, 0:0>>) -> !moore.packed<range<bit, 3:0>>
  %0 = moore.concat %b1 : (!moore.bit) -> !moore.packed<range<bit, 0:0>>
  moore.replicate %0 : (!moore.packed<range<bit, 0:0>>) -> !moore.packed<range<bit, 3:0>>

  // CHECK: moore.extract %b5 from %b1 : !moore.packed<range<bit, 4:0>>, !moore.bit -> !moore.bit
  moore.extract %b5 from %b1 : !moore.packed<range<bit, 4:0>>, !moore.bit -> !moore.bit
  // CHECK: [[VAL1:%.*]] = moore.constant 0 : !moore.int
  // CHECK: [[VAL2:%.*]] = moore.extract %arr from [[VAL1]] : !moore.unpacked<range<range<packed<range<bit, 7:0>>, 0:3>, 0:1>>, !moore.int -> !moore.unpacked<range<packed<range<bit, 7:0>>, 0:3>>
  %1 = moore.constant 0 : !moore.int
  %2 = moore.extract %arr from %1 : !moore.unpacked<range<range<packed<range<bit, 7:0>>, 0:3>, 0:1>>, !moore.int -> !moore.unpacked<range<packed<range<bit, 7:0>>, 0:3>>
  // CHECK: [[VAL3:%.*]] = moore.constant 3 : !moore.int
  // CHECK: [[VAL4:%.*]] = moore.extract [[VAL2]] from [[VAL3]] : !moore.unpacked<range<packed<range<bit, 7:0>>, 0:3>>, !moore.int -> !moore.packed<range<bit, 7:0>>
  %3 = moore.constant 3 : !moore.int
  %4 = moore.extract %2 from %3 : !moore.unpacked<range<packed<range<bit, 7:0>>, 0:3>>, !moore.int -> !moore.packed<range<bit, 7:0>>
  // CHECK: [[VAL5:%.*]] = moore.constant 2 : !moore.int
  // CHECK: moore.extract [[VAL4]] from [[VAL5]] : !moore.packed<range<bit, 7:0>>, !moore.int -> !moore.packed<range<bit, 6:2>>
  %5 = moore.constant 2 : !moore.int
  moore.extract %4 from %5 : !moore.packed<range<bit, 7:0>>, !moore.int -> !moore.packed<range<bit, 6:2>>
}
