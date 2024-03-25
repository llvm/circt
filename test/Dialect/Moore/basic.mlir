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
}
