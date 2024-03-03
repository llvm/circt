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

// CHECK-LABEL: llhd.entity @test1
llhd.entity @test1() -> () {
  // CHECK-NEXT: [[CONST:%.*]] = moore.mir.constant 5 : !moore.int
  %0 = moore.mir.constant 5 : !moore.int
}

// CHECK-LABEL: func @Expressions
func.func @Expressions(%a: !moore.bit, %b: !moore.logic, %c: !moore.packed<range<bit, 4:0>>) {
  // CHECK: %0 = moore.mir.concat
  // CHECK: %1 = moore.mir.concat
  %0 = moore.mir.concat %a, %a : (!moore.bit, !moore.bit) -> !moore.packed<range<bit, 1:0>>
  %1 = moore.mir.concat %b, %b : (!moore.logic, !moore.logic) -> !moore.packed<range<logic, 1:0>>

  // CHECK: %2 = moore.mir.shl %
  // CHECK: %3 = moore.mir.shl arithmetic %
  %2 = moore.mir.shl %b, %a : !moore.logic, !moore.bit
  %3 = moore.mir.shl arithmetic %c, %a : !moore.packed<range<bit, 4:0>>, !moore.bit

  // CHECK: %4 = moore.mir.shr %
  // CHECK: %5 = moore.mir.shr arithmetic %
  %4 = moore.mir.shr %b, %a : !moore.logic, !moore.bit
  %5 = moore.mir.shr arithmetic %c, %a : !moore.packed<range<bit, 4:0>>, !moore.bit

  return
}
