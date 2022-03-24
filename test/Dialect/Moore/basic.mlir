// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK-LABEL: llhd.entity @test1
llhd.entity @test1() -> () {
  // CHECK-NEXT: [[CONST:%.*]] = moore.mir.constant 5 : !moore.int
  // CHECK-NEXT: [[VAR:%.*]] = moore.mir.vardecl "varname" = 3 : !moore.int
  // CHECK-NEXT: moore.mir.assign [[VAR]], [[CONST]] : !moore.int
  %0 = moore.mir.constant 5 : !moore.int
  %1 = moore.mir.vardecl "varname" = 3 : !moore.int
  moore.mir.assign %1, %0 : !moore.int
}

// CHECK-LABEL: func @Expressions
func @Expressions(%a: !moore.bit, %b: !moore.logic) {
  // CHECK: %0 = moore.mir.concat
  // CHECK: %1 = moore.mir.concat
  %0 = moore.mir.concat %a, %a : (!moore.bit, !moore.bit) -> !moore.packed<range<bit, 1:0>>
  %1 = moore.mir.concat %b, %b : (!moore.logic, !moore.logic) -> !moore.packed<range<logic, 1:0>>
  return
}
