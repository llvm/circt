// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK-LABEL: llhd.entity @test1
llhd.entity @test1() -> () {
  // CHECK-NEXT: [[CONST:%.*]] = moore.mir.constant 5 : !moore.rvalue<!moore.int>
  // CHECK-NEXT: [[VAR:%.*]] = moore.mir.vardecl "varname" = 3 : !moore.lvalue<!moore.int>
  // CHECK-NEXT: moore.mir.assign [[VAR]], [[CONST]] : !moore.lvalue<!moore.int>, !moore.rvalue<!moore.int>
  %0 = moore.mir.constant 5 : !moore.rvalue<!moore.int>
  %1 = moore.mir.vardecl "varname" = 3 : !moore.lvalue<!moore.int>
  moore.mir.assign %1, %0 : !moore.lvalue<!moore.int>, !moore.rvalue<!moore.int>
}
