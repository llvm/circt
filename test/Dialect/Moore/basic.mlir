// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK-LABEL: llhd.entity @test1
llhd.entity @test1() -> () {
    // CHECK-NEXT: [[CONST:%.*]] = moore.mir.constant 5 : !moore.rvalue<!moore.sv.int>
    // CHECK-NEXT: [[VAR:%.*]] = moore.mir.vardecl "varname" = 3 : !moore.lvalue<!moore.sv.int>
    // CHECK-NEXT: moore.mir.assign [[VAR]], [[CONST]] : !moore.lvalue<!moore.sv.int>, !moore.rvalue<!moore.sv.int>
    %0 = moore.mir.constant 5 : !moore.rvalue<!moore.sv.int>
    %1 = moore.mir.vardecl "varname" = 3 : !moore.lvalue<!moore.sv.int>
    moore.mir.assign %1, %0 : !moore.lvalue<!moore.sv.int>, !moore.rvalue<!moore.sv.int>
}
