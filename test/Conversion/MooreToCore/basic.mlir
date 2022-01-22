// RUN: circt-opt %s -convert-moore-to-core -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK-LABEL: llhd.entity @test1
llhd.entity @test1() -> () {
    // CHECK-NEXT: %c5_i32 = hw.constant 5 : i32
    %0 = moore.mir.constant 5 : !moore.rvalue<!moore.sv.int>
    // CHECK-NEXT: %c3_i32 = hw.constant 3 : i32
    // CHECK-NEXT: [[SIG:%.*]] = llhd.sig "varname" %c3_i32 : i32
    %1 = moore.mir.vardecl "varname" = 3 : !moore.lvalue<!moore.sv.int>
    // CHECK-NEXT: [[TIME:%.*]] = llhd.constant_time <0s, 0d, 1e>
    // CHECK-NEXT: llhd.drv [[SIG]], %c5_i32 after [[TIME]] : !llhd.sig<i32>
    moore.mir.assign %1, %0 : !moore.lvalue<!moore.sv.int>, !moore.rvalue<!moore.sv.int>
}
