// RUN: circt-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: @cpus
// CHECK-SAME: !rtgtest.cpu
rtg.target @cpus : !rtg.dict<cpu: !rtgtest.cpu> {
  // CHECK: rtgtest.cpu_decl <0>
  %0 = rtgtest.cpu_decl <0>
  rtg.yield %0 : !rtgtest.cpu
}

rtg.test @misc : !rtg.dict<> {
  // CHECK: rtgtest.constant_test i32 {value = "str"}
  %0 = rtgtest.constant_test i32 {value = "str"}
}

// CHECK-LABEL: rtg.test @registers
// CHECK-SAME: !rtgtest.ireg
rtg.test @registers : !rtg.dict<reg: !rtgtest.ireg> {
^bb0(%reg: !rtgtest.ireg):
  // CHECK: rtg.fixed_reg #rtgtest.zero : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.ra : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.sp : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.gp : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.tp : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.t0 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.t1 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.t2 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.s0 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.s1 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.a0 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.a1 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.a2 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.a3 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.a4 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.a5 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.a6 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.a7 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.s2 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.s3 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.s4 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.s5 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.s6 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.s7 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.s8 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.s9 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.s10 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.s11 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.t3 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.t4 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.t5 : !rtgtest.ireg
  // CHECK: rtg.fixed_reg #rtgtest.t6 : !rtgtest.ireg
  rtg.fixed_reg #rtgtest.zero
  rtg.fixed_reg #rtgtest.ra
  rtg.fixed_reg #rtgtest.sp
  rtg.fixed_reg #rtgtest.gp
  rtg.fixed_reg #rtgtest.tp
  rtg.fixed_reg #rtgtest.t0
  rtg.fixed_reg #rtgtest.t1
  rtg.fixed_reg #rtgtest.t2
  rtg.fixed_reg #rtgtest.s0
  rtg.fixed_reg #rtgtest.s1
  rtg.fixed_reg #rtgtest.a0
  rtg.fixed_reg #rtgtest.a1
  rtg.fixed_reg #rtgtest.a2
  rtg.fixed_reg #rtgtest.a3
  rtg.fixed_reg #rtgtest.a4
  rtg.fixed_reg #rtgtest.a5
  rtg.fixed_reg #rtgtest.a6
  rtg.fixed_reg #rtgtest.a7
  rtg.fixed_reg #rtgtest.s2
  rtg.fixed_reg #rtgtest.s3
  rtg.fixed_reg #rtgtest.s4
  rtg.fixed_reg #rtgtest.s5
  rtg.fixed_reg #rtgtest.s6
  rtg.fixed_reg #rtgtest.s7
  rtg.fixed_reg #rtgtest.s8
  rtg.fixed_reg #rtgtest.s9
  rtg.fixed_reg #rtgtest.s10
  rtg.fixed_reg #rtgtest.s11
  rtg.fixed_reg #rtgtest.t3
  rtg.fixed_reg #rtgtest.t4
  rtg.fixed_reg #rtgtest.t5
  rtg.fixed_reg #rtgtest.t6

  // CHECK: rtg.virtual_reg [#rtgtest.ra : !rtgtest.ireg, #rtgtest.sp : !rtgtest.ireg]
  rtg.virtual_reg [#rtgtest.ra, #rtgtest.sp]
}

// -----

rtg.test @emptyAllowed : !rtg.dict<> {
  // expected-error @below {{must have at least one allowed register}}
  rtg.virtual_reg []
}

// -----

rtg.test @invalidAllowedAttr : !rtg.dict<> {
  // expected-error @below {{allowed register attributes must be of RegisterAttrInterface}}
  rtg.virtual_reg ["invalid"]
}
