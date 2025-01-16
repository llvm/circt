// RUN: circt-opt %s | FileCheck %s

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
  // CHECK: rtgtest.reg zero
  // CHECK: rtgtest.reg ra
  // CHECK: rtgtest.reg sp
  // CHECK: rtgtest.reg gp
  // CHECK: rtgtest.reg tp
  // CHECK: rtgtest.reg t0
  // CHECK: rtgtest.reg t1
  // CHECK: rtgtest.reg t2
  // CHECK: rtgtest.reg s0
  // CHECK: rtgtest.reg s1
  // CHECK: rtgtest.reg a0
  // CHECK: rtgtest.reg a1
  // CHECK: rtgtest.reg a2
  // CHECK: rtgtest.reg a3
  // CHECK: rtgtest.reg a4
  // CHECK: rtgtest.reg a5
  // CHECK: rtgtest.reg a6
  // CHECK: rtgtest.reg a7
  // CHECK: rtgtest.reg s2
  // CHECK: rtgtest.reg s3
  // CHECK: rtgtest.reg s4
  // CHECK: rtgtest.reg s5
  // CHECK: rtgtest.reg s6
  // CHECK: rtgtest.reg s7
  // CHECK: rtgtest.reg s8
  // CHECK: rtgtest.reg s9
  // CHECK: rtgtest.reg s10
  // CHECK: rtgtest.reg s11
  // CHECK: rtgtest.reg t3
  // CHECK: rtgtest.reg t4
  // CHECK: rtgtest.reg t5
  // CHECK: rtgtest.reg t6
  // CHECK: rtgtest.reg Virtual
  %1 = rtgtest.reg zero
  %2 = rtgtest.reg ra
  %3 = rtgtest.reg sp
  %4 = rtgtest.reg gp
  %5 = rtgtest.reg tp
  %6 = rtgtest.reg t0
  %7 = rtgtest.reg t1
  %8 = rtgtest.reg t2
  %9 = rtgtest.reg s0
  %10 = rtgtest.reg s1
  %11 = rtgtest.reg a0
  %12 = rtgtest.reg a1
  %13 = rtgtest.reg a2
  %14 = rtgtest.reg a3
  %15 = rtgtest.reg a4
  %16 = rtgtest.reg a5
  %17 = rtgtest.reg a6
  %18 = rtgtest.reg a7
  %19 = rtgtest.reg s2
  %20 = rtgtest.reg s3
  %21 = rtgtest.reg s4
  %22 = rtgtest.reg s5
  %23 = rtgtest.reg s6
  %24 = rtgtest.reg s7
  %25 = rtgtest.reg s8
  %26 = rtgtest.reg s9
  %27 = rtgtest.reg s10
  %28 = rtgtest.reg s11
  %29 = rtgtest.reg t3
  %30 = rtgtest.reg t4
  %31 = rtgtest.reg t5
  %32 = rtgtest.reg t6
  %33 = rtgtest.reg Virtual
}
