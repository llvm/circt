// RUN: circt-opt --llhd-temporal-code-motion %s | FileCheck %s

hw.module @basic(in %cond: i1) {
  %false = hw.constant false
  %true = hw.constant true
  %c0_i4 = hw.constant 0 : i4
  %c0_i5 = hw.constant 0 : i5
  // CHECK: [[V0:%.+]] = llhd.constant_time <0ns, 1d, 0e>
  // CHECK: [[V1:%.+]] = llhd.constant_time <0ns, 0d, 1e>
  %0 = llhd.constant_time <0ns, 1d, 0e>
  %1 = llhd.constant_time <0ns, 0d, 1e>
  // CHECK: [[V2:%.+]] = llhd.sig "a"
  // CHECK: [[V3:%.+]] = llhd.sig "b"
  // CHECK: [[V4:%.+]] = llhd.sig "c"
  // CHECK: [[V5:%.+]] = llhd.sig "d"
  // CHECK: [[V6:%.+]] = llhd.sig "e"
  // CHECK: [[V7:%.+]] = llhd.sig "f"
  // CHECK: [[V8:%.+]] = llhd.sig "g"
  // CHECK: [[V9:%.+]] = llhd.sig "h"
  // CHECK: [[V10:%.+]] = llhd.sig "i"
  // CHECK: [[V11:%.+]] = llhd.sig "j"
  // CHECK: [[V12:%.+]] = llhd.sig "k"
  // CHECK: [[V13:%.+]] = llhd.sig "l"
  // CHECK: [[V13_1:%.+]] = llhd.sig "m"
  // CHECK: [[V13_2:%.+]] = llhd.sig "n"
  // CHECK: [[V13_3:%.+]] = llhd.sig "o"
  %2 = llhd.sig "a" %false : i1
  %3 = llhd.sig "b" %false : i1
  %4 = llhd.sig "c" %false : i1
  %5 = llhd.sig "d" %false : i1
  %6 = llhd.sig "e" %false : i1
  %7 = llhd.sig "f" %false : i1
  %8 = llhd.sig "g" %c0_i4 : i4
  %9 = llhd.sig "h" %c0_i4 : i4
  %10 = llhd.sig "i" %c0_i4 : i4
  %11 = llhd.sig "j" %false : i1
  %12 = llhd.sig "k" %c0_i5 : i5
  %13 = llhd.sig "l" %c0_i5 : i5
  %14 = llhd.sig "m" %c0_i5 : i5
  %15 = llhd.sig "n" %c0_i5 : i5
  %16 = llhd.sig "o" %c0_i5 : i5

  // COM: Check that an auxillary block is created and all drives are moved to
  // COM: the exit block with the correct enable condition
  // CHECK: llhd.process
  llhd.process {
  // CHECK: cf.br ^[[BB1:.+]]
    cf.br ^bb1
  // CHECK: ^[[BB1]]:
  ^bb1:
    // CHECK: llhd.wait ({{.*}}), ^[[BB2:.+]]
    llhd.wait (%12, %4, %6, %9, %5, %7, %8 : !hw.inout<i5>, !hw.inout<i1>, !hw.inout<i1>, !hw.inout<i4>, !hw.inout<i1>, !hw.inout<i1>, !hw.inout<i4>), ^bb2
  // CHECK: ^[[BB2]]:
  ^bb2:
    // CHECK: [[V14:%.+]] = llhd.prb [[V12]]
    // CHECK: [[V15:%.+]] = llhd.prb [[V4]]
    // CHECK: [[V16:%.+]] = llhd.prb [[V6]]
    // CHECK: [[V17:%.+]] = llhd.prb [[V9]]
    // CHECK: [[V18:%.+]] = comb.concat %false{{.*}}, [[V17]] : i1, i4
    // CHECK: [[V19:%.+]] = llhd.prb [[V5]]
    // CHECK: [[V20:%.+]] = llhd.prb [[V7]]
    // CHECK: [[V21:%.+]] = llhd.prb [[V12]]
    // CHECK: [[V22:%.+]] = llhd.prb [[V8]]
    // CHECK: [[V23:%.+]] = comb.concat %false{{.*}}, [[V22]] : i1, i4
    // CHECK: [[V24:%.+]] = comb.sub [[V21]], [[V23]] : i5
    // CHECK: [[V25:%.+]] = llhd.prb [[V12]]
    // CHECK: [[V26:%.+]] = llhd.prb [[V8]]
    // CHECK: [[V27:%.+]] = comb.concat %false{{.*}}, [[V26]] : i1, i4
    // CHECK: [[V28:%.+]] = comb.add [[V25]], [[V27]] : i5
    // CHECK: cf.cond_br [[V15]], ^[[BB3:.+]], ^[[BB4:.+]]
    %25 = llhd.prb %12 : !hw.inout<i5>
    llhd.drv %12, %25 after %1 : !hw.inout<i5>
    %26 = llhd.prb %4 : !hw.inout<i1>
    %27 = llhd.prb %6 : !hw.inout<i1>
    %28 = llhd.prb %9 : !hw.inout<i4>
    %29 = comb.concat %false, %28 : i1, i4
    %30 = llhd.prb %5 : !hw.inout<i1>
    %31 = llhd.prb %7 : !hw.inout<i1>
    %32 = llhd.prb %12 : !hw.inout<i5>
    %33 = llhd.prb %8 : !hw.inout<i4>
    %34 = comb.concat %false, %33 : i1, i4
    %35 = comb.sub %32, %34 : i5
    %36 = llhd.prb %12 : !hw.inout<i5>
    %37 = llhd.prb %8 : !hw.inout<i4>
    %38 = comb.concat %false, %37 : i1, i4
    %39 = comb.add %36, %38 : i5
    cf.cond_br %26, ^bb3, ^bb4
  // CHECK: ^[[BB3]]:
  ^bb3:
    llhd.drv %13, %c0_i5 after %1 if %cond : !hw.inout<i5>
    // CHECK: cf.br ^[[BB10:.+]]
    cf.br ^bb1
  // CHECK: ^[[BB4]]:
  ^bb4:
    // CHECK: cf.cond_br [[V16]], ^[[BB5:.+]], ^[[BB6:.+]]
    cf.cond_br %27, ^bb5, ^bb6
  // CHECK: ^[[BB5]]:
  ^bb5:
    llhd.drv %14, %29 after %1 : !hw.inout<i5>
    // CHECK: cf.br ^[[BB10]]
    cf.br ^bb1
  // CHECK: ^[[BB6]]:
  ^bb6:
    // CHECK: cf.cond_br [[V19]], ^[[BB7:.+]], ^[[BB10]]
    cf.cond_br %30, ^bb7, ^bb1
  // CHECK: ^[[BB7]]:
  ^bb7:
    // CHECK: cf.cond_br [[V20]], ^[[BB8:.+]], ^[[BB9:.+]]
    cf.cond_br %31, ^bb8, ^bb9
  // CHECK: ^[[BB8]]:
  ^bb8:
    llhd.drv %15, %35 after %1 : !hw.inout<i5>
    // CHECK: cf.br ^[[BB10]]
    cf.br ^bb1
  // CHECK: ^[[BB9]]:
  ^bb9:
    llhd.drv %16, %39 after %1 : !hw.inout<i5>
    // CHECK: cf.br ^[[BB10]]
    cf.br ^bb1
    // CHECK: ^[[BB10]]:
    // CHECK: llhd.drv [[V12]], [[V14]] after [[V1]] if %true{{.*}} : !hw.inout<i5>

    // CHECK: [[V29:%.+]] = comb.and %true{{.*}}, [[V15]] : i1
    // CHECK: [[V30:%.+]] = comb.or %false{{.*}}, [[V29]] : i1
    // CHECK: [[V31:%.+]] = comb.and %cond, [[V30]] : i1
    // CHECK: llhd.drv [[V13]], %c0_i5 after [[V1]] if [[V31]] : !hw.inout<i5>

    // CHECK: [[V33:%.+]] = comb.xor [[V15]], %true{{.*}} : i1
    // CHECK: [[V34:%.+]] = comb.and %true{{.*}}, [[V33]] : i1
    // CHECK: [[V35:%.+]] = comb.or %false{{.*}}, [[V34]] : i1
    // CHECK: [[V36:%.+]] = comb.and [[V35]], [[V16]] : i1
    // CHECK: [[V37:%.+]] = comb.or %false{{.*}}, [[V36]] : i1
    // CHECK: llhd.drv [[V13_1]], [[V18]] after [[V1]] if [[V37]] : !hw.inout<i5>

    // CHECK: [[V40:%.+]] = comb.xor [[V16]], %true{{.*}} : i1
    // CHECK: [[V41:%.+]] = comb.and [[V35]], [[V40]] : i1
    // CHECK: [[V42:%.+]] = comb.or %false{{.*}}, [[V41]] : i1
    // CHECK: [[V43:%.+]] = comb.and [[V42]], [[V19]] : i1
    // CHECK: [[V44:%.+]] = comb.or %false{{.*}}, [[V43]] : i1
    // CHECK: [[V45:%.+]] = comb.and [[V44]], [[V20]] : i1
    // CHECK: [[V46:%.+]] = comb.or %false{{.*}}, [[V45]] : i1
    // CHECK: llhd.drv [[V13_2]], [[V24]] after [[V1]] if [[V46]] : !hw.inout<i5>

    // CHECK: [[V49:%.+]] = comb.xor [[V20]], %true{{.*}} : i1
    // CHECK: [[V50:%.+]] = comb.and [[V44]], [[V49]] : i1
    // CHECK: [[V51:%.+]] = comb.or %false{{.*}}, [[V50]] : i1
    // CHECK: llhd.drv [[V13_3]], [[V28]] after [[V1]] if [[V51]] : !hw.inout<i5>
    // CHECK: cf.br ^[[BB1]]
  }

  // COM: check drive coalescing behavior
  llhd.process {
    cf.br ^bb1
  ^bb1:
    llhd.wait ^bb2
  ^bb2:
    // CHECK:      [[V20:%.+]] = llhd.prb [[V13]]
    // CHECK-NEXT: [[V21:%.+]] = llhd.prb [[V13_1]]
    %20 = llhd.prb %13 : !hw.inout<i5>
    %21 = llhd.prb %14 : !hw.inout<i5>

    // CHECK-NEXT: llhd.drv [[V12]], [[V21]] after [[V1]] : !hw.inout<i5>
    llhd.drv %12, %20 after %1 : !hw.inout<i5>
    llhd.drv %12, %21 after %1 : !hw.inout<i5>

    // CHECK-NEXT: [[V22:%.+]] = comb.mux %cond, [[V21]], [[V20]]
    // CHECK-NEXT: llhd.drv [[V13]], [[V22]] after [[V1]] : !hw.inout<i5>
    llhd.drv %13, %20 after %1 : !hw.inout<i5>
    llhd.drv %13, %21 after %1 if %cond : !hw.inout<i5>

    // CHECK-NEXT: [[V23:%.+]] = comb.xor %cond, %true
    // CHECK-NEXT: [[V24:%.+]] = comb.mux %cond, [[V21]], [[V20]]
    // CHECK-NEXT: [[V25:%.+]] = comb.or %cond, [[V23]]
    // CHECK-NEXT: llhd.drv [[V13_1]], [[V24]] after [[V1]] if [[V25]] : !hw.inout<i5>
    %22 = comb.xor %cond, %true : i1
    llhd.drv %14, %20 after %1 if %22 : !hw.inout<i5>
    llhd.drv %14, %21 after %1 if %cond : !hw.inout<i5>

    // CHECK-NEXT: cf.br
    cf.br ^bb1
  }
}

// The following processes should stay unmodified, just make sure the pass doesn't crash on them

// CHECK-LABEL: hw.module @more_than_one_wait
hw.module @more_than_one_wait() {
  llhd.process {
    cf.br ^bb1
  ^bb1:
    llhd.wait ^bb2
  ^bb2:
    llhd.wait ^bb1
  }
}

// CHECK-LABEL: hw.module @more_than_two_TRs
hw.module @more_than_two_TRs() {
  llhd.process {
    cf.br ^bb1
  ^bb1:
    llhd.wait ^bb2
  ^bb2:
    llhd.wait ^bb3
  ^bb3:
    llhd.wait ^bb1
  }
}

// CHECK-LABEL: hw.module @more_than_one_TR_wait_terminator
hw.module @more_than_one_TR_wait_terminator(in %cond: i1) {
  llhd.process {
    cf.br ^bb1
  ^bb1:
    cf.cond_br %cond, ^bb2, ^bb3
  ^bb2:
    llhd.wait ^bb4
  ^bb3:
    llhd.wait ^bb4
  ^bb4:
    cf.br ^bb1
  }
}
