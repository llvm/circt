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
  // CHECK: %c = llhd.sig
  // CHECK: %d = llhd.sig
  // CHECK: %e = llhd.sig
  // CHECK: %f = llhd.sig
  // CHECK: %g = llhd.sig
  // CHECK: %h = llhd.sig
  // CHECK: %k = llhd.sig
  // CHECK: %l = llhd.sig
  // CHECK: %m = llhd.sig
  // CHECK: %n = llhd.sig
  // CHECK: %o = llhd.sig
  %c = llhd.sig %false : i1
  %d = llhd.sig %false : i1
  %e = llhd.sig %false : i1
  %f = llhd.sig %false : i1
  %g = llhd.sig %c0_i4 : i4
  %h = llhd.sig %c0_i4 : i4
  %k = llhd.sig %c0_i5 : i5
  %l = llhd.sig %c0_i5 : i5
  %m = llhd.sig %c0_i5 : i5
  %n = llhd.sig %c0_i5 : i5
  %o = llhd.sig %c0_i5 : i5

  %prb_k = llhd.prb %k : !hw.inout<i5>
  %prb_c = llhd.prb %c : !hw.inout<i1>
  %prb_e = llhd.prb %e : !hw.inout<i1>
  %prb_h = llhd.prb %h : !hw.inout<i4>
  %prb_d = llhd.prb %d : !hw.inout<i1>
  %prb_f = llhd.prb %f : !hw.inout<i1>
  %prb_g = llhd.prb %g : !hw.inout<i4>

  // COM: Check that an auxillary block is created and all drives are moved to
  // COM: the exit block with the correct enable condition
  // CHECK: llhd.process
  llhd.process {
  // CHECK: cf.br ^[[BB1:.+]]
    cf.br ^bb1
  // CHECK: ^[[BB1]]:
  ^bb1:
    // CHECK: llhd.wait ({{.*}}), ^[[BB2:.+]]
    llhd.wait (%prb_k, %prb_c, %prb_e, %prb_h, %prb_d, %prb_f, %prb_g : i5, i1, i1, i4, i1, i1, i4), ^bb2
  // CHECK: ^[[BB2]]:
  ^bb2:
    // CHECK: [[V14:%.+]] = llhd.prb %c
    // CHECK: [[V15:%.+]] = llhd.prb %e
    // CHECK: [[V16:%.+]] = llhd.prb %h
    // CHECK: [[V17:%.+]] = comb.concat %false{{.*}}, [[V16]] : i1, i4
    // CHECK: [[V18:%.+]] = llhd.prb %d
    // CHECK: [[V19:%.+]] = llhd.prb %f
    // CHECK: [[V20:%.+]] = llhd.prb %k
    // CHECK: [[V21:%.+]] = llhd.prb %g
    // CHECK: [[V22:%.+]] = comb.concat %false{{.*}}, [[V21]] : i1, i4
    // CHECK: [[V23:%.+]] = comb.sub [[V20]], [[V22]] : i5
    // CHECK: [[V24:%.+]] = llhd.prb %k
    // CHECK: [[V25:%.+]] = llhd.prb %g
    // CHECK: [[V26:%.+]] = comb.concat %false{{.*}}, [[V25]] : i1, i4
    // CHECK: [[V27:%.+]] = comb.add [[V24]], [[V26]] : i5
    // CHECK: [[V28:%.+]] = llhd.prb %k
    %25 = llhd.prb %k : !hw.inout<i5>
    llhd.drv %k, %25 after %1 : !hw.inout<i5>
    %26 = llhd.prb %c : !hw.inout<i1>
    %27 = llhd.prb %e : !hw.inout<i1>
    %28 = llhd.prb %h : !hw.inout<i4>
    %29 = comb.concat %false, %28 : i1, i4
    %30 = llhd.prb %d : !hw.inout<i1>
    %31 = llhd.prb %f : !hw.inout<i1>
    %32 = llhd.prb %k : !hw.inout<i5>
    %33 = llhd.prb %g : !hw.inout<i4>
    %34 = comb.concat %false, %33 : i1, i4
    %35 = comb.sub %32, %34 : i5
    %36 = llhd.prb %k : !hw.inout<i5>
    %37 = llhd.prb %g : !hw.inout<i4>
    %38 = comb.concat %false, %37 : i1, i4
    %39 = comb.add %36, %38 : i5
    cf.cond_br %26, ^bb3, ^bb4
  ^bb3:
    llhd.drv %l, %c0_i5 after %1 if %cond : !hw.inout<i5>
    cf.br ^bb1
  ^bb4:
    cf.cond_br %27, ^bb5, ^bb6
  ^bb5:
    llhd.drv %m, %29 after %1 : !hw.inout<i5>
    cf.br ^bb1
  ^bb6:
    cf.cond_br %30, ^bb7, ^bb1
  ^bb7:
    cf.cond_br %31, ^bb8, ^bb9
  ^bb8:
    llhd.drv %n, %35 after %1 : !hw.inout<i5>
    cf.br ^bb1
  ^bb9:
    llhd.drv %o, %39 after %1 : !hw.inout<i5>
    cf.br ^bb1
    // CHECK: llhd.drv %k, [[V28]] after [[V1]] if %true{{.*}} : !hw.inout<i5>

    // CHECK: [[V29:%.+]] = comb.and %true{{.*}}, [[V14]] : i1
    // CHECK: [[V30:%.+]] = comb.or %false{{.*}}, [[V29]] : i1
    // CHECK: [[V31:%.+]] = comb.and %cond, [[V30]] : i1
    // CHECK: llhd.drv %l, %c0_i5 after [[V1]] if [[V31]] : !hw.inout<i5>

    // CHECK: [[V33:%.+]] = comb.xor [[V14]], %true{{.*}} : i1
    // CHECK: [[V34:%.+]] = comb.and %true{{.*}}, [[V33]] : i1
    // CHECK: [[V35:%.+]] = comb.or %false{{.*}}, [[V34]] : i1
    // CHECK: [[V36:%.+]] = comb.and [[V35]], [[V15]] : i1
    // CHECK: [[V37:%.+]] = comb.or %false{{.*}}, [[V36]] : i1
    // CHECK: llhd.drv %m, [[V17]] after [[V1]] if [[V37]] : !hw.inout<i5>

    // CHECK: [[V40:%.+]] = comb.xor [[V15]], %true{{.*}} : i1
    // CHECK: [[V41:%.+]] = comb.and [[V35]], [[V40]] : i1
    // CHECK: [[V42:%.+]] = comb.or %false{{.*}}, [[V41]] : i1
    // CHECK: [[V43:%.+]] = comb.and [[V42]], [[V18]] : i1
    // CHECK: [[V44:%.+]] = comb.or %false{{.*}}, [[V43]] : i1
    // CHECK: [[V45:%.+]] = comb.and [[V44]], [[V19]] : i1
    // CHECK: [[V46:%.+]] = comb.or %false{{.*}}, [[V45]] : i1
    // CHECK: llhd.drv %n, [[V23]] after [[V1]] if [[V46]] : !hw.inout<i5>

    // CHECK: [[V49:%.+]] = comb.xor [[V19]], %true{{.*}} : i1
    // CHECK: [[V50:%.+]] = comb.and [[V44]], [[V49]] : i1
    // CHECK: [[V51:%.+]] = comb.or %false{{.*}}, [[V50]] : i1
    // CHECK: llhd.drv %o, [[V27]] after [[V1]] if [[V51]] : !hw.inout<i5>
    // CHECK: cf.br ^[[BB1]]
  }

  // COM: check drive coalescing behavior
  llhd.process {
    cf.br ^bb1
  ^bb1:
    llhd.wait ^bb2
  ^bb2:
    // CHECK:      [[V20:%.+]] = llhd.prb %l
    // CHECK-NEXT: [[V21:%.+]] = llhd.prb %m
    %20 = llhd.prb %l : !hw.inout<i5>
    %21 = llhd.prb %m : !hw.inout<i5>

    // CHECK-NEXT: llhd.drv %k, [[V21]] after [[V1]] : !hw.inout<i5>
    llhd.drv %k, %20 after %1 : !hw.inout<i5>
    llhd.drv %k, %21 after %1 : !hw.inout<i5>

    // CHECK-NEXT: [[V22:%.+]] = comb.mux %cond, [[V21]], [[V20]]
    // CHECK-NEXT: llhd.drv %l, [[V22]] after [[V1]] : !hw.inout<i5>
    llhd.drv %l, %20 after %1 : !hw.inout<i5>
    llhd.drv %l, %21 after %1 if %cond : !hw.inout<i5>

    // CHECK-NEXT: [[V23:%.+]] = comb.xor %cond, %true
    // CHECK-NEXT: [[V24:%.+]] = comb.mux %cond, [[V21]], [[V20]]
    // CHECK-NEXT: [[V25:%.+]] = comb.or %cond, [[V23]]
    // CHECK-NEXT: llhd.drv %m, [[V24]] after [[V1]] if [[V25]] : !hw.inout<i5>
    %22 = comb.xor %cond, %true : i1
    llhd.drv %m, %20 after %1 if %22 : !hw.inout<i5>
    llhd.drv %m, %21 after %1 if %cond : !hw.inout<i5>

    // CHECK-NEXT: cf.br
    cf.br ^bb1
  }
}

hw.module @value_motion(in %a_in : i1, in %clk_in : i1, out b : i1) {
  %true = hw.constant true
  %false = hw.constant false

  // CHECK: [[V0:%.+]] = llhd.constant_time <0ns, 0d, 1e>
  // CHECK: [[V1:%.+]] = llhd.constant_time <0ns, 1d, 0e>
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %1 = llhd.constant_time <0ns, 1d, 0e>

  // CHECK: %a = llhd.sig
  // CHECK: %clk = llhd.sig
  // CHECK: %b = llhd.sig
  %a = llhd.sig name "a" %false : i1
  %clk = llhd.sig name "clk" %false : i1
  %b = llhd.sig %false : i1

  // CHECK: [[V2:%.+]] = llhd.prb %clk
  %2 = llhd.prb %clk : !hw.inout<i1>

  // CHECK: llhd.process
  llhd.process {
  // CHECK: cf.br ^[[BB1:.+]]
    cf.br ^bb1
  // CHECK: ^[[BB1]]:
  ^bb1:
    // CHECK: [[V3:%.+]] = llhd.prb %clk
    // CHECK: llhd.wait ([[V2]] : i1), ^[[BB2:.+]]
    %4 = llhd.prb %clk : !hw.inout<i1>
    llhd.wait (%2 : i1), ^bb2
  // CHECK: ^[[BB2]]:
  ^bb2:
    // CHECK: [[V4:%.+]] = llhd.prb %clk
    // CHECK: [[V5:%.+]] = comb.xor bin [[V3]], %true{{.*}}
    // CHECK: [[V6:%.+]] = comb.and bin [[V5]], [[V4]]
    %5 = llhd.prb %clk : !hw.inout<i1>
    %6 = comb.xor bin %4, %true : i1
    %7 = comb.and bin %6, %5 : i1
    cf.cond_br %7, ^bb3, ^bb1
  ^bb3:
    %8 = llhd.prb %a : !hw.inout<i1>
    %9 = comb.xor %8, %true : i1
    llhd.drv %b, %9 after %1 : !hw.inout<i1>
    cf.br ^bb1
    // CHECK: [[V7:%.+]] = comb.and %true{{.*}}, [[V6]]
    // CHECK: [[V8:%.+]] = comb.or %false{{.*}}, [[V7]]
    // CHECK: [[V9:%.+]] = llhd.prb %a
    // CHECK: [[V10:%.+]] = comb.xor [[V9]], %true{{.*}}
    // CHECK: llhd.drv %b, [[V10]] after [[V1]] if [[V8]] : !hw.inout<i1>
    // CHECK-NEXT: cf.br ^[[BB1:.+]]
  }

  // Unused in pass, just check that it stay unmodified
  // CHECK: llhd.drv %a, %a_in after [[V0]]
  // CHECK: llhd.drv %clk, %clk_in after [[V0]]
  // CHECK: [[V11:%.+]] = llhd.prb %b
  // CHECK: hw.output [[V11]]
  llhd.drv %a, %a_in after %0 : !hw.inout<i1>
  llhd.drv %clk, %clk_in after %0 : !hw.inout<i1>
  %3 = llhd.prb %b : !hw.inout<i1>
  hw.output %3 : i1
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

// CHECK-LABEL: @unsupportedLoop
hw.module @unsupportedLoop() {
  // CHECK-NEXT: llhd.process {
  // CHECK-NEXT:   cf.br ^bb
  // CHECK-NEXT: ^bb
  // CHECK-NEXT:   llhd.wait ^bb
  // CHECK-NEXT: }
  llhd.process {
    cf.br ^bb1
  ^bb1:
    llhd.wait ^bb1
  }
  hw.output
}
