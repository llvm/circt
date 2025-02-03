// RUN: circt-opt --llhd-desequentialize -canonicalize %s | FileCheck %s

// COM: it's ugly to run canonicalization here, but otherwise a lot of dead code
// COM: is present which we'd also need to file check (at least partially). Maybe
// COM: consider adding some basic DCE to the pass itself?

// CHECK-LABEL: @noResetNoEnable
// CHECK-SAME: (inout [[CLK:%.+]] : i1, inout [[SIG:%.+]] : i1)
hw.module @noResetNoEnable(inout %clk : i1, inout %sig : i1) {
  // CHECK: [[V0:%.+]] = llhd.prb [[CLK]]
  // CHECK: [[V1:%.+]] = seq.to_clock [[V0]]
  // CHECK: [[V2:%.+]] = seq.compreg %false{{.*}}, [[V1]]
  // CHECK: llhd.drv [[SIG]], [[V2]] after
  %time = llhd.constant_time <0ns, 1d, 0e>
  %false = hw.constant false
  %true = hw.constant true
  %p2 = llhd.prb %clk : !hw.inout<i1>
  llhd.process {
    cf.br ^bb1
  ^bb1:
    %p1 = llhd.prb %clk : !hw.inout<i1>
    llhd.wait (%p2 : i1), ^bb2
  ^bb2:
    %old = comb.xor %p1, %true : i1
    %posedge = comb.and %old, %p2 : i1
    llhd.drv %sig, %false after %time if %posedge : !hw.inout<i1>
    cf.br ^bb1
  }
}

// CHECK-LABEL: @negedgeClk
// CHECK-SAME: (inout [[CLK:%.+]] : i1, inout [[SIG:%.+]] : i1)
hw.module @negedgeClk(inout %clk : i1, inout %sig : i1) {
  // CHECK: [[V0:%.+]] = llhd.prb [[CLK]]
  // CHECK: [[V1:%.+]] = seq.to_clock [[V0]]
  // CHECK: [[V2:%.+]] = seq.clock_inv [[V1]]
  // CHECK: [[V3:%.+]] = seq.compreg %false{{.*}}, [[V2]]
  // CHECK: llhd.drv [[SIG]], [[V3]] after
  %time = llhd.constant_time <0ns, 1d, 0e>
  %false = hw.constant false
  %true = hw.constant true
  %p2 = llhd.prb %clk : !hw.inout<i1>
  llhd.process {
    cf.br ^bb1
  ^bb1:
    %p1 = llhd.prb %clk : !hw.inout<i1>
    llhd.wait (%p2 : i1), ^bb2
  ^bb2:
    %old = comb.xor %p2, %true : i1
    %posedge = comb.and %old, %p1 : i1
    llhd.drv %sig, %false after %time if %posedge : !hw.inout<i1>
    cf.br ^bb1
  }
}

// CHECK-LABEL: @enable
// CHECK-SAME: (in [[C:%.+]] : i1, inout [[CLK:%.+]] : i1, inout [[SIG:%.+]] : i1)
hw.module @enable(in %c : i1, inout %clk : i1, inout %sig : i1) {
  // CHECK: [[V0:%.+]] = llhd.prb [[CLK]]
  // CHECK: [[V1:%.+]] = seq.to_clock [[V0]]
  // CHECK: [[V2:%.+]] = seq.clock_gate [[V1]], [[C]]
  // CHECK: [[V3:%.+]] = seq.compreg %false{{.*}}, [[V2]]
  // CHECK: llhd.drv [[SIG]], [[V3]] after
  %time = llhd.constant_time <0ns, 1d, 0e>
  %false = hw.constant false
  %true = hw.constant true
  %p2 = llhd.prb %clk : !hw.inout<i1>
  llhd.process {
    cf.br ^bb1
  ^bb1:
    %p1 = llhd.prb %clk : !hw.inout<i1>
    llhd.wait (%p2 : i1), ^bb2
  ^bb2:
    %old = comb.xor %p1, %true : i1
    %posedge = comb.and %old, %p2 : i1
    %cond = comb.and %posedge, %c : i1
    llhd.drv %sig, %false after %time if %cond : !hw.inout<i1>
    cf.br ^bb1
  }
}

// CHECK-LABEL: @asyncReset
// CHECK-SAME: (inout [[RST:%.+]] : i1, inout [[CLK:%.+]] : i1, inout [[SIG:%.+]] : i1)
hw.module @asyncReset(inout %rst : i1, inout %clk : i1, inout %sig : i1) {
  // CHECK: [[V0:%.+]] = llhd.prb [[CLK]]
  // CHECK: [[V1:%.+]] = llhd.prb [[RST]]
  // CHECK: [[V2:%.+]] = seq.to_clock [[V0]]
  // CHECK: [[V3:%.+]] = comb.xor [[V1]], %true{{.*}}
  // CHECK: [[V4:%.+]] = seq.compreg %false{{.*}}, [[V2]] reset [[V3]], %false{{.*}}
  // CHECK: llhd.drv [[SIG]], [[V4]] after
  %time = llhd.constant_time <0ns, 1d, 0e>
  %false = hw.constant false
  %true = hw.constant true
  %p2 = llhd.prb %clk : !hw.inout<i1>
  %r2 = llhd.prb %rst : !hw.inout<i1>
  llhd.process {
    cf.br ^bb1
  ^bb1:
    %p1 = llhd.prb %clk : !hw.inout<i1>
    %r1 = llhd.prb %rst : !hw.inout<i1>
    llhd.wait (%p2, %r2 : i1, i1), ^bb2
  ^bb2:
    %old = comb.xor %p1, %true : i1
    %posedge = comb.and %old, %p2 : i1
    %0 = comb.xor %r2, %true : i1
    %negedge = comb.and %0, %r1 : i1
    %cond = comb.or %negedge, %posedge : i1
    llhd.drv %sig, %false after %time if %cond : !hw.inout<i1>
    cf.br ^bb1
  }
}

// CHECK-LABEL: @asyncResetNotObserved
// CHECK-SAME: (inout [[RST:%.+]] : i1, inout [[CLK:%.+]] : i1, inout [[SIG:%.+]] : i1)
hw.module @asyncResetNotObserved(inout %rst : i1, inout %clk : i1, inout %sig : i1) {
  // CHECK: llhd.process
  %time = llhd.constant_time <0ns, 1d, 0e>
  %false = hw.constant false
  %true = hw.constant true
  %p2 = llhd.prb %clk : !hw.inout<i1>
  %r2 = llhd.prb %rst : !hw.inout<i1>
  llhd.process {
    cf.br ^bb1
  ^bb1:
    %p1 = llhd.prb %clk : !hw.inout<i1>
    %r1 = llhd.prb %rst : !hw.inout<i1>
    llhd.wait (%p2 : i1), ^bb2
  ^bb2:
    %old = comb.xor %p1, %true : i1
    %posedge = comb.and %old, %p2 : i1
    %0 = comb.xor %r2, %true : i1
    %negedge = comb.and %0, %r1 : i1
    %cond = comb.or %negedge, %posedge : i1
    llhd.drv %sig, %false after %time if %cond : !hw.inout<i1>
    cf.br ^bb1
  }
}

// CHECK-LABEL: @compareClkPrb
// CHECK-SAME: (inout [[CLK:%.+]] : i1, inout [[OUT:%.+]] : i1)
hw.module @compareClkPrb(inout %clk : i1, inout %out : i1) {
  // CHECK: [[V0:%.+]] = llhd.prb [[CLK]]
  // CHECK: [[V1:%.+]] = seq.to_clock [[V0]]
  // CHECK: [[V2:%.+]] = comb.xor [[V0]], %true{{.*}}
  // CHECK: [[V3:%.+]] = seq.compreg %false{{.*}}, [[V1]] reset [[V2]], %false{{.*}}
  // CHECK: llhd.drv [[OUT]], [[V3]] after
  %false = hw.constant false
  %time = llhd.constant_time <0ns, 1d, 0e>
  %clk_0 = llhd.prb %clk : !hw.inout<i1>
  llhd.process {
    cf.br ^bb1
  ^bb1:
    %clk_1 = llhd.prb %clk : !hw.inout<i1>
    llhd.wait (%clk_0 : i1), ^bb2
  ^bb2:
    %clk_2 = llhd.prb %clk : !hw.inout<i1>
    %cond = comb.icmp ne %clk_1, %clk_2 : i1
    llhd.drv %out, %false after %time if %cond : !hw.inout<i1>
    cf.br ^bb1
  }
}
