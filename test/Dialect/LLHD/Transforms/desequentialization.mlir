// RUN: circt-opt --llhd-desequentialize -canonicalize %s | FileCheck %s

// COM: it's ugly to run canonicalization here, but otherwise a lot of dead code
// COM: is present which we'd also need to file check (at least partially). Maybe
// COM: consider adding some basic DCE to the pass itself?

// CHECK-LABEL: @noResetNoEnable
// CHECK-SAME: (inout [[CLK:%.+]] : i1, inout [[SIG:%.+]] : i1)
hw.module @noResetNoEnable(inout %clk : i1, inout %sig : i1, out out : i1) {
  // CHECK: [[V0:%.+]] = llhd.prb [[CLK]]
  // CHECK: [[V1:%.+]] = seq.to_clock [[V0]]
  // CHECK: [[V2:%.+]] = seq.compreg %false{{.*}}, [[V1]]
  // CHECK: llhd.drv [[SIG]], [[V2]] after
  %time = llhd.constant_time <0ns, 1d, 0e>
  %false = hw.constant false
  %true = hw.constant true
  %0 = llhd.process -> i1 {
    %p = llhd.prb %clk : !hw.inout<i1>
    %old = comb.xor %0, %true : i1
    %posedge = comb.and %old, %p : i1
    llhd.drv %sig, %false after %time if %posedge : !hw.inout<i1>
    llhd.yield %p : i1
  }

  // CHECK: [[V3:%.+]] = seq.compreg %false{{.*}}, [[V1]]
  %1:2 = llhd.process -> i1, i1 {
    %p = llhd.prb %clk : !hw.inout<i1>
    %old = comb.xor %1#1, %true : i1
    %posedge = comb.and %old, %p : i1
    %2 = comb.mux %posedge, %false, %1#0 : i1
    llhd.yield %2, %p : i1, i1
  }

  // CHECK: hw.output [[V3]] :
  hw.output %1#0 : i1
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
  %0 = llhd.process -> i1 {
    %p = llhd.prb %clk : !hw.inout<i1>
    %old = comb.xor %p, %true : i1
    %posedge = comb.and %old, %0 : i1
    llhd.drv %sig, %false after %time if %posedge : !hw.inout<i1>
    llhd.yield %p : i1
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
  %0 = llhd.process -> i1 {
    %p = llhd.prb %clk : !hw.inout<i1>
    %old = comb.xor %0, %true : i1
    %posedge = comb.and %old, %p : i1
    %cond = comb.and %posedge, %c : i1
    llhd.drv %sig, %false after %time if %cond : !hw.inout<i1>
    llhd.yield %p : i1
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
  %0:2 = llhd.process -> i1, i1 {
    %p = llhd.prb %clk : !hw.inout<i1>
    %r = llhd.prb %rst : !hw.inout<i1>
    %old = comb.xor %0#0, %true : i1
    %posedge = comb.and %old, %p : i1
    %1 = comb.xor %r, %true : i1
    %negedge = comb.and %1, %0#1 : i1
    %cond = comb.or %negedge, %posedge : i1
    llhd.drv %sig, %false after %time if %cond : !hw.inout<i1>
    llhd.yield %p, %r : i1, i1
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
  %0 = llhd.process -> i1 {
    %clk_prb = llhd.prb %clk : !hw.inout<i1>
    %cond = comb.icmp bin ne %0, %clk_prb : i1
    llhd.drv %out, %false after %time if %cond : !hw.inout<i1>
    llhd.yield %clk_prb : i1
  }
}

// CHECK-LABEL: hw.module @simple
hw.module @simple() {
  // CHECK-NEXT: hw.output
  llhd.process {
    llhd.yield
  }
}

// CHECK-LABEL: hw.module @probes
hw.module @probes(inout %arg0 : i64) {
  // CHECK-NEXT: hw.output
  %0 = llhd.prb %arg0 : !hw.inout<i64>
  %1:2 = llhd.process -> i64, i64 {
    %2 = llhd.prb %arg0 : !hw.inout<i64>
    llhd.yield %0, %2 : i64, i64
  }
}

// CHECK-LABEL: @multipleBlocksNotAllowed
hw.module @multipleBlocksNotAllowed() {
  // CHECK-NEXT: llhd.process
  llhd.process {
    cf.br ^bb1
  ^bb1:
    cf.br ^bb2
  ^bb2:
    llhd.yield
  }
}
