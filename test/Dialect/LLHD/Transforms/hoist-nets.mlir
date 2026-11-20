// RUN: circt-opt --llhd-hoist-nets %s | FileCheck %s

// CHECK-LABEL: @HoistInProcess
hw.module @HoistInProcess() {
  // CHECK: [[NET:%.+]] = llhd.net wire
  %net = llhd.net wire : <i8>
  // CHECK: [[P:%.+]] = llhd.prb_net [[NET]]
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.prb_net
    %0 = llhd.prb_net %net : i8
    // CHECK: comb.add [[P]], [[P]]
    %1 = comb.add %0, %0 : i8
    llhd.halt
  }
}

// CHECK-LABEL: @HoistAcrossWait
hw.module @HoistAcrossWait(in %a : i1) {
  // CHECK: [[NET:%.+]] = llhd.net wire
  %net = llhd.net wire : <i1>
  // CHECK: [[P:%.+]] = llhd.prb_net [[NET]]
  // CHECK: llhd.process
  llhd.process {
    %0 = llhd.prb_net %net : i1
    // CHECK: llhd.wait ^[[BB:[a-zA-Z0-9]+]]([[P]] : i1)
    llhd.wait ^bb1
  ^bb1:
    // CHECK: ^[[BB]]([[ARG:%.+]]: i1):
    // CHECK: comb.xor [[ARG]]
    %1 = comb.xor %0, %0 : i1
    llhd.halt
  }
}

// CHECK-LABEL: @HoistInCombinational
hw.module @HoistInCombinational(out out : i8) {
  // CHECK: [[NET:%.+]] = llhd.net wire
  %net = llhd.net wire : <i8>
  // CHECK: [[P:%.+]] = llhd.prb_net [[NET]]
  // CHECK: llhd.combinational
  %r = llhd.combinational -> i8 {
    // CHECK-NOT: llhd.prb_net
    %0 = llhd.prb_net %net : i8
    // CHECK: llhd.yield [[P]] : i8
    llhd.yield %0 : i8
  }
  hw.output %r : i8
}

// CHECK-LABEL: @HoistFromNestedIf
hw.module @HoistFromNestedIf(in %c : i1) {
  // CHECK: [[NET:%.+]] = llhd.net wire
  %net = llhd.net wire : <i8>
  // CHECK: [[P:%.+]] = llhd.prb_net [[NET]]
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.prb_net
    scf.if %c {
      %0 = llhd.prb_net %net : i8
      // CHECK: comb.add [[P]], [[P]]
      %1 = comb.add %0, %0 : i8
    }
    llhd.halt
  }
}

// CHECK-LABEL: @NetDefinedInProcessNotHoisted
hw.module @NetDefinedInProcessNotHoisted() {
  // CHECK: llhd.process
  llhd.process {
    // CHECK: [[NET:%.+]] = llhd.net wire
    // CHECK: llhd.prb_net [[NET]]
    %net = llhd.net wire : <i8>
    %0 = llhd.prb_net %net : i8
    llhd.halt
  }
}
