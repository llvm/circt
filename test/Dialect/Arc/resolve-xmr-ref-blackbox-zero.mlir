// RUN: circt-opt --arc-resolve-xmr=lower-blackbox-internal-to-zero %s --verify-diagnostics | FileCheck %s

module {
  hw.hierpath @bbInternal [@Top::@bb, @BlackBox::@hidden]

  hw.module.extern @BlackBox(out out_clk : i1 {hw.exportPort = #hw<innerSym@out_clk>})

  // CHECK-LABEL: hw.module @Top
  // CHECK: %[[FALSE:.+]] = hw.constant false
  // CHECK: hw.output %[[FALSE]] : i1
  hw.module @Top(out o : i1) {
    // expected-warning @below {{internal to blackbox. Lowering to 0.}}
    hw.instance "bb" sym @bb @BlackBox() -> (out_clk: i1)
    %x = sv.xmr.ref @bbInternal : !hw.inout<i1>
    %r = sv.read_inout %x : !hw.inout<i1>
    hw.output %r : i1
  }
}
