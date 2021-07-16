// REQUIRES: llhd-sim
// RUN: llhd-sim %s -shared-libs=%shlibdir/libcirct-llhd-signals-runtime-wrappers%shlibext | FileCheck %s

// CHECK: 0ps 0d 0e  root/proc/toggle  0x01
// CHECK-NEXT: 0ps 0d 0e  root/toggle  0x01
// CHECK-NEXT: 1000ps 0d 1e  root/proc/toggle  0x00
// CHECK-NEXT: 1000ps 0d 1e  root/toggle  0x00
llhd.entity @root () -> () {
  %0 = llhd.const 1 : i1
  %1 = llhd.sig "toggle" %0 : i1
  llhd.inst "proc" @p () -> (%1) : () -> (!llhd.sig<i1>)
}

llhd.proc @p () -> (%a : !llhd.sig<i1>) {
  br ^wait
^wait:
  %1 = llhd.prb %a : !llhd.sig<i1>
  %0 = llhd.not %1 : i1
  %wt = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
  llhd.wait for %wt, ^drive
^drive:
  %dt = llhd.const #llhd.time<0ns, 0d, 1e> : !llhd.time
  llhd.drv %a, %0 after %dt : !llhd.sig<i1>
  llhd.halt
}
