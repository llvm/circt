// REQUIRES: llhd-sim
// REQUIRES: llhd-sim-fixed
// RUN: llhd-sim %s -shared-libs=%shlibdir/libcirct-llhd-signals-runtime-wrappers%shlibext | FileCheck %s

// CHECK: 0ps 0d 0e  root/proc/toggle  0x01
// CHECK-NEXT: 0ps 0d 0e  root/toggle  0x01
// CHECK-NEXT: 1000ps 0d 1e  root/proc/toggle  0x00
// CHECK-NEXT: 1000ps 0d 1e  root/toggle  0x00
hw.module @root() {
  %0 = hw.constant 1 : i1
  %a = llhd.sig "toggle" %0 : i1
  llhd.process {
    cf.br ^wait
  ^wait:
    %1 = llhd.prb %a : !hw.inout<i1>
    %allset = hw.constant 1 : i1
    %0 = comb.xor %1, %allset : i1
    %wt = llhd.constant_time #llhd.time<1ns, 0d, 0e>
    llhd.wait for %wt, ^drive
  ^drive:
    %dt = llhd.constant_time #llhd.time<0ns, 0d, 1e>
    llhd.drv %a, %0 after %dt : !hw.inout<i1>
    llhd.halt
  }
}
