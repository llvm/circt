// REQUIRES: llhd-sim
// RUN: llhd-sim %s -shared-libs=%shlibdir/libcirct-llhd-signals-runtime-wrappers%shlibext | FileCheck %s

// CHECK: 0ps 0d 0e  root/shl  0x01
// CHECK-NEXT: 0ps 0d 0e  root/shr  0x08
// CHECK-NEXT: 1000ps 0d 0e  root/shl  0x02
// CHECK-NEXT: 1000ps 0d 0e  root/shr  0x0c
// CHECK-NEXT: 2000ps 0d 0e  root/shl  0x04
// CHECK-NEXT: 2000ps 0d 0e  root/shr  0x0e
// CHECK-NEXT: 3000ps 0d 0e  root/shl  0x08
// CHECK-NEXT: 3000ps 0d 0e  root/shr  0x0f
// CHECK-NEXT: 4000ps 0d 0e  root/shl  0x00

llhd.entity @root () -> () {
  %time = llhd.constant_time #llhd.time<1ns, 0d, 0e>

  %init = hw.constant 8 : i4
  %hidden = hw.constant 1 : i2
  %amnt = hw.constant 1 : i1

  %sig = llhd.sig "shr" %init : i4
  %prbd = llhd.prb %sig : !llhd.sig<i4>
  %shr = llhd.shr %prbd, %hidden, %amnt : (i4, i2, i1) -> i4
  llhd.drv %sig, %shr after %time : !llhd.sig<i4>

  %init1 = hw.constant 1 : i4
  %hidden1 = hw.constant 0 : i1
  %amnt1 = hw.constant 1 : i1

  %sig1 = llhd.sig "shl" %init1 : i4
  %prbd1 = llhd.prb %sig1 : !llhd.sig<i4>
  %shl = llhd.shl %prbd1, %hidden1, %amnt1 : (i4, i1, i1) -> i4
  llhd.drv %sig1, %shl after %time : !llhd.sig<i4>
}
