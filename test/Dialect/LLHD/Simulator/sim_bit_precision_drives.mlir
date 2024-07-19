// REQUIRES: llhd-sim
// RUN: llhd-sim %s -shared-libs=%shlibdir/libcirct-llhd-signals-runtime-wrappers%shlibext | FileCheck %s

// CHECK: 0ps 0d 0e  root/sameByte  0xffffffff
// CHECK-NEXT: 0ps 0d 0e  root/spanBytes  0xffffffff
// CHECK-NEXT: 0ps 0d 0e  root/twoBytes  0x12345678
// CHECK-NEXT: 1000ps 0d 0e  root/sameByte  0xfffffffc
// CHECK-NEXT: 1000ps 0d 0e  root/spanBytes  0xfffff00f
// CHECK-NEXT: 1000ps 0d 0e  root/twoBytes  0x1234ffff
hw.module @root() {
  %0 = hw.constant 0x12345678 : i32
  %1 = hw.constant 0xffffffff : i32
  %s0 = llhd.sig "twoBytes" %0 : i32
  %s1 = llhd.sig "spanBytes" %1 : i32
  %s2 = llhd.sig "sameByte" %1 : i32
  %c0 = hw.constant 0xffff : i16
  %c1 = hw.constant 0 : i8
  %c2 = hw.constant 0 : i1
  %2 = hw.constant 0 : i5
  %3 = hw.constant 1 : i5
  %4 = hw.constant 4 : i5
  %t = llhd.constant_time #llhd.time<1ns, 0d, 0e>
  %e0 = llhd.sig.extract %s0 from %2 : (!hw.inout<i32>) -> !hw.inout<i16>
  %e1 = llhd.sig.extract %s1 from %4 : (!hw.inout<i32>) -> !hw.inout<i8>
  %e2 = llhd.sig.extract %s2 from %2 : (!hw.inout<i32>) -> !hw.inout<i1>
  %e3 = llhd.sig.extract %s2 from %3 : (!hw.inout<i32>) -> !hw.inout<i1>
  llhd.drv %e0, %c0 after %t : !hw.inout<i16>
  llhd.drv %e1, %c1 after %t : !hw.inout<i8>
  llhd.drv %e2, %c2 after %t : !hw.inout<i1>
  llhd.drv %e3, %c2 after %t : !hw.inout<i1>
}
