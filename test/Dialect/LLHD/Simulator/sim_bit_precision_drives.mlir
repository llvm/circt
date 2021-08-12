// REQUIRES: llhd-sim
// RUN: llhd-sim %s -shared-libs=%shlibdir/libcirct-llhd-signals-runtime-wrappers%shlibext | FileCheck %s

// CHECK: 0ps 0d 0e  root/sameByte  0xffffffff
// CHECK-NEXT: 0ps 0d 0e  root/spanBytes  0xffffffff
// CHECK-NEXT: 0ps 0d 0e  root/twoBytes  0x12345678
// CHECK-NEXT: 1000ps 0d 0e  root/sameByte  0xfffffffc
// CHECK-NEXT: 1000ps 0d 0e  root/spanBytes  0xfffff00f
// CHECK-NEXT: 1000ps 0d 0e  root/twoBytes  0x1234ffff
llhd.entity @root () -> () {
  %0 = llhd.const 0x12345678 : i32
  %1 = llhd.const 0xffffffff : i32
  %s0 = llhd.sig "twoBytes" %0 : i32
  %s1 = llhd.sig "spanBytes" %1 : i32
  %s2 = llhd.sig "sameByte" %1 : i32
  %c0 = llhd.const 0xffff : i16
  %c1 = llhd.const 0 : i8
  %c2 = llhd.const 0 : i1
  %t = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
  %e0 = llhd.extract_slice %s0, 0 : !llhd.sig<i32> -> !llhd.sig<i16>
  %e1 = llhd.extract_slice %s1, 4 : !llhd.sig<i32> -> !llhd.sig<i8>
  %e2 = llhd.extract_slice %s2, 0 : !llhd.sig<i32> -> !llhd.sig<i1>
  %e3 = llhd.extract_slice %s2, 1 : !llhd.sig<i32> -> !llhd.sig<i1>
  llhd.drv %e0, %c0 after %t : !llhd.sig<i16>
  llhd.drv %e1, %c1 after %t : !llhd.sig<i8>
  llhd.drv %e2, %c2 after %t : !llhd.sig<i1>
  llhd.drv %e3, %c2 after %t : !llhd.sig<i1>
}
