// REQUIRES: llhd-sim
// RUN: llhd-sim %s -shared-libs=%shlibdir/libcirct-llhd-signals-runtime-wrappers%shlibext | FileCheck %s

// CHECK: 0ps 0d 0e  root/sig[0]  0xffff
// CHECK-NEXT: 0ps 0d 0e  root/sig[1]  0xffff
// CHECK-NEXT: 0ps 0d 1e  root/sig[1]  0x0000
llhd.entity @root () -> () {
    %0 = hw.constant -1 : i8
    %1 = hw.array_create %0, %0 : i8
    %3 = hw.array_create %1, %1 : !hw.array<2 x i8>
    %2 = llhd.sig "sig" %3 : !hw.array<2 x !hw.array<2 x i8>>
    %index = hw.constant 1 : i1
    %ext = llhd.dyn_extract_slice %3, %index : (!hw.array<2 x !hw.array<2 x i8>>, i1) -> !hw.array<2 x !hw.array<2 x i8>>
    %time = llhd.constant_time #llhd.time<0ns, 0d, 1e>
    llhd.drv %2, %ext after %time : !llhd.sig<!hw.array<2 x !hw.array<2 x i8>>>
}
