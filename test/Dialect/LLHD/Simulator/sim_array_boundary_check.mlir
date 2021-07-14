// REQUIRES: llhd-sim
// RUN: llhd-sim %s -shared-libs=%shlibdir/libcirct-llhd-signals-runtime-wrappers%shlibext | FileCheck %s

// CHECK: 0ps 0d 0e  root/sig[0]  0xffff
// CHECK-NEXT: 0ps 0d 0e  root/sig[1]  0xffff
// CHECK-NEXT: 0ps 0d 1e  root/sig[1]  0x0000
llhd.entity @root () -> () {
    %0 = llhd.const -1 : i8
    %1 = llhd.array_uniform %0 : !llhd.array<2 x i8>
    %3 = llhd.array_uniform %1 : !llhd.array<2 x !llhd.array<2 x i8>>
    %2 = llhd.sig "sig" %3 : !llhd.array<2 x !llhd.array<2 x i8>>
    %index = llhd.const 1 : i1
    %ext = llhd.dyn_extract_slice %3, %index : (!llhd.array<2 x !llhd.array<2 x i8>>, i1) -> !llhd.array<2 x !llhd.array<2 x i8>>
    %time = llhd.const #llhd.time<0ns, 0d, 1e> : !llhd.time
    llhd.drv %2, %ext after %time : !llhd.sig<!llhd.array<2 x !llhd.array<2 x i8>>>
}
