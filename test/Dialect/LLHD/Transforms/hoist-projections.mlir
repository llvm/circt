// RUN: circt-opt --llhd-hoist-projections %s | FileCheck %s

// CHECK-LABEL: @HoistOutOfProcess
hw.module @HoistOutOfProcess(in %idx: i3, in %v: i1) {
  %c0_i8 = hw.constant 0 : i8
  %a = llhd.sig %c0_i8 : <i8>
  // CHECK: [[C2:%.+]] = hw.constant 2 : i3
  // CHECK-NEXT: [[A0:%.+]] = llhd.sig.extract %a from [[C2]]
  // CHECK-NEXT: [[A1:%.+]] = llhd.sig.extract %a from %idx
  // CHECK-NEXT: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.sig.extract
    %c2_i3 = hw.constant 2 : i3
    %0 = llhd.sig.extract %a from %c2_i3 : <i8> -> <i1>
    %1 = llhd.sig.extract %a from %idx : <i8> -> <i1>
    %t = llhd.constant_time <0ns, 1d>
    // CHECK: llhd.drv [[A0]], %v
    // CHECK-NEXT: llhd.drv [[A1]], %v
    llhd.drv %0, %v after %t : i1
    llhd.drv %1, %v after %t : i1
    llhd.halt
  }
}

// CHECK-LABEL: @HoistOutOfCombinational
hw.module @HoistOutOfCombinational(in %v: i1) {
  %c0_i2 = hw.constant 0 : i2
  %array = hw.array_create %c0_i2, %c0_i2 : i2
  %a = llhd.sig %array : <!hw.array<2xi2>>
  // CHECK: [[C1:%.+]] = hw.constant true
  // CHECK-NEXT: [[A0:%.+]] = llhd.sig.array_get %a{{\[}}[[C1]]]
  // CHECK-NEXT: [[C0:%.+]] = hw.constant false
  // CHECK-NEXT: [[A1:%.+]] = llhd.sig.extract [[A0]] from [[C0]]
  // CHECK-NEXT: llhd.combinational
  llhd.combinational {
    // CHECK-NOT: llhd.sig
    %true = hw.constant true
    %0 = llhd.sig.array_get %a[%true] : <!hw.array<2xi2>>
    %false = hw.constant false
    %1 = llhd.sig.extract %0 from %false : <i2> -> <i1>
    %t = llhd.constant_time <0ns, 1d>
    // CHECK: llhd.drv [[A1]], %v
    llhd.drv %1, %v after %t : i1
    llhd.yield
  }
}
// CHECK-LABEL: @DontHoistLocalOperands
hw.module @DontHoistLocalOperands(in %x: i3) {
  %c0_i8 = hw.constant 0 : i8
  %a = llhd.sig %c0_i8 : <i8>
  // CHECK: llhd.process
  llhd.process {
    // CHECK: [[IDX:%.+]] = comb.add
    // CHECK-NEXT: llhd.sig.extract %a from [[IDX]]
    %idx = comb.add %x, %x : i3
    %0 = llhd.sig.extract %a from %idx : <i8> -> <i1>
    %1 = llhd.prb %0 : i1
    llhd.halt
  }
}

// CHECK-LABEL: @HoistOutOfFinal
hw.module @HoistOutOfFinal() {
  %s0 = hw.aggregate_constant [false, false] : !hw.struct<foo: i1, bar: i1>
  %s = llhd.sig %s0 : <!hw.struct<foo: i1, bar: i1>>
  // CHECK: [[S:%.+]] = llhd.sig.struct_extract %s["bar"]
  // CHECK-NEXT: llhd.final
  llhd.final {
    // CHECK-NOT: llhd.sig.struct_extract
    // CHECK: llhd.prb [[S]]
    %0 = llhd.sig.struct_extract %s["bar"] : <!hw.struct<foo: i1, bar: i1>>
    %1 = llhd.prb %0 : i1
    llhd.halt
  }
}
