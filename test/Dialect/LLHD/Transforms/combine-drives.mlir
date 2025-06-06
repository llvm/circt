// RUN: circt-opt --llhd-combine-drives %s | FileCheck %s

// Basic aggregation of drives through multiple layers of aggregates.
// CHECK-LABEL: @Basic
hw.module @Basic(in %u0: i42, in %u1: i42, in %u2: i42, in %u3: i20, in %u4: i22) {
  %c0_i2 = hw.constant 0 : i2
  %c1_i2 = hw.constant 1 : i2
  %false = hw.constant false
  %true = hw.constant true
  %c0_i6 = hw.constant 0 : i6
  %c20_i6 = hw.constant 20 : i6
  %0 = builtin.unrealized_conversion_cast to !hw.struct<a: !hw.array<3xi42>, b: i42>
  %1 = llhd.constant_time <0ns, 0d, 1e>
  // CHECK: %s = llhd.sig
  // CHECK-NEXT: [[STRUCT:%.+]] = hw.struct_create ([[A:%.+]], [[B:%.+]])
  // CHECK-NEXT: llhd.drv %s, [[STRUCT]] after {{%.+}}
  %s = llhd.sig %0 : !hw.struct<a: !hw.array<3xi42>, b: i42>
  // CHECK-NEXT: [[A]] = hw.array_create %u2, %u1, %u0
  %a = llhd.sig.struct_extract %s["a"] : !hw.inout<struct<a: !hw.array<3xi42>, b: i42>>
  %a12 = llhd.sig.array_slice %a at %c1_i2 : (!hw.inout<array<3xi42>>) -> !hw.inout<array<2xi42>>
  %a0 = llhd.sig.array_get %a[%c0_i2] : !hw.inout<array<3xi42>>
  %a1 = llhd.sig.array_get %a12[%false] : !hw.inout<array<2xi42>>
  %a2 = llhd.sig.array_get %a12[%true] : !hw.inout<array<2xi42>>
  llhd.drv %a0, %u0 after %1 : !hw.inout<i42>  // s.a[0] = u0
  llhd.drv %a1, %u1 after %1 : !hw.inout<i42>  // s.a[2:1][0] = u1
  llhd.drv %a2, %u2 after %1 : !hw.inout<i42>  // s.a[2:1][1] = u2
  // CHECK-NEXT: [[B]] = comb.concat %u4, %u3
  %bX = llhd.sig.struct_extract %s["b"] : !hw.inout<struct<a: !hw.array<3xi42>, b: i42>>
  %bY = llhd.sig.struct_extract %s["b"] : !hw.inout<struct<a: !hw.array<3xi42>, b: i42>>
  %b0 = llhd.sig.extract %bX from %c0_i6 : (!hw.inout<i42>) -> !hw.inout<i20>
  %b1 = llhd.sig.extract %bY from %c20_i6 : (!hw.inout<i42>) -> !hw.inout<i22>
  llhd.drv %b0, %u3 after %1 : !hw.inout<i20>  // s.b[19:0] = u3
  llhd.drv %b1, %u4 after %1 : !hw.inout<i22>  // s.b[41:20] = u4
}

// CHECK-LABEL: @DriveParameters
hw.module @DriveParameters(in %u: i21, in %v: i21, in %e: i1) {
  %c0_i42 = hw.constant 0 : i42
  %c0_i6 = hw.constant 0 : i6
  %c21_i6 = hw.constant 21 : i6
  // CHECK:      [[T0:%.+]] = llhd.constant_time <0ns, 0d, 1e>
  // CHECK-NEXT: [[T1:%.+]] = llhd.constant_time <0ns, 1d, 0e>
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %1 = llhd.constant_time <0ns, 1d, 0e>
  // CHECK-NEXT: %a = llhd.sig
  %a = llhd.sig %c0_i42 : i42
  // CHECK-NEXT: [[TMP:%.+]] = comb.concat %v, %u
  // CHECK-NEXT: llhd.drv %a, [[TMP]] after [[T0]] :
  // CHECK-NEXT: [[TMP:%.+]] = comb.concat %u, %v
  // CHECK-NEXT: llhd.drv %a, [[TMP]] after [[T0]] if %e :
  // CHECK-NEXT: [[TMP:%.+]] = comb.concat %u, %u
  // CHECK-NEXT: llhd.drv %a, [[TMP]] after [[T1]] :
  // CHECK-NEXT: [[TMP:%.+]] = comb.concat %v, %v
  // CHECK-NEXT: llhd.drv %a, [[TMP]] after [[T1]] if %e :
  %a0 = llhd.sig.extract %a from %c0_i6 : (!hw.inout<i42>) -> !hw.inout<i21>
  %a1 = llhd.sig.extract %a from %c21_i6 : (!hw.inout<i42>) -> !hw.inout<i21>
  llhd.drv %a0, %u after %0 : !hw.inout<i21>
  llhd.drv %a1, %v after %0 : !hw.inout<i21>
  llhd.drv %a0, %v after %0 if %e : !hw.inout<i21>
  llhd.drv %a1, %u after %0 if %e : !hw.inout<i21>
  llhd.drv %a0, %u after %1 : !hw.inout<i21>
  llhd.drv %a1, %u after %1 : !hw.inout<i21>
  llhd.drv %a0, %v after %1 if %e : !hw.inout<i21>
  llhd.drv %a1, %v after %1 if %e : !hw.inout<i21>
}

// CHECK-LABEL: @DrivesToBlockArgSignals
hw.module @DrivesToBlockArgSignals(inout %a: i42, in %u: i20, in %v: i22) {
  // CHECK-DAG: [[TMP:%.+]] = comb.concat %v, %u
  // CHECK-DAG: [[T:%.+]] = llhd.constant_time
  // CHECK-DAG: llhd.drv %a, [[TMP]] after [[T]]
  %c0_i6 = hw.constant 0 : i6
  %c20_i6 = hw.constant 20 : i6
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %1 = llhd.sig.extract %a from %c0_i6 : (!hw.inout<i42>) -> !hw.inout<i20>
  %2 = llhd.sig.extract %a from %c20_i6 : (!hw.inout<i42>) -> !hw.inout<i22>
  llhd.drv %1, %u after %0 : !hw.inout<i20>
  llhd.drv %2, %v after %0 : !hw.inout<i22>
}

// CHECK-LABEL: @DrivesToOpaqueSignals
hw.module @DrivesToOpaqueSignals(in %u: i20, in %v: i22) {
  // CHECK-DAG: [[A:%.+]] = builtin.unrealized_conversion_cast
  // CHECK-DAG: [[TMP:%.+]] = comb.concat %v, %u
  // CHECK-DAG: [[T:%.+]] = llhd.constant_time
  // CHECK-DAG: llhd.drv [[A]], [[TMP]] after [[T]]
  %c0_i6 = hw.constant 0 : i6
  %c20_i6 = hw.constant 20 : i6
  %a = builtin.unrealized_conversion_cast to !hw.inout<i42>
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %1 = llhd.sig.extract %a from %c0_i6 : (!hw.inout<i42>) -> !hw.inout<i20>
  %2 = llhd.sig.extract %a from %c20_i6 : (!hw.inout<i42>) -> !hw.inout<i22>
  llhd.drv %1, %u after %0 : !hw.inout<i20>
  llhd.drv %2, %v after %0 : !hw.inout<i22>
}

// CHECK-LABEL: @SkipDynamicExtract
hw.module @SkipDynamicExtract(in %u: i6, in %v: i8) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %1 = builtin.unrealized_conversion_cast to i42
  %a = llhd.sig %1 : i42
  // CHECK: [[TMP:%.+]] = llhd.sig.extract %a from %u
  // CHECK: llhd.drv [[TMP]], %v
  %2 = llhd.sig.extract %a from %u : (!hw.inout<i42>) -> !hw.inout<i8>
  llhd.drv %2, %v after %0 : !hw.inout<i8>
}

// CHECK-LABEL: @SkipDynamicArraySlice
hw.module @SkipDynamicArraySlice(in %u: i6, in %v: !hw.array<8xi42>) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %1 = builtin.unrealized_conversion_cast to !hw.array<42xi42>
  %a = llhd.sig %1 : !hw.array<42xi42>
  // CHECK: [[TMP:%.+]] = llhd.sig.array_slice %a at %u
  // CHECK: llhd.drv [[TMP]], %v
  %2 = llhd.sig.array_slice %a at %u : (!hw.inout<array<42xi42>>) -> !hw.inout<array<8xi42>>
  llhd.drv %2, %v after %0 : !hw.inout<array<8xi42>>
}

// CHECK-LABEL: @SkipDynamicArrayGet
hw.module @SkipDynamicArrayGet(in %u: i6, in %v: i42) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %1 = builtin.unrealized_conversion_cast to !hw.array<42xi42>
  %a = llhd.sig %1 : !hw.array<42xi42>
  // CHECK: [[TMP:%.+]] = llhd.sig.array_get %a[%u]
  // CHECK: llhd.drv [[TMP]], %v
  %2 = llhd.sig.array_get %a[%u] : !hw.inout<array<42xi42>>
  llhd.drv %2, %v after %0 : !hw.inout<i42>
}

// CHECK-LABEL: @SkipIfGapsPresent
hw.module @SkipIfGapsPresent(in %u: i11, in %v: i30) {
  %c0_i6 = hw.constant 0 : i6
  %c12_i6 = hw.constant 12 : i6
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %1 = builtin.unrealized_conversion_cast to i42
  %a = llhd.sig %1 : i42
  // Bit 11 is not covered.
  // CHECK: [[TMP1:%.+]] = llhd.sig.extract %a from %c0_i6
  // CHECK: [[TMP2:%.+]] = llhd.sig.extract %a from %c12_i6
  // CHECK: llhd.drv [[TMP1]], %u
  // CHECK: llhd.drv [[TMP2]], %v
  %2 = llhd.sig.extract %a from %c0_i6 : (!hw.inout<i42>) -> !hw.inout<i11>
  %3 = llhd.sig.extract %a from %c12_i6 : (!hw.inout<i42>) -> !hw.inout<i30>
  llhd.drv %2, %u after %0 : !hw.inout<i11>
  llhd.drv %3, %v after %0 : !hw.inout<i30>
}

// CHECK-LABEL: @SkipIfOverlapsPresent
hw.module @SkipIfOverlapsPresent(in %u: i13, in %v: i30) {
  %c0_i6 = hw.constant 0 : i6
  %c12_i6 = hw.constant 12 : i6
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %1 = builtin.unrealized_conversion_cast to i42
  %a = llhd.sig %1 : i42
  // Bit 12 is covered twice.
  // CHECK: [[TMP1:%.+]] = llhd.sig.extract %a from %c0_i6
  // CHECK: [[TMP2:%.+]] = llhd.sig.extract %a from %c12_i6
  // CHECK: llhd.drv [[TMP1]], %u
  // CHECK: llhd.drv [[TMP2]], %v
  %2 = llhd.sig.extract %a from %c0_i6 : (!hw.inout<i42>) -> !hw.inout<i13>
  %3 = llhd.sig.extract %a from %c12_i6 : (!hw.inout<i42>) -> !hw.inout<i30>
  llhd.drv %2, %u after %0 : !hw.inout<i13>
  llhd.drv %3, %v after %0 : !hw.inout<i30>
}

// CHECK-LABEL: @RegressionOverlappingDrives
hw.module @RegressionOverlappingDrives(in %u: i153, in %v: i5) {
  %c42_i8 = hw.constant 42 : i8
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %a = llhd.sig %u : i153
  // CHECK: llhd.drv %a, %u
  llhd.drv %a, %u after %0 : !hw.inout<i153>
  // CHECK: [[TMP:%.+]] = llhd.sig.extract
  // CHECK: llhd.drv [[TMP]], %v
  %1 = llhd.sig.extract %a from %c42_i8 : (!hw.inout<i153>) -> !hw.inout<i5>
  llhd.drv %1, %v after %0 : !hw.inout<i5>
}

// CHECK-LABEL: @IgnoreNestedSignalUsers
hw.module @IgnoreNestedSignalUsers(in %u: i20, in %v: i22) {
  %c0_i6 = hw.constant 0 : i6
  %c20_i6 = hw.constant 20 : i6
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %1 = builtin.unrealized_conversion_cast to i42

  %a = llhd.sig %1 : i42
  // CHECK: [[TMP1:%.+]] = llhd.sig.extract %a from %c0_i6
  // CHECK: [[TMP2:%.+]] = llhd.sig.extract %a from %c20_i6
  %2 = llhd.sig.extract %a from %c0_i6 : (!hw.inout<i42>) -> !hw.inout<i20>
  %3 = llhd.sig.extract %a from %c20_i6 : (!hw.inout<i42>) -> !hw.inout<i22>
  // CHECK: llhd.drv [[TMP1]], %u
  llhd.drv %2, %u after %0 : !hw.inout<i20>
  llhd.process {
    // CHECK: llhd.drv [[TMP2]], %v
    llhd.drv %3, %v after %0 : !hw.inout<i22>
    llhd.halt
  }

  %b = llhd.sig %1 : i42
  // CHECK: [[TMP1:%.+]] = llhd.sig.extract %b from %c0_i6
  %4 = llhd.sig.extract %b from %c0_i6 : (!hw.inout<i42>) -> !hw.inout<i20>
  // CHECK: llhd.drv [[TMP1]], %u
  llhd.drv %4, %u after %0 : !hw.inout<i20>
  llhd.process {
    // CHECK: [[TMP2:%.+]] = llhd.sig.extract %b from %c20_i6
    %5 = llhd.sig.extract %b from %c20_i6 : (!hw.inout<i42>) -> !hw.inout<i22>
    // CHECK: llhd.drv [[TMP2]], %v
    llhd.drv %5, %v after %0 : !hw.inout<i22>
    llhd.halt
  }
}

// Driving a single array element still requires an aggregate to be created.
// CHECK-LABEL: @RegressionSingleElementArray
hw.module @RegressionSingleElementArray(in %u: i1) {
  %c0_i0 = hw.constant 0 : i0
  %0 = builtin.unrealized_conversion_cast to !hw.array<1xi1>
  // CHECK: [[T:%.+]] = llhd.constant_time <0ns, 0d, 1e>
  %1 = llhd.constant_time <0ns, 0d, 1e>
  // CHECK-NEXT: %a = llhd.sig
  // CHECK-NEXT: [[TMP:%.+]] = hw.array_create %u
  // CHECK-NEXT: llhd.drv %a, [[TMP]] after [[T]] :
  %a = llhd.sig %0 : !hw.array<1xi1>
  %2 = llhd.sig.array_get %a[%c0_i0] : !hw.inout<array<1xi1>>
  llhd.drv %2, %u after %1 : !hw.inout<i1>
}

// Driving a single struct field still requires an aggregate to be created.
// CHECK-LABEL: @RegressionSingleFieldStruct
hw.module @RegressionSingleFieldStruct(in %u: i1) {
  %c0_i0 = hw.constant 0 : i0
  %0 = builtin.unrealized_conversion_cast to !hw.struct<x: i1>
  // CHECK: [[T:%.+]] = llhd.constant_time <0ns, 0d, 1e>
  %1 = llhd.constant_time <0ns, 0d, 1e>
  // CHECK-NEXT: %a = llhd.sig
  // CHECK-NEXT: [[TMP:%.+]] = hw.struct_create (%u)
  // CHECK-NEXT: llhd.drv %a, [[TMP]] after [[T]] :
  %a = llhd.sig %0 : !hw.struct<x: i1>
  %2 = llhd.sig.struct_extract %a["x"] : !hw.inout<struct<x: i1>>
  llhd.drv %2, %u after %1 : !hw.inout<i1>
}
