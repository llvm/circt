// RUN: circt-opt %s --arc-lower-state | FileCheck %s

// A module-level drive of an array element
// (`llhd.drv (llhd.sig.array_get %sig[%idx])`) lowers as a read-modify-write
// of the parent signal's storage through `hw.array_inject` -- the array-typed
// sibling of the bit-slice splice.

// CHECK-LABEL: arc.model @top
hw.module @top(in %v : i4, in %idx : i1) {
  %c0_i4 = hw.constant 0 : i4
  %true = hw.constant true
  %t = llhd.constant_time <0ns, 0d, 1e>
  %init = hw.array_create %c0_i4, %c0_i4 : i4
  %sig = llhd.sig %init : !hw.array<2xi4>
  %elem = llhd.sig.array_get %sig[%true] : <!hw.array<2xi4>>
  %elemDyn = llhd.sig.array_get %sig[%idx] : <!hw.array<2xi4>>
  // CHECK-DAG: [[CUR:%.+]] = arc.state_read %{{.+}} : <!hw.array<2xi4>>
  // CHECK-DAG: [[NEW:%.+]] = hw.array_inject [[CUR]][%{{.+}}], %{{.+}} : !hw.array<2xi4>
  // CHECK: arc.state_write %{{.+}} = [[NEW]]
  llhd.drv %elem, %v after %t : i4
  // The dynamic-index element drive takes the same path.
  // CHECK: [[CUR2:%.+]] = arc.state_read %{{.+}} : <!hw.array<2xi4>>
  // CHECK: [[NEW2:%.+]] = hw.array_inject [[CUR2]][%{{.+}}], %{{.+}} : !hw.array<2xi4>
  // CHECK: arc.state_write %{{.+}} = [[NEW2]]
  llhd.drv %elemDyn, %v after %t : i4
}

// A drive through a CHAIN of element refs (multi-dimensional array,
// `sig.array_get (sig.array_get %sig[%i])[%j]`) peels the storage value
// outward-in with hw.array_get and re-injects inward-out.

// CHECK-LABEL: arc.model @nested
hw.module @nested(in %v : i4, in %i : i1, in %j : i1) {
  %c0_i4 = hw.constant 0 : i4
  %t = llhd.constant_time <0ns, 0d, 1e>
  %row = hw.array_create %c0_i4, %c0_i4 : i4
  %init = hw.array_create %row, %row : !hw.array<2xi4>
  %sig = llhd.sig %init : !hw.array<2xarray<2xi4>>
  %rowRef = llhd.sig.array_get %sig[%i] : <!hw.array<2xarray<2xi4>>>
  %elemRef = llhd.sig.array_get %rowRef[%j] : <!hw.array<2xi4>>
  // CHECK: [[CUR:%.+]] = arc.state_read %{{.+}} : <!hw.array<2xarray<2xi4>>>
  // CHECK: [[ROW:%.+]] = hw.array_get [[CUR]][%{{.+}}] : !hw.array<2xarray<2xi4>>
  // CHECK: [[NEWROW:%.+]] = hw.array_inject [[ROW]][%{{.+}}], %{{.+}} : !hw.array<2xi4>
  // CHECK: [[NEWARR:%.+]] = hw.array_inject [[CUR]][%{{.+}}], [[NEWROW]] : !hw.array<2xarray<2xi4>>
  // CHECK: arc.state_write %{{.+}} = [[NEWARR]]
  llhd.drv %elemRef, %v after %t : i4
}

// A struct-field drive (`llhd.drv (llhd.sig.struct_extract %sig["f"])`)
// takes the same RMW path through hw.struct_inject.

// CHECK-LABEL: arc.model @structField
hw.module @structField(in %v : i1) {
  %false = hw.constant false
  %c0_i8 = hw.constant 0 : i8
  %t = llhd.constant_time <0ns, 0d, 1e>
  %init = hw.struct_create (%false, %c0_i8) : !hw.struct<valid: i1, data: i8>
  %sig = llhd.sig %init : !hw.struct<valid: i1, data: i8>
  %fieldRef = llhd.sig.struct_extract %sig["valid"] : <!hw.struct<valid: i1, data: i8>>
  // CHECK: [[CUR:%.+]] = arc.state_read %{{.+}} : <!hw.struct<valid: i1, data: i8>>
  // CHECK: [[NEW:%.+]] = hw.struct_inject [[CUR]]["valid"], %{{.+}} : !hw.struct<valid: i1, data: i8>
  // CHECK: arc.state_write %{{.+}} = [[NEW]]
  llhd.drv %fieldRef, %v after %t : i1
}
