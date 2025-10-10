// RUN: circt-opt --sroa %s | FileCheck %s

// CHECK-LABEL: @struct
hw.module @struct(in %init : !hw.struct<a: i32, b: i32>, in %v1 : i32, in %v2 : i32, out out1 : !hw.struct<a: i32, b: i32>, out out2 : !hw.struct<a: i32, b: i32>) {
  %time = llhd.constant_time <0ns, 0d, 1e>

  //      CHECK: [[V0:%.+]] = hw.struct_extract %init["a"] : !hw.struct<a: i32, b: i32>
  // CHECK-NEXT: [[V1:%.+]] = llhd.sig{{ }}
  // CHECK-SAME: [[V0]]
  // CHECK-NEXT: [[V2:%.+]] = hw.struct_extract %init["b"] : !hw.struct<a: i32, b: i32>
  // CHECK-NEXT: [[V3:%.+]] = llhd.sig{{ }}
  // CHECK-SAME: [[V2]]
  %sig = llhd.sig %init : !hw.struct<a: i32, b: i32>

  // CHECK-NEXT: [[V4:%.+]] = llhd.prb [[V1]]
  // CHECK-NEXT: [[V5:%.+]] = llhd.prb [[V3]]
  // CHECK-NEXT: [[V6:%.+]] = hw.struct_create ([[V4]], [[V5]]) : !hw.struct<a: i32, b: i32>
  %prb = llhd.prb %sig : !hw.struct<a: i32, b: i32>

  %a = llhd.sig.struct_extract %sig["a"] : !llhd.ref<!hw.struct<a: i32, b: i32>>
  %b = llhd.sig.struct_extract %sig["b"] : !llhd.ref<!hw.struct<a: i32, b: i32>>

  // CHECK-NEXT: llhd.drv [[V1]], %v1 after
  // CHECK-NEXT: llhd.drv [[V3]], %v2 after
  llhd.drv %a, %v1 after %time : i32
  llhd.drv %b, %v2 after %time : i32

  // CHECK-NEXT: [[V7:%.+]] = hw.struct_extract %init["a"] : !hw.struct<a: i32, b: i32>
  // CHECK-NEXT: [[V8:%.+]] = llhd.sig{{ }}
  // CHECK-SAME: [[V7]]
  // CHECK-NEXT: [[V9:%.+]] = hw.struct_extract %init["b"] : !hw.struct<a: i32, b: i32>
  // CHECK-NEXT: [[V10:%.+]] = llhd.sig{{ }}
  // CHECK-SAME: [[V9]]
  %sig2 = llhd.sig %init : !hw.struct<a: i32, b: i32>

  // CHECK-NEXT: [[V11:%.+]] = llhd.prb [[V8]]
  // CHECK-NEXT: [[V12:%.+]] = llhd.prb [[V10]]
  // CHECK-NEXT: [[V13:%.+]] = hw.struct_create ([[V11]], [[V12]]) : !hw.struct<a: i32, b: i32>
  %prb2 = llhd.prb %sig2 : !hw.struct<a: i32, b: i32>

  //      CHECK: [[V14:%.+]] = hw.struct_extract %init["a"]
  // CHECK-NEXT: llhd.drv [[V8]], [[V14]] after
  //      CHECK: [[V15:%.+]] = hw.struct_extract %init["b"]
  // CHECK-NEXT: llhd.drv [[V10]], [[V15]] after
  llhd.drv %sig2, %init after %time : !hw.struct<a: i32, b: i32>

  // CHECK-NEXT: hw.output [[V6]], [[V13]] : !hw.struct<a: i32, b: i32>, !hw.struct<a: i32, b: i32>
  hw.output %prb, %prb2 : !hw.struct<a: i32, b: i32>, !hw.struct<a: i32, b: i32>
}

// CHECK-LABEL: @array
hw.module @array(in %init : !hw.array<2xi32>, in %v1 : i32, in %v2 : i32, out out1 : !hw.array<2xi32>, out out2 : !hw.array<2xi32>) {
  %time = llhd.constant_time <0ns, 0d, 1e>
  %false = hw.constant false
  %true = hw.constant true

  //      CHECK: [[V0:%.+]] = hw.array_get %init[%false
  // CHECK-NEXT: [[V1:%.+]] = llhd.sig
  // CHECK-SAME: [[V0]]
  //      CHECK: [[V2:%.+]] = hw.array_get %init[%true
  // CHECK-NEXT: [[V3:%.+]] = llhd.sig
  // CHECK-SAME: [[V2]]
  %sig = llhd.sig %init : !hw.array<2xi32>

  // CHECK-NEXT: [[V4:%.+]] = llhd.prb [[V1]]
  // CHECK-NEXT: [[V5:%.+]] = llhd.prb [[V3]]
  // CHECK-NEXT: [[V6:%.+]] = hw.array_create [[V4]], [[V5]] : i32
  %prb = llhd.prb %sig : !hw.array<2xi32>

  %a = llhd.sig.array_get %sig[%false] : !llhd.ref<!hw.array<2xi32>>
  %b = llhd.sig.array_get %sig[%true] : !llhd.ref<!hw.array<2xi32>>

  // CHECK-NEXT: llhd.drv [[V1]], %v1 after
  // CHECK-NEXT: llhd.drv [[V3]], %v2 after
  llhd.drv %a, %v1 after %time : i32
  llhd.drv %b, %v2 after %time : i32

  //      CHECK: [[V7:%.+]] = hw.array_get %init[%false
  // CHECK-NEXT: [[V8:%.+]] = llhd.sig
  // CHECK-SAME: [[V7]]
  //      CHECK: [[V9:%.+]] = hw.array_get %init[%true
  // CHECK-NEXT: [[V10:%.+]] = llhd.sig
  // CHECK-SAME: [[V9]]
  %sig2 = llhd.sig %init : !hw.array<2xi32>

  // CHECK-NEXT: [[V11:%.+]] = llhd.prb [[V8]]
  // CHECK-NEXT: [[V12:%.+]] = llhd.prb [[V10]]
  // CHECK-NEXT: [[V13:%.+]] = hw.array_create [[V11]], [[V12]] : i32
  %prb2 = llhd.prb %sig2 : !hw.array<2xi32>

  //      CHECK: [[V14:%.+]] = hw.array_get %init[%false
  // CHECK-NEXT: llhd.drv [[V8]], [[V14]] after
  //      CHECK: [[V15:%.+]] = hw.array_get %init[%true
  // CHECK-NEXT: llhd.drv [[V10]], [[V15]] after
  llhd.drv %sig2, %init after %time : !hw.array<2xi32>

  // CHECK-NEXT: hw.output [[V6]], [[V13]] : !hw.array<2xi32>, !hw.array<2xi32>
  hw.output %prb, %prb2 : !hw.array<2xi32>, !hw.array<2xi32>
}

// CHECK-LABEL: @arrayDynamicAccess
hw.module @arrayDynamicAccess(in %init : !hw.array<2xi32>, in %v1 : i32, in %idx : i1, out out : !hw.array<2xi32>) {
  %time = llhd.constant_time <0ns, 0d, 1e>
  %false = hw.constant false

  // No SROA performed when array access indices are not statically known
  //      CHECK: [[V0:%.+]] = llhd.sig %init
  // CHECK-NEXT: [[V1:%.+]] = llhd.prb [[V0]]
  // CHECK-NEXT: [[V2:%.+]] = llhd.sig.array_get [[V0]][%false
  // CHECK-NEXT: [[V3:%.+]] = llhd.sig.array_get [[V0]][%idx
  // CHECK-NEXT: llhd.drv [[V2]], %v1 after
  // CHECK-NEXT: llhd.drv [[V3]], %v1 after
  // CHECK-NEXT: hw.output [[V1]]
  %sig = llhd.sig %init : !hw.array<2xi32>
  %prb = llhd.prb %sig : !hw.array<2xi32>
  %a = llhd.sig.array_get %sig[%false] : !llhd.ref<!hw.array<2xi32>>
  %b = llhd.sig.array_get %sig[%idx] : !llhd.ref<!hw.array<2xi32>>
  llhd.drv %a, %v1 after %time : i32
  llhd.drv %b, %v1 after %time : i32

  hw.output %prb : !hw.array<2xi32>
}

// CHECK-LABEL: @nested
hw.module @nested(in %init : !hw.array<2x!hw.struct<a: i32, b: i32>>, out out : !hw.array<2x!hw.struct<a: i32, b: i32>>) {
  %time = llhd.constant_time <0ns, 0d, 1e>

  // CHECK: [[V0:%.+]] = hw.array_get %init[%false
  // CHECK: [[V1:%.+]] = hw.struct_extract [[V0]]["a"]
  // CHECK: [[V2:%.+]] = llhd.sig{{ }}
  // CHECK-SAME: [[V1]]
  // CHECK: [[V3:%.+]] = hw.struct_extract [[V0]]["b"]
  // CHECK: [[V4:%.+]] = llhd.sig{{ }}
  // CHECK-SAME: [[V3]]
  // CHECK: [[V5:%.+]] = hw.array_get %init[%true
  // CHECK: [[V6:%.+]] = hw.struct_extract [[V5]]["a"]
  // CHECK: [[V7:%.+]] = llhd.sig{{ }}
  // CHECK-SAME: [[V6]]
  // CHECK: [[V8:%.+]] = hw.struct_extract [[V5]]["b"]
  // CHECK: [[V9:%.+]] = llhd.sig{{ }}
  // CHECK-SAME: [[V8]]
  %sig = llhd.sig %init : !hw.array<2x!hw.struct<a: i32, b: i32>>

  // CHECK: [[V10:%.+]] = llhd.prb [[V2]]
  // CHECK: [[V11:%.+]] = llhd.prb [[V4]]
  // CHECK: [[V12:%.+]] = hw.struct_create ([[V10]], [[V11]]) : !hw.struct<a: i32, b: i32>
  // CHECK: [[V13:%.+]] = llhd.prb [[V7]]
  // CHECK: [[V14:%.+]] = llhd.prb [[V9]]
  // CHECK: [[V15:%.+]] = hw.struct_create ([[V13]], [[V14]]) : !hw.struct<a: i32, b: i32>
  // CHECK: [[V16:%.+]] = hw.array_create [[V12]], [[V15]] :
  %0 = llhd.prb %sig : !hw.array<2x!hw.struct<a: i32, b: i32>>

  // CHECK: [[V17:%.+]] = hw.array_get %init[%false
  // CHECK: [[V18:%.+]] = hw.struct_extract [[V17]]["a"]
  // CHECK: llhd.drv [[V2]], [[V18]] after
  // CHECK: [[V19:%.+]] = hw.struct_extract [[V17]]["b"]
  // CHECK: llhd.drv [[V4]], [[V19]] after
  // CHECK: [[V20:%.+]] = hw.array_get %init[%true
  // CHECK: [[V21:%.+]] = hw.struct_extract [[V20]]["a"]
  // CHECK: llhd.drv [[V7]], [[V21]] after
  // CHECK: [[V22:%.+]] = hw.struct_extract [[V20]]["b"]
  // CHECK: llhd.drv [[V9]], [[V22]] after
  llhd.drv %sig, %init after %time : !hw.array<2x!hw.struct<a: i32, b: i32>>

  // CHECK: hw.output [[V16]] : !hw.array<2xstruct<a: i32, b: i32>>
  hw.output %0 : !hw.array<2x!hw.struct<a: i32, b: i32>>
}
