// RUN: circt-opt %s --arc-flatten-structs | FileCheck %s

// CHECK-LABEL: hw.module @StructPorts(in %arg0 : i8, out out : i8)
hw.module @StructPorts(in %arg0 : !hw.struct<a: i3, b: i5>, out out : !hw.struct<a: i3, b: i5>) {
  hw.output %arg0 : !hw.struct<a: i3, b: i5>
}

// CHECK-LABEL: hw.module @StructExtractAndCreate(in %arg0 : i15, out out : i15)
hw.module @StructExtractAndCreate(in %arg0 : !hw.struct<x: i3, y: i5, z: i7>, out out : !hw.struct<x: i3, y: i5, z: i7>) {
  // CHECK: %[[X:.+]] = comb.extract %arg0 from 12 : (i15) -> i3
  // CHECK: %[[Y:.+]] = comb.extract %arg0 from 7 : (i15) -> i5
  // CHECK: %[[Z:.+]] = comb.extract %arg0 from 0 : (i15) -> i7
  // CHECK: %[[RES:.+]] = comb.concat %[[X]], %[[Y]], %[[Z]] : i3, i5, i7
  // CHECK: hw.output %[[RES]] : i15
  %x = hw.struct_extract %arg0["x"] : !hw.struct<x: i3, y: i5, z: i7>
  %y = hw.struct_extract %arg0["y"] : !hw.struct<x: i3, y: i5, z: i7>
  %z = hw.struct_extract %arg0["z"] : !hw.struct<x: i3, y: i5, z: i7>
  %res = hw.struct_create(%x, %y, %z) : !hw.struct<x: i3, y: i5, z: i7>
  hw.output %res : !hw.struct<x: i3, y: i5, z: i7>
}

// CHECK-LABEL: hw.module @StructConstant(out out : i11)
hw.module @StructConstant(out out : !hw.struct<tag: i5, val: i6>) {
  // CHECK: %[[C:.+]] = hw.constant -814 : i11
  // CHECK: hw.output %[[C]] : i11
  %c = hw.aggregate_constant [19 : i5, 18 : i6] : !hw.struct<tag: i5, val: i6>
  hw.output %c : !hw.struct<tag: i5, val: i6>
}

// CHECK-LABEL: hw.module @StructInject(in %arg0 : i10, in %new_val : i3, out out : i10)
hw.module @StructInject(in %arg0 : !hw.struct<a: i2, b: i3, c: i5>, in %new_val : i3, out out : !hw.struct<a: i2, b: i3, c: i5>) {
  // CHECK: %[[A:.+]] = comb.extract %arg0 from 8 : (i10) -> i2
  // CHECK: %[[C:.+]] = comb.extract %arg0 from 0 : (i10) -> i5
  // CHECK: %[[RES:.+]] = comb.concat %[[A]], %new_val, %[[C]] : i2, i3, i5
  // CHECK: hw.output %[[RES]] : i10
  %res = hw.struct_inject %arg0["b"], %new_val : !hw.struct<a: i2, b: i3, c: i5>
  hw.output %res : !hw.struct<a: i2, b: i3, c: i5>
}

// CHECK-LABEL: hw.module @NestedStructs(in %arg0 : i12, out out : i12)
hw.module @NestedStructs(in %arg0 : !hw.struct<outer: !hw.struct<a: i2, b: i4>, c: i6>, out out : !hw.struct<outer: !hw.struct<a: i2, b: i4>, c: i6>) {
  // CHECK: %[[OUTER:.+]] = comb.extract %arg0 from 6 : (i12) -> i6
  // CHECK: %[[A:.+]] = comb.extract %[[OUTER]] from 4 : (i6) -> i2
  // CHECK: %[[B:.+]] = comb.extract %[[OUTER]] from 0 : (i6) -> i4
  // CHECK: %[[C:.+]] = comb.extract %arg0 from 0 : (i12) -> i6
  // CHECK: %[[NEW_OUTER:.+]] = comb.concat %[[A]], %[[B]] : i2, i4
  // CHECK: %[[RES:.+]] = comb.concat %[[NEW_OUTER]], %[[C]] : i6, i6
  // CHECK: hw.output %[[RES]] : i12
  %outer = hw.struct_extract %arg0["outer"] : !hw.struct<outer: !hw.struct<a: i2, b: i4>, c: i6>
  %a = hw.struct_extract %outer["a"] : !hw.struct<a: i2, b: i4>
  %b = hw.struct_extract %outer["b"] : !hw.struct<a: i2, b: i4>
  %c = hw.struct_extract %arg0["c"] : !hw.struct<outer: !hw.struct<a: i2, b: i4>, c: i6>
  %new_outer = hw.struct_create(%a, %b) : !hw.struct<a: i2, b: i4>
  %res = hw.struct_create(%new_outer, %c) : !hw.struct<outer: !hw.struct<a: i2, b: i4>, c: i6>
  hw.output %res : !hw.struct<outer: !hw.struct<a: i2, b: i4>, c: i6>
}

// CHECK-LABEL: hw.module.extern @ExternStructPorts(in %arg0 : i8, out out : i8)
hw.module.extern @ExternStructPorts(in %arg0 : !hw.struct<val: i8>, out out : !hw.struct<val: i8>)

// CHECK-LABEL: hw.module @SingleElementStruct(in %arg0 : i8, out out : i8)
hw.module @SingleElementStruct(in %arg0 : !hw.struct<sole: i8>, out out : !hw.struct<sole: i8>) {
  // CHECK: %[[EX:.+]] = comb.extract %arg0 from 0 : (i8) -> i8
  // CHECK-NOT: comb.concat
  // CHECK: hw.output %[[EX]] : i8
  %sole = hw.struct_extract %arg0["sole"] : !hw.struct<sole: i8>
  %res = hw.struct_create(%sole) : !hw.struct<sole: i8>
  hw.output %res : !hw.struct<sole: i8>
}

