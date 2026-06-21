// RUN: circt-opt %s --split-input-file --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: func.func @uarray_ref_to_int_ref(
// CHECK-SAME: %[[ARG0:.*]]: !llhd.ref<!hw.array<16xi1>>) -> !llhd.ref<i16>
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !llhd.ref<!hw.array<16xi1>> to !llhd.ref<i16>
// CHECK: return %[[CAST]] : !llhd.ref<i16>
func.func @uarray_ref_to_int_ref(%arg0: !moore.ref<uarray<16 x l1>>) -> !moore.ref<l16> {
  %0 = moore.conversion %arg0 : !moore.ref<uarray<16 x l1>> -> !moore.ref<l16>
  return %0 : !moore.ref<l16>
}

// CHECK-LABEL: func.func @struct_ref_to_int_ref(
// CHECK-SAME: %[[ARG0:.*]]: !llhd.ref<!hw.struct<a: i3, b: i2>>) -> !llhd.ref<i5>
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !llhd.ref<!hw.struct<a: i3, b: i2>> to !llhd.ref<i5>
// CHECK: return %[[CAST]] : !llhd.ref<i5>
func.func @struct_ref_to_int_ref(%arg0: !moore.ref<struct<{a: l3, b: l2}>>) -> !moore.ref<l5> {
  %0 = moore.conversion %arg0 : !moore.ref<struct<{a: l3, b: l2}>> -> !moore.ref<l5>
  return %0 : !moore.ref<l5>
}
