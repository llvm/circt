// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK-LABEL: om.class @Thingy
// CHECK-SAME: (%blue_1: i8, %blue_2: i32)
om.class @Thingy(%blue_1: i8, %blue_2: i32) {
  // CHECK: %[[widget:.+]] = om.object @Widget(%blue_1, %blue_2) : (i8, i32) -> !om.class.type<@Widget>
  %0 = om.object @Widget(%blue_1, %blue_2) : (i8, i32) -> !om.class.type<@Widget>
  // CHECK: om.class.field @widget, %[[widget]] : !om.class.type<@Widget>
  om.class.field @widget, %0 : !om.class.type<@Widget>

  // CHECK: %[[gadget:.+]] = om.object @Gadget(%blue_1, %blue_2) : (i8, i32) -> !om.class.type<@Gadget>
  %1 = om.object @Gadget(%blue_1, %blue_2) : (i8, i32) -> !om.class.type<@Gadget>
  // CHECK: om.class.field @gadget, %[[gadget]] : !om.class.type<@Gadget>
  om.class.field @gadget, %1 : !om.class.type<@Gadget>

  // CHECK: om.class.field @blue_1, %blue_1 : i8
  om.class.field @blue_1, %blue_1 : i8

  // CHECK: %[[widget_field:.+]] = om.object.field %[[widget]], [@blue_1] : (!om.class.type<@Widget>) -> i8
  %2 = om.object.field %0, [@blue_1] : (!om.class.type<@Widget>) -> i8
  // CHECK: om.class.field @blue_2, %[[widget_field]] : i8
  om.class.field @blue_2, %2 : i8
}

// CHECK-LABEL: om.class @Widget
// CHECK-SAME: (%blue_1: i8, %green_1: i32)
om.class @Widget(%blue_1: i8, %green_1: i32) {
  // CHECK: om.class.field @blue_1, %blue_1 : i8
  om.class.field @blue_1, %blue_1 : i8
  // CHECK: om.class.field @green_1, %green_1 : i32
  om.class.field @green_1, %green_1 : i32
}

// CHECK-LABEL: om.class @Gadget
// CHECK-SAME: (%green_1: i8, %green_2: i32)
om.class @Gadget(%green_1: i8, %green_2: i32) {
  // CHECK: om.class.field @green_1, %green_1 : i8
  om.class.field @green_1, %green_1 : i8
  // CHECK: om.class.field @green_2, %green_2 : i32
  om.class.field @green_2, %green_2 : i32
}

// CHECK-LABEL: om.class @Empty
om.class @Empty() {}

om.class @NestedField1(%arg0: i1) {
  om.class.field @baz, %arg0 : i1
}

om.class @NestedField2(%arg0: i1) {
  %0 = om.object @NestedField1(%arg0) : (i1) -> !om.class.type<@NestedField1>
  om.class.field @bar, %0 : !om.class.type<@NestedField1>
}

om.class @NestedField3(%arg0: i1) {
  %0 = om.object @NestedField2(%arg0) : (i1) -> !om.class.type<@NestedField2>
  om.class.field @foo, %0 : !om.class.type<@NestedField2>
}

// CHECK-LABEL: @NestedField4
om.class @NestedField4(%arg0: i1) {
  // CHECK: %[[nested:.+]] = om.object @NestedField3
  %0 = om.object @NestedField3(%arg0) : (i1) -> !om.class.type<@NestedField3>
  // CHECK: %{{.+}} = om.object.field %[[nested]], [@foo, @bar, @baz] : (!om.class.type<@NestedField3>) -> i1
  %1 = om.object.field %0, [@foo, @bar, @baz] : (!om.class.type<@NestedField3>) -> i1
}
