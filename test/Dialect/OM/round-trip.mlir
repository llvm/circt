// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK-LABEL: om.class @Thingy
// CHECK-SAME: (%blue_1: i8)
om.class @Thingy(%blue_1: i8) {
  // CHECK: om.class.field @blue_1, %blue_1 : i8
  om.class.field @blue_1, %blue_1 : i8
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

// CHECK-LABEL: om.class @DiscardableAttrs
om.class @DiscardableAttrs() attributes {foo.bar="baz"} {}
