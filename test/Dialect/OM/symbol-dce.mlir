// RUN: circt-opt -symbol-dce %s | FileCheck %s

// CHECK-LABEL: om.class private @Referenced
// CHECK-NOT:   om.class private @Unreferenced
om.class private @Referenced() -> () {
  om.class.fields
}

om.class private @Unreferenced() -> () {
  om.class.fields
}

// CHECK-LABEL: om.class @Top
// CHECK-NEXT: om.object @Referenced
// CHECK-NEXT: om.elaborated_object @Referenced
om.class @Top() -> () {
  %0 = om.object @Referenced() : () -> !om.class.type<@Referenced>
  %1 = om.elaborated_object @Referenced() : () -> !om.class.type<@Referenced>
  om.class.fields
}

