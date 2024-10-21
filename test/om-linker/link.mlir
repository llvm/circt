// RUN: om-linker %S/Inputs/a.mlir %S/Inputs/b.mlir %S/Inputs/other.mlir | FileCheck %s
// CHECK:      module {
// CHECK-NEXT:   om.class @A(%arg: i1) {
// CHECK-NEXT:     om.class.fields
// CHECK-NEXT:   }
// CHECK-NEXT:   om.class @Conflict_a() {
// CHECK-NEXT:     om.class.fields
// CHECK-NEXT:   }
// CHECK-NEXT:   om.class @B(%arg: i2) {
// CHECK-NEXT:     om.class.fields
// CHECK-NEXT:   }
// CHECK-NEXT:   om.class @Conflict_b() {
// CHECK-NEXT:     om.class.fields
// CHECK-NEXT:   }
// CHECK-NEXT: }
