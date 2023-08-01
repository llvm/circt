// RUN: om-linker %S/Inputs/a.mlir %S/Inputs/b.mlir | FileCheck %s
// CHECK:      module {
// CHECK-NEXT:   module {
// CHECK-NEXT:     om.class @A(%arg: i1) {
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   module {
// CHECK-NEXT:     om.class.extern @A(%arg: i1) {
// CHECK-NEXT:     }
// CHECK-NEXT:     om.class @B(%arg: i2) {
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
