// RUN: om-linker %S/Inputs/a.mlir %S/Inputs/b.mlir | FileCheck %s
// CHECK:      module {
// CHECK-NEXT:   om.class @A(%arg: i1) {
// CHECK-NEXT:   }
// CHECK-NEXT:   om.class @Conflict_a() {
// CHECK-NEXT:   }
// CHECK-NEXT:   hw.module private @M2_1() {
// CHECK-NEXT:     hw.output
// CHECK-NEXT:   }
// CHECK-NEXT:   hw.module @M1(in %a : i1) {
// CHECK-NEXT:     hw.instance "" @M2_1() -> ()
// CHECK-NEXT:     hw.output
// CHECK-NEXT:   }
// CHECK-NEXT:   hw.module private @M2_2(in %a : i1) {
// CHECK-NEXT:     hw.instance "" @M1(a: %a: i1) -> ()
// CHECK-NEXT:     hw.output
// CHECK-NEXT:   }
// CHECK-NEXT:   hw.module public @Top(in %a : i1) {
// CHECK-NEXT:     hw.instance "" @M2_2(a: %a: i1) -> ()
// CHECK-NEXT:     hw.output
// CHECK-NEXT:   }
// CHECK-NEXT:   om.class @B(%arg: i2) {
// CHECK-NEXT:   }
// CHECK-NEXT:   om.class @Conflict_b() {
// CHECK-NEXT:   }
// CHECK-NEXT: }
