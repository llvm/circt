// RUN: om-linker %S/Inputs/a.mlir %S/Inputs/b.mlir | FileCheck %s
// CHECK:      module {
// CHECK-NEXT:   hw.hierpath private @xmr_0 [@M1::@s1, @M2_0]
// CHECK-NEXT:   om.class @A(%arg: i1) {
// CHECK-NEXT:   }
// CHECK-NEXT:   om.class @Conflict_a() {
// CHECK-NEXT:   }
// CHECK-NEXT:   hw.module private @M2_0() {
// CHECK-NEXT:     hw.output
// CHECK-NEXT:   }
// CHECK-NEXT:   hw.module @M1(in %a : i1) {
// CHECK-NEXT:     sv.always posedge %a {
// CHECK-NEXT:       sv.if %a {
// CHECK-NEXT:         hw.instance "" sym @s1 @M2_0() -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     hw.output
// CHECK-NEXT:   }
// CHECK-NEXT:   hw.module private @M2_1(in %a : i1) {
// CHECK-NEXT:     hw.instance "" @M1(a: %a: i1) -> ()
// CHECK-NEXT:     hw.output
// CHECK-NEXT:   }
// CHECK-NEXT:   hw.module @xmr() {
// CHECK-NEXT:     hw.output
// CHECK-NEXT:   }
// CHECK-NEXT:   hw.module public @Top(in %a : i1) {
// CHECK-NEXT:     hw.instance "" @M2_1(a: %a: i1) -> ()
// CHECK-NEXT:     hw.output
// CHECK-NEXT:   }
// CHECK-NEXT:   om.class @B(%arg: i2) {
// CHECK-NEXT:   }
// CHECK-NEXT:   om.class @Conflict_b() {
// CHECK-NEXT:   }
// CHECK-NEXT: }
