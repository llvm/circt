// RUN: om-linker %S/Inputs/a.mlir %S/Inputs/b.mlir %S/Inputs/other.mlir | FileCheck %s
// CHECK:      module {
// CHECK-NEXT:   module attributes {om.namespace = "a"} {
// CHECK-NEXT:     hw.module @hello() {
// CHECK-NEXT:       hw.output
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   module attributes {om.namespace = "b"} {
// CHECK-NEXT:     hw.module.extern @hello()
// CHECK-NEXT:   }
// CHECK-NEXT:   module attributes {om.namespace = "other"} {
// CHECK-NEXT:     hw.module @HW(in %a : i1, out b : i1) {
// CHECK-NEXT:       %x_i1 = sv.constantX : i1
// CHECK-NEXT:       %0 = comb.xor %a, %x_i1 : i1
// CHECK-NEXT:       %1 = ltl.and %0 : i1
// CHECK-NEXT:       verif.assume %0 : i1
// CHECK-NEXT:       hw.output %0 : i1
// CHECK-NEXT:     }
// CHECK-NEXT:     emit.file "foo.sv" sym @Emit {
// CHECK-NEXT:     }
// CHECK-NEXT:   }
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