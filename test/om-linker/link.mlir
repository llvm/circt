// RUN: om-linker %S/Inputs/a.mlir %S/Inputs/b.mlir %S/Inputs/other.mlir | FileCheck %s
// RUN: om-linker %S/Inputs/a.mlir %S/Inputs/a.mlir | FileCheck %s --check-prefix=DUP
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
// CHECK:            hw.output
// CHECK-NEXT:     }
// CHECK-NEXT:     emit.file "foo.sv" sym @Emit {
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   om.class @A(%arg: i1) {
// CHECK-NEXT:     om.class.fields
// CHECK-NEXT:   }
// CHECK-NEXT:   om.class @Conflict_a()
// CHECK-NEXT:     %{{.+}} = om.constant #om.list<!om.class.type<@Conflict_a>, []> : !om.list<!om.class.type<@Conflict_a>>{{$}}
// CHECK-NEXT:     om.class.fields
// CHECK-NEXT:   }
// CHECK-NEXT:   om.class @B(%arg: i2) {
// CHECK-NEXT:     om.class.fields
// CHECK-NEXT:   }
// CHECK-NEXT:   om.class @Conflict_b()
// CHECK-NEXT:     %{{.+}} = om.constant #om.list<!om.class.type<@Conflict_b>, []> : !om.list<!om.class.type<@Conflict_b>>{{$}}
// CHECK-NEXT:     om.class.fields
// CHECK-NEXT:   }
// CHECK-NEXT: }

// Check duplicate filenames/namespaces as well as distinct input namespaces.
// DUP-LABEL: om.class @Conflict_a()
// DUP-NEXT: %{{.+}} = om.constant #om.list<!om.class.type<@Conflict_a>, []> : !om.list<!om.class.type<@Conflict_a>>{{$}}
// DUP-LABEL: om.class @Conflict_0_a()
// DUP-NEXT: %{{.+}} = om.constant #om.list<!om.class.type<@Conflict_0_a>, []> : !om.list<!om.class.type<@Conflict_0_a>>{{$}}
