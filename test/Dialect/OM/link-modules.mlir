// RUN: circt-opt %s --verify-diagnostics -pass-pipeline='builtin.module(om-link-modules)' --split-input-file -allow-unregistered-dialect | FileCheck %s --implicit-check-not=om.class.extern

module {
  // CHECK-LABEL: module
  // CHECK-NEXT:   module
  // CHECK-NOT:   om.class.extern
  // CHECK-LABEL: om.class @A
  // CHECK-LABEL: om.class @Conflict_A
  // CHECK-LABEL: om.class @UseConflict_A
  // CHECK-SAME:      -> (c: !om.class.type<@Conflict_A>, d: !om.class.type<@Conflict_A>)
  // CHECK-NEXT:    om.object @Conflict_A() : () -> !om.class.type<@Conflict_A>
  // CHECK:         om.elaborated_object @Conflict_A(%{{.+}}) : (i1) -> !om.class.type<@Conflict_A>
  // CHECK-NEXT:    om.class.fields %{{.+}}, %{{.+}} : !om.class.type<@Conflict_A>, !om.class.type<@Conflict_A>

  // CHECK-LABEL: om.class @Conflict_B
  // CHECK-LABEL: om.class @UseConflict_B()
  // CHECK-NEXT:    om.object @Conflict_B() : () -> !om.class.type<@Conflict_B>
  // CHECK-NEXT:    om.object.field %{{.+}}["c"] : (!om.class.type<@Conflict_B>) -> i1
  // CHECK:         om.elaborated_object @Conflict_B(%{{.+}}) : (i1) -> !om.class.type<@Conflict_B>
  // CHECK-NEXT:    om.object.field %{{.+}}["c"] : (!om.class.type<@Conflict_B>) -> i1

  // CHECK-LABEL: om.class @Conflict_module_0()
  // CHECK-LABEL: om.class @UseConflict_module_0()
  // CHECK-NEXT:    om.object @Conflict_module_0() : () -> !om.class.type<@Conflict_module_0>
  // CHECK-NEXT:    om.object.field %{{.+}}["c"] : (!om.class.type<@Conflict_module_0>) -> i1
  // CHECK:         om.elaborated_object @Conflict_module_0(%{{.+}}) : (i1) -> !om.class.type<@Conflict_module_0>

  module attributes {om.namespace = "A"} {
    om.class @A(%arg: i1) -> (a: i1) {
      om.class.fields %arg : i1
    }
    om.class @Conflict() -> (c: i1) {
      %0 = om.constant 0 : i1
      om.class.fields %0 : i1
    }
    om.class @UseConflict() -> (c: !om.class.type<@Conflict>, d: !om.class.type<@Conflict>){
      %0 = om.object @Conflict() : () -> !om.class.type<@Conflict>
      %1 = om.constant 0 : i1
      %2 = om.elaborated_object @Conflict(%1) : (i1) -> !om.class.type<@Conflict>
      om.class.fields %0, %2 : !om.class.type<@Conflict>, !om.class.type<@Conflict>
    }
  }
  module attributes {om.namespace = "B"} {
    om.class.extern @A(%arg: i1) -> (a: i1) {}
    om.class @Conflict() -> (c: i1) {
      %0 = om.constant 0 : i1
      om.class.fields %0 : i1
    }
    om.class @UseConflict() -> (c: i1, d: i1) {
      %0 = om.object @Conflict() : () -> !om.class.type<@Conflict>
      %1 = om.object.field %0["c"] : (!om.class.type<@Conflict>) -> i1
      %2 = om.elaborated_object @Conflict(%1) : (i1) -> !om.class.type<@Conflict>
      %3 = om.object.field %2["c"] : (!om.class.type<@Conflict>) -> i1
      om.class.fields %1, %3 : i1, i1
    }
  }
  module {
    om.class @Conflict() -> (c: i1) {
      %0 = om.constant 0 : i1
      om.class.fields %0 : i1
    }
    om.class @UseConflict() {
     %0 = om.object @Conflict() : () -> !om.class.type<@Conflict>
     %1 = om.object.field %0["c"] : (!om.class.type<@Conflict>) -> i1
     %2 = om.elaborated_object @Conflict(%1) : (i1) -> !om.class.type<@Conflict>
     om.class.fields
    }
  }
}

// -----

// Check that OM ops are not deleted.  Make the "dont-delete-me" op a landmine that will
// cause a symbol collision if it is moved to the top level.
module {
  module {
    // CHECK: dont-delete-me
    "dont-delete-me"() {sym_name = "Bar"} : () -> ()
    om.class @Foo() {
      om.class.fields
    }
  }
  module {
    om.class @Bar() {
      om.class.fields
    }
  }
}

// -----

// Deduplicate equivalent definitions, including recursive container types.
// CHECK-LABEL: om.class @A(
// CHECK-SAME: !om.list<!om.class.type<@B>>
// CHECK-LABEL: om.class @B(
// CHECK-SAME: !om.class.type<@A>
// CHECK-NOT: om.class @
module {
  module {
    om.class @A(%b: !om.list<!om.class.type<@B>>) -> (b: !om.list<!om.class.type<@B>>) {
      om.class.fields %b : !om.list<!om.class.type<@B>>
    }
    om.class @B(%a: !om.class.type<@A>) -> (a: !om.class.type<@A>) { om.class.fields %a : !om.class.type<@A> }
  }
  module {
    om.class @A(%b: !om.list<!om.class.type<@B>>) -> (b: !om.list<!om.class.type<@B>>) {
      om.class.fields %b : !om.list<!om.class.type<@B>>
    }
    om.class @B(%a: !om.class.type<@A>) -> (a: !om.class.type<@A>) { om.class.fields %a : !om.class.type<@A> }
  }
  module {
    om.class.extern @A(%b: !om.list<!om.class.type<@B>>) -> (b: !om.list<!om.class.type<@B>>) {}
  }
}

// -----

// Keep different field mappings and their users separate.
// CHECK-LABEL: om.class @Mapping_A
// CHECK-LABEL: om.class @User_A
// CHECK-SAME: !om.class.type<@Mapping_A>
// CHECK-LABEL: om.class @Mapping_B
// CHECK-LABEL: om.class @User_B
// CHECK-SAME: !om.class.type<@Mapping_B>
module {
  module attributes {om.namespace = "A"} {
    om.class @Mapping(%x: i1, %y: i1) -> (value: i1) { om.class.fields %x : i1 }
    om.class @User(%obj: !om.class.type<@Mapping>) -> (obj: !om.class.type<@Mapping>) {
      om.class.fields %obj : !om.class.type<@Mapping>
    }
  }
  module attributes {om.namespace = "B"} {
    om.class @Mapping(%x: i1, %y: i1) -> (value: i1) { om.class.fields %y : i1 }
    om.class @User(%obj: !om.class.type<@Mapping>) -> (obj: !om.class.type<@Mapping>) {
      om.class.fields %obj : !om.class.type<@Mapping>
    }
  }
}

// -----

// Classes referencing a unique computational definition can be deduplicated.
// CHECK-LABEL: om.class @Holder(
// CHECK-SAME: !om.class.type<@Helper>
// CHECK-LABEL: om.class private @Helper()
// CHECK-NOT: om.class @Holder
module {
  module {
    om.class @Holder(%obj: !om.class.type<@Helper>) { om.class.fields }
    om.class private @Helper() -> (x: i1) { %x = om.constant false om.class.fields %x : i1 }
  }
  module {
    om.class.extern private @Helper() -> (x: i1) {}
    om.class @Holder(%obj: !om.class.type<@Helper>) { om.class.fields }
  }
}
