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

// Coalescing relies on the ODR contract, not structural equivalence.
// CHECK-LABEL: om.class @Compute
// CHECK-SAME: attributes {linkonce_odr}
// CHECK-NEXT: om.class.fields %x : !om.integer
// CHECK-NOT: om.class @Compute
// CHECK-NOT: om.integer.add
module {
  module {
    om.class @Compute(%x: !om.integer) -> (value: !om.integer) attributes {linkonce_odr} {
      om.class.fields %x : !om.integer
    }
  }
  module {
    om.class @Compute(%x: !om.integer) -> (value: !om.integer) attributes {linkonce_odr} {
      %zero = om.constant #om.integer<0>
      %result = om.integer.add %x, %zero : !om.integer
      om.class.fields %result : !om.integer
    }
  }
  module {
    om.class.extern @Compute(%x: !om.integer) -> (value: !om.integer) {}
  }
}

// -----

// Public linkonce_odr classes coalesce without external declarations, while
// private classes without the attribute are renamed on collision.
// CHECK-LABEL: om.class @Shared()
// CHECK-SAME: attributes {linkonce_odr}
// CHECK-LABEL: om.class private @Local_A()
// CHECK-LABEL: om.class private @Local_B()
// CHECK-NOT: om.class{{[ ]}}
module {
  module attributes {om.namespace = "A"} {
    om.class @Shared() attributes {linkonce_odr} { om.class.fields }
    om.class private @Local() { om.class.fields }
  }
  module attributes {om.namespace = "B"} {
    om.class @Shared() attributes {linkonce_odr} { om.class.fields }
    om.class private @Local() { om.class.fields }
  }
}

// -----

// A private definition without linkonce_odr can resolve an external class.
// CHECK-LABEL: om.class @Caller()
// CHECK-NEXT: om.object @PrivateDefinition()
// CHECK-LABEL: om.class private @PrivateDefinition()
module {
  module {
    om.class.extern @PrivateDefinition() {}
    om.class @Caller() {
      %0 = om.object @PrivateDefinition() : () -> !om.class.type<@PrivateDefinition>
      om.class.fields
    }
  }
  module {
    om.class private @PrivateDefinition() { om.class.fields }
  }
}
