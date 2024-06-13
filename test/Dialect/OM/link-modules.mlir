// RUN: circt-opt %s --verify-diagnostics -pass-pipeline='builtin.module(om-link-modules)' --split-input-file -allow-unregistered-dialect | FileCheck %s

module {
  // CHECK-LABEL: module
  // CHECK-NOT:   module
  // CHECK-NOT:   om.class.extern
  // CHECK-LABEL: om.class @A
  // CHECK-LABEL: om.class @Conflict_A
  // CHECK-LABEL: om.class @UseConflict_A
  // CHECK-NEXT:    om.object @Conflict_A() : () -> !om.class.type<@Conflict_A>
  // CHECK-NEXT:    om.class.field @c, %{{.+}} : !om.class.type<@Conflict_A>

  // CHECK-LABEL: om.class @Conflict_B
  // CHECK-LABEL: om.class @UseConflict_B()
  // CHECK-NEXT:    om.object @Conflict_B() : () -> !om.class.type<@Conflict_B>
  // CHECK-NEXT:    om.object.field %{{.+}}, [@c] : (!om.class.type<@Conflict_B>) -> i1

  // CHECK-LABEL: om.class @Conflict_module_0()
  // CHECK-LABEL: om.class @UseConflict_module_0()
  // CHECK-NEXT:    om.object @Conflict_module_0() : () -> !om.class.type<@Conflict_module_0>
  // CHECK-NEXT:    om.object.field %{{.+}}, [@c] : (!om.class.type<@Conflict_module_0>) -> i1

  module attributes {om.namespace = "A"} {
    om.class @A(%arg: i1) {
      om.class.field @a, %arg: i1
    }
    om.class @Conflict() {
      %0 = om.constant 0 : i1
      om.class.field @c, %0: i1
    }
    om.class @UseConflict() {
      %0 = om.object @Conflict() : () -> !om.class.type<@Conflict>
      om.class.field @c, %0: !om.class.type<@Conflict>
    }
  }
  module attributes {om.namespace = "B"} {
    om.class.extern @A(%arg: i1) {
      om.class.extern.field @a: i1
    }
    om.class @Conflict() {
      %0 = om.constant 0 : i1
      om.class.field @c, %0: i1
    }
    om.class @UseConflict() {
     %0 = om.object @Conflict() : () -> !om.class.type<@Conflict>
     %1 = om.object.field %0, [@c] : (!om.class.type<@Conflict>) -> i1
     om.class.field @c, %1: i1
    }
  }
  module {
    om.class @Conflict() {
      %0 = om.constant 0 : i1
      om.class.field @c, %0: i1
    }
    om.class @UseConflict() {
     %0 = om.object @Conflict() : () -> !om.class.type<@Conflict>
     %1 = om.object.field %0, [@c] : (!om.class.type<@Conflict>) -> i1
    }
  }
}

// -----

// Check that all conflicting classes are updated.
module {
  module {
    // CHECK: hw.module @Conflict
    hw.module @Conflict () {}
  }
  module {
    // CHECK: om.class @Conflict_module_1
    om.class @Conflict(){}
  }
}

// -----

// Check for linking of multiple modules.
module {
  module {
    // CHECK: hw.module private @nla_0() {
    hw.module private @nla() {}
    // CHECK: hw.module private @M2_0
    hw.module private @M2() {}
  }
  module {
    // CHECK: hw.module private @nla_1() {
    hw.module private @nla() {}
    // CHECK: hw.module private @M2_1
    hw.module private @M2() {}
  }
  module {
    hw.module @top() {}
    // CHECK: hw.hierpath private @nla_2 [@M1::@s1, @M2_2]
    hw.hierpath private @nla [@M1::@s1, @M2]
    // CHECK: hw.hierpath private @xmr [@M1::@s1, @M2_2]
    hw.hierpath private @xmr [@M1::@s1, @M2]
    // CHECK: hw.module private @M2_2()
    hw.module private @M2() {}
    hw.module @M1(in %a: i1) {
      hw.instance "" sym @s1 @M2() -> ()
    }
    om.class @PathTest(%basepath : !om.basepath) {
      // CHECK: %0 = om.path_create reference %basepath @nla_2
      %0 = om.path_create reference %basepath @nla
    }
  }
}

// -----

// Check that symbol users are updated in nested regions.
module {
  module {
    hw.module private @M2() {}
    hw.module private @C2() {}
    hw.module private @C1() {}
  }
  module {
    hw.module private @M2() {}
    hw.module private @C2() {}
    hw.module private @C1() {}
    hw.module private @M3() {}
  }
  module {
    // CHECK: om.class @C2_module_2() {
    om.class @C2() {
    }
    // CHECK: om.class @C1_module_2() {
    om.class @C1() {
    }
    hw.module private @M2() {}
    // CHECK: hw.module private @M3_1() {
    hw.module private @M3() {}
    hw.module @M1(in %a: i1) {
      // CHECK:  %1 = "r3"() {symRefName = @M2_2, symRefName2 = @M3_1} : () -> !om.class.type<@C2_module_2>
      %x = "r1"() ({
        "r2"() ({
          %b = "r3"() {symRefName = @M2, symRefName2 = @M3} : () -> !om.class.type<@C2>
        }, {
        }) : () -> ()
      // CHECK:  {ref = @C1_module_2} : () -> !om.class.type<@C2_module_2>
      }) {ref = @C1 } : () -> !om.class.type<@C2>
    }
  }
}
