// RUN: circt-opt %s | circt-opt | FileCheck %s
// RUN: circt-opt %s -firrtl-lower-classes | FileCheck %s --check-prefix=OM

firrtl.circuit "Create" {
  firrtl.extclass @Class(in str: !firrtl.string, out result: !firrtl.string, in int: !firrtl.integer)
  // CHECK-LABEL: firrtl.module @Create
  // OM-LABEL: om.class @Create_Class
  firrtl.module @Create(in %str: !firrtl.string, in %int: !firrtl.integer, out %out: !firrtl.string) {
    // CHECK: %obj = firrtl.object.create(%str, %int) : (!firrtl.string, !firrtl.integer) -> !firrtl.class<@Class(in str: !firrtl.string, out result: !firrtl.string, in int: !firrtl.integer)>
    // OM: %[[OBJ:.*]] = om.object @Class(%{{.*}}, %{{.*}}, %{{.*}})
    %obj = firrtl.object.create(%str, %int) : (!firrtl.string, !firrtl.integer) -> !firrtl.class<@Class(in str: !firrtl.string, out result: !firrtl.string, in int: !firrtl.integer)>
    // CHECK: firrtl.object.subfield %obj[result]
    // OM: om.object.field %[[OBJ]]["result"]
    %result = firrtl.object.subfield %obj[result] : !firrtl.class<@Class(in str: !firrtl.string, out result: !firrtl.string, in int: !firrtl.integer)>
    firrtl.propassign %out, %result : !firrtl.string
  }
}
