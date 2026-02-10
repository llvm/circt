// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "om.object.field.*usedField" --include om-class-field-pruner --keep-best=0 | FileCheck %s

// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129

// This test verifies that the OMClassFieldPruner can remove unused output fields
// from om.class definitions.

module {
  // CHECK-LABEL: om.class @Foo(%input: !om.integer) -> (usedField: !om.integer) {
  om.class @Foo(%input: !om.integer) -> (unusedField: !om.integer, usedField: !om.integer, anotherUnused: !om.string) {
    %0 = om.constant "unused" : !om.string

    // CHECK: om.class.fields %{{.+}} : !om.integer
    om.class.fields %input, %input, %0 : !om.integer, !om.integer, !om.string
  }

  // CHECK-LABEL: om.class @Bar
  om.class @Bar(%basepath: !om.frozenbasepath) {
    %0 = om.constant #om.integer<42 : si64> : !om.integer
    %1 = om.object @Foo(%0) : (!om.integer) -> !om.class.type<@Foo>

    // Only access the usedField - the other fields should be removed
    // CHECK: om.object.field %{{.+}}, [@usedField]
    %2 = om.object.field %1, [@usedField] : (!om.class.type<@Foo>) -> !om.integer

    om.class.fields
  }
}
