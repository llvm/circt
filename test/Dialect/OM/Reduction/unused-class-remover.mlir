// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "om.class @UsedClass" --include om-unused-class-remover --keep-best=0 | FileCheck %s

// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129

// This test verifies that the OMUnusedClassRemover can remove om.class definitions
// that are never instantiated via om.object.

module {
  // This class is never instantiated, so it should be removed
  // CHECK-NOT: om.class @UnusedClass
  om.class @UnusedClass (%input: !om.integer) -> (output: !om.integer) {
    om.class.fields %input : !om.integer
  }

  // This class is only referenced via om.unknown type, but we can't remove it
  // without also removing the om.unknown operation that references it.
  // The generic dead code elimination will handle removing the om.unknown,
  // then this class can be removed.
  // CHECK: om.class @OnlyUnknownClass
  om.class @OnlyUnknownClass() {
    om.class.fields
  }

  // This class is actually instantiated, so it should be kept
  // CHECK: om.class @UsedClass
  om.class @UsedClass(%input: !om.string) -> (output: !om.string) {
    om.class.fields %input : !om.string
  }

  // CHECK: om.class @Main
  om.class @Main(%basepath: !om.frozenbasepath) -> (result: !om.any) {
    %0 = om.constant "test" : !om.string

    // This instantiation keeps @UsedClass alive
    // CHECK: om.object @UsedClass
    %1 = om.object @UsedClass(%0) : (!om.string) -> !om.class.type<@UsedClass>

    // This unknown reference keeps @OnlyUnknownClass alive because
    // we can't remove a class that is referenced in a type
    // CHECK: om.unknown
    %2 = om.unknown : !om.class.type<@OnlyUnknownClass>

    %3 = om.any_cast %1 : (!om.class.type<@UsedClass>) -> !om.any

    om.class.fields %3 : !om.any
  }
}
