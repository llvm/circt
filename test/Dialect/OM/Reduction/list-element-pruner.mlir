// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "list_create.*%%3.*:" --include om-list-element-pruner --keep-best=0 | FileCheck %s --check-prefixes=CHECK,CHECK-ELEM1
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "list_create.*%%4.*:" --include om-list-element-pruner --keep-best=0 | FileCheck %s --check-prefixes=CHECK,CHECK-ELEM2

// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129

// This test verifies that the OMListElementPruner can selectively remove individual
// elements from om.list_create operations.

module {
  // CHECK-LABEL: om.class @Foo
  om.class @Foo() {
    om.class.fields
  }

  // CHECK-LABEL: om.class @ListTest
  om.class @ListTest(%basepath: !om.frozenbasepath) -> (result: !om.list<!om.any>) {
    %elem1 = om.constant "element1" : !om.string
    %elem2 = om.constant "element2" : !om.string
    %elem3 = om.constant "element3" : !om.string

    %0 = om.any_cast %elem1 : (!om.string) -> !om.any
    %1 = om.any_cast %elem2 : (!om.string) -> !om.any
    %2 = om.any_cast %elem3 : (!om.string) -> !om.any

    // When looking for elem1, should keep elem1 and remove elem2, elem3
    // CHECK: %[[LIST:.+]] = om.list_create
    // CHECK-ELEM1-SAME: %3
    // CHECK-ELEM1-NOT: %4
    // CHECK-ELEM1-NOT: %5
    // CHECK-ELEM2-SAME: %4
    // CHECK-ELEM2-NOT: %3
    // CHECK-ELEM2-NOT: %5
    %3 = om.list_create %0, %1, %2 : !om.any

    om.class.fields %3 : !om.list<!om.any>
  }

  // CHECK-LABEL: om.class @ObjectListTest
  om.class @ObjectListTest(%basepath: !om.frozenbasepath) -> (result: !om.list<!om.class.type<@Foo>>) {
    %0 = om.object @Foo() : () -> !om.class.type<@Foo>
    %1 = om.object @Foo() : () -> !om.class.type<@Foo>
    %2 = om.object @Foo() : () -> !om.class.type<@Foo>

    // CHECK: %[[OBJLIST:.+]] = om.list_create
    %4 = om.list_create %0, %1, %2 : !om.class.type<@Foo>

    om.class.fields %4 : !om.list<!om.class.type<@Foo>>
  }
}

