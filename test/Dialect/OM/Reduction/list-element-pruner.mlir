// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "list_create.*%%0.*:" --include om-list-element-pruner --keep-best=0 | FileCheck %s --check-prefixes=CHECK,CHECK-0
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "list_create.*%%1.*:" --include om-list-element-pruner --keep-best=0 | FileCheck %s --check-prefixes=CHECK,CHECK-1
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "list_create.*%%0.*%%1.*:" --include om-list-element-pruner --keep-best=0 | FileCheck %s --check-prefixes=CHECK,CHECK-0,CHECK-1
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "list_create" --include om-list-element-pruner --keep-best=0 | FileCheck %s --check-prefixes=CHECK,CHECK-NONE

// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129

// This test verifies that the OMListElementPruner can selectively remove individual
// elements from om.list_create operations.

module {
  // CHECK-LABEL: om.class @Foo
  om.class @Foo(%basepath: !om.frozenbasepath) -> (result: !om.list<!om.string>) {
    %0 = om.constant "foo" : !om.string
    %1 = om.constant "bar" : !om.string
    // CHECK: om.list_create
    // CHECK-0-SAME: %0
    // CHECK-0-NOT: %1
    // CHECK-1-NOT: %0
    // CHECK-1-SAME: %1
    // CHECK-NONE-NOT: %0
    // CHECK-NONE-NOT: %1
    %2 = om.list_create %0, %1 : !om.string
    om.class.fields %2 : !om.list<!om.string>
  }
}
