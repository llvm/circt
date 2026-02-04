// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "list.create.*%%s1" --include list-create-element-remover --keep-best=0 | FileCheck %s --check-prefixes=CHECK,CHECK-S1
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "list.create.*%%s2" --include list-create-element-remover --keep-best=0 | FileCheck %s --check-prefixes=CHECK,CHECK-S2

// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129

// This test verifies that the ListCreateElementRemover can selectively remove individual elements
// from list create operations. The test uses grep to look for specific elements, so the reducer
// should keep that element but remove others that don't match the grep pattern.

firrtl.circuit "ListCreateElementRemover" {
  // CHECK-LABEL: firrtl.class @ClassTest
  firrtl.class @ClassTest() {
  }

  // CHECK-LABEL: firrtl.module @ListCreateElementRemover
  firrtl.module @ListCreateElementRemover(
    in %s1: !firrtl.string,
    in %s2: !firrtl.string,
    in %s3: !firrtl.string,
    in %c1: !firrtl.class<@ClassTest()>,
    in %c2: !firrtl.class<@ClassTest()>,
    in %c3: !firrtl.class<@ClassTest()>,
    out %out_strings: !firrtl.list<string>,
    out %out_objs: !firrtl.list<class<@ClassTest()>>
  ) {
    // Test with string list - when looking for s1, should keep s1 and remove s2, s3
    // CHECK: %[[STRINGS:.+]] = firrtl.list.create
    // CHECK-S1-SAME: %s1
    // CHECK-S1-NOT: %s2
    // CHECK-S1-NOT: %s3
    // CHECK-S2-SAME: %s2
    // CHECK-S2-NOT: %s1
    // CHECK-S2-NOT: %s3
    %0 = firrtl.list.create %s1, %s2, %s3 : !firrtl.list<string>
    firrtl.propassign %out_strings, %0 : !firrtl.list<string>

    // Test with object list
    // CHECK: %[[OBJS:.+]] = firrtl.list.create
    %1 = firrtl.list.create %c1, %c2, %c3 : !firrtl.list<class<@ClassTest()>>
    firrtl.propassign %out_objs, %1 : !firrtl.list<class<@ClassTest()>>
  }
}

