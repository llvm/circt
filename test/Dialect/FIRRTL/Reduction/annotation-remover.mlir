// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "class = \"a\"" --include annotation-remover --keep-best=0 | FileCheck %s --check-prefixes=CHECK,CHECK-A
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "class = \"x\"" --include annotation-remover --keep-best=0 | FileCheck %s --check-prefixes=CHECK,CHECK-X

// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129

// This test verifies that the AnnotationRemover can selectively remove individual annotations.
// The test uses grep to look for annotation "a", so the reducer should keep that annotation
// but remove annotations "b" and "c" that don't match the grep pattern.

// CHECK-LABEL: firrtl.circuit "TestAnnotationRemover"
firrtl.circuit "TestAnnotationRemover" {
  // CHECK: firrtl.module @TestAnnotationRemover
  // CHECK-A-SAME: [{class = "a"}]
  firrtl.module @TestAnnotationRemover(
    in %a: !firrtl.uint<1> [
      {class = "a"},
      {class = "b"},
      {class = "c"}
    ]
  ) {
    // CHECK: firrtl.wire
    // CHECK-X-SAME: [{class = "x"}]
    %someWire = firrtl.wire {annotations = [
      {class = "x"},
      {class = "y"},
      {class = "z"}
    ]} : !firrtl.uint<8>
  }
}

// CHECK-LABEL: firrtl.circuit "DontRemoveNLAsWithUsesOutsideOfAnnotations"
firrtl.circuit "DontRemoveNLAsWithUsesOutsideOfAnnotations" {
  firrtl.extmodule @DontRemoveNLAsWithUsesOutsideOfAnnotations()

  // CHECK: hw.hierpath private @nla1
  hw.hierpath private @nla1 [@Foo::@bar]
  // CHECK-NOT: hw.hierpath private @nla2
  hw.hierpath private @nla2 [@Foo::@bar]

  // CHECK: firrtl.module @Foo
  // CHECK-SAME: someSymbolUse = @nla1
  firrtl.module @Foo() attributes {someSymbolUse = @nla1} {
    // CHECK-NEXT: firrtl.instance bar
    // CHECK-NOT: @nla1
    // CHECK-NOT: @nla2
    firrtl.instance bar sym @bar {annotations = [
      {circt.nonlocal = @nla1},
      {circt.nonlocal = @nla2}
    ]} @Bar()
  }

  firrtl.extmodule @Bar()
}
