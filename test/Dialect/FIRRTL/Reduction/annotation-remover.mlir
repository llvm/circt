// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "class = \"a\"" --include annotation-remover --keep-best=0 | FileCheck %s --check-prefixes=CHECK,CHECK-A
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "class = \"x\"" --include annotation-remover --keep-best=0 | FileCheck %s --check-prefixes=CHECK,CHECK-X

// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129

// This test verifies that the AnnotationRemover can selectively remove individual annotations.
// The test uses grep to look for annotation "a", so the reducer should keep that annotation
// but remove annotations "b" and "c" that don't match the grep pattern.

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
