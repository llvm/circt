// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg true --include must-dedup-children | FileCheck %s

// Test that MustDedup annotations are moved from parent modules to their child modules

// CHECK: firrtl.circuit "Top" attributes {annotations = [
// CHECK-DAG: {class = "firrtl.transforms.MustDeduplicateAnnotation", modules = ["~Top|ChildA", "~Top|ChildB"]}
// CHECK-DAG: {class = "firrtl.transforms.MustDeduplicateAnnotation", modules = ["~Top|ChildC", "~Top|ChildD"]}
// CHECK-DAG: {class = "firrtl.transforms.MustDeduplicateAnnotation", modules = ["~Top|ParentA", "~Top|ParentB"]}
// CHECK: ]}

firrtl.circuit "Top" attributes {annotations = [{
  class = "firrtl.transforms.MustDeduplicateAnnotation",
  modules = ["~Top|ParentA", "~Top|ParentB"]
}]} {
  firrtl.module @Top() {
    firrtl.instance parentA @ParentA()
    firrtl.instance parentB @ParentB()
  }

  firrtl.module private @ParentA() {
    firrtl.instance child1 @ChildA()
    firrtl.instance child2 @ChildC()
  }

  firrtl.module private @ParentB() {
    firrtl.instance child1 @ChildB()
    firrtl.instance child2 @ChildD()
  }

  firrtl.module private @ChildA() {
    %w = firrtl.wire : !firrtl.uint<8>
  }

  firrtl.module private @ChildB() {
    %w = firrtl.wire : !firrtl.uint<8>
  }

  firrtl.module private @ChildC() {
    %w = firrtl.wire : !firrtl.uint<8>
  }

  firrtl.module private @ChildD() {
    %w = firrtl.wire : !firrtl.uint<8>
  }
}
