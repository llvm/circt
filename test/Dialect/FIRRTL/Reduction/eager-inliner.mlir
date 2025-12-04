// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg true --keep-best=0 --include firrtl-eager-inliner | FileCheck %s

// Test that the EagerInliner reduction does not inline instances that participate in NLAs
// CHECK-LABEL: firrtl.circuit "SkipInstancesWithNLAs"
firrtl.circuit "SkipInstancesWithNLAs" {
  // NLA that goes through the instance we want to preserve
  hw.hierpath private @nla [@SkipInstancesWithNLAs::@sym, @Child]

  // CHECK: firrtl.module @SkipInstancesWithNLAs
  firrtl.module @SkipInstancesWithNLAs() {
    // CHECK-NEXT: firrtl.instance with_nla
    firrtl.instance with_nla sym @sym @Child()
    // CHECK-NOT: firrtl.instance without_nla
    firrtl.instance without_nla @Child()
  }

  firrtl.module private @Child() {}
}

// Test that EagerInliner does not inline instances if that leads to inner
// symbol collisions.
// CHECK-LABEL: firrtl.circuit "InnerSymCollisions"
firrtl.circuit "InnerSymCollisions" {
  // CHECK: firrtl.module @InnerSymCollisions
  firrtl.module @InnerSymCollisions() {
    // Cannot inline child1 because inner symbol @foo already defined.
    // CHECK-NEXT: firrtl.instance child1
    firrtl.instance child1 @ChildWithSymFoo()

    // Can inline child2 because inner symbol @bar is new.
    // CHECK-NOT: firrtl.instance child2
    firrtl.instance child2 @ChildWithSymBar()

    // Define a local @foo inner symbol.
    firrtl.instance ext sym @foo @Ext()
  }

  firrtl.module private @ChildWithSymFoo() {
    firrtl.instance ext sym @foo @Ext()
  }

  // CHECK-NOT: firrtl.module @ChildWithSymBar
  firrtl.module private @ChildWithSymBar() {
    firrtl.instance ext sym @bar @Ext()
  }

  firrtl.extmodule private @Ext()
}
