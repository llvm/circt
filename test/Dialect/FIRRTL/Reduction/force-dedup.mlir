// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg true --include firrtl-force-dedup | FileCheck %s

// Test that the MustDedup reducer can consolidate modules in a dedup group
// by replacing all module names with a single canonical module name.

firrtl.circuit "MustDedup" attributes {annotations = [{
    class = "firrtl.transforms.MustDeduplicateAnnotation",
    modules = ["~MustDedup|Simple0", "~MustDedup|Simple1", "~MustDedup|Simple2"]
  }]} {

  // CHECK: hw.hierpath private @nla [@MustDedup::@simple1, @Simple0]
  hw.hierpath private @nla [@MustDedup::@simple1, @Simple1]

  // CHECK: firrtl.module private @Simple0
  firrtl.module private @Simple0() {
    %w = firrtl.wire : !firrtl.uint<1>
  }

  // CHECK-NOT: firrtl.module private @Simple1
  firrtl.module private @Simple1() {
    %w = firrtl.wire : !firrtl.uint<1>
  }

  // CHECK-NOT: firrtl.module private @Simple2
  firrtl.module private @Simple2() {
    %w = firrtl.wire : !firrtl.uint<1>
  }

  // CHECK: firrtl.module @MustDedup
  firrtl.module @MustDedup() {
    // CHECK: firrtl.instance simple0 @Simple0
    firrtl.instance simple0 @Simple0()
    // CHECK: firrtl.instance simple1 sym @simple1 {annotations = [{circt.nonlocal = @nla, class = "test"}]} @Simple0()
    firrtl.instance simple1 sym @simple1 {annotations = [{circt.nonlocal = @nla, class = "test"}]} @Simple1()
    // CHECK: firrtl.instance simple2 @Simple0
    firrtl.instance simple2 @Simple2()
  }
}

// Test with multiple NLAs referencing different modules in the dedup group
firrtl.circuit "MultiNLA" attributes {annotations = [{
    class = "firrtl.transforms.MustDeduplicateAnnotation",
    modules = ["~MultiNLA|ModA", "~MultiNLA|ModB"]
  }]} {

  // CHECK: hw.hierpath private @nla1 [@MultiNLA::@instA, @ModA]
  hw.hierpath private @nla1 [@MultiNLA::@instA, @ModA]
  // CHECK: hw.hierpath private @nla2 [@MultiNLA::@instB, @ModA]
  hw.hierpath private @nla2 [@MultiNLA::@instB, @ModB]

  // CHECK: firrtl.module private @ModA
  firrtl.module private @ModA() {
    %w = firrtl.wire : !firrtl.uint<1>
  }

  // CHECK-NOT: firrtl.module private @ModB
  firrtl.module private @ModB() {
    %w = firrtl.wire : !firrtl.uint<1>
  }

  // CHECK: firrtl.module @MultiNLA
  firrtl.module @MultiNLA() {
    // CHECK: firrtl.instance instA sym @instA {annotations = [{circt.nonlocal = @nla1, class = "test1"}]} @ModA()
    firrtl.instance instA sym @instA {annotations = [{circt.nonlocal = @nla1, class = "test1"}]} @ModA()
    // CHECK: firrtl.instance instB sym @instB {annotations = [{circt.nonlocal = @nla2, class = "test2"}]} @ModA()
    firrtl.instance instB sym @instB {annotations = [{circt.nonlocal = @nla2, class = "test2"}]} @ModB()
  }
}

// Test with multiple NLAs referencing different nested modules in the dedup group
firrtl.circuit "MultiSubNLA" attributes {annotations = [{
    class = "firrtl.transforms.MustDeduplicateAnnotation",
    modules = ["~MultiSubNLA|ModA", "~MultiSubNLA|ModB"]
  }]} {

  // CHECK: hw.hierpath private @nla1 [@MultiSubNLA::@instA, @ModA::@sub, @SubModA::@wire]
  hw.hierpath private @nla1 [@MultiSubNLA::@instA, @ModA::@sub, @SubModA::@wire]
  // CHECK: hw.hierpath private @nla2 [@MultiSubNLA::@instB, @ModA::@sub, @SubModA::@wire]
  hw.hierpath private @nla2 [@MultiSubNLA::@instB, @ModB::@sub, @SubModB::@wire]

  // CHECK: firrtl.module private @ModA
  firrtl.module private @ModA() {
    firrtl.instance sub sym @sub @SubModA()
  }

  // CHECK-NOT: firrtl.module private @ModB
  firrtl.module private @ModB() {
    firrtl.instance sub sym @sub @SubModB()
  }

  // CHECK: firrtl.module private @SubModA
  firrtl.module private @SubModA() {
    %w = firrtl.wire sym @wire {annotations = [{circt.nonlocal = @nla1, class = "test1"}]} : !firrtl.uint<1>
  }

  // CHECK: firrtl.module private @SubModB
  firrtl.module private @SubModB() {
    %w = firrtl.wire sym @wire {annotations = [{circt.nonlocal = @nla2, class = "test2"}]} : !firrtl.uint<1>
  }

  // CHECK: firrtl.module @MultiSubNLA
  firrtl.module @MultiSubNLA() {
    // CHECK: firrtl.instance instA sym @instA @ModA()
    firrtl.instance instA sym @instA @ModA()
    // CHECK: firrtl.instance instB sym @instB @ModA()
    firrtl.instance instB sym @instB @ModB()
  }
}
