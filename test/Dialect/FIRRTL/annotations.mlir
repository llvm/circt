// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-lower-annotations)' -split-input-file %s |FileCheck %s

// circt.test copies the annotation to the target
// circt.testNT puts the targetless annotation on the circuit


// A non-local annotation should work.

// CHECK-LABEL: firrtl.circuit "FooNL"
// CHECK: firrtl.nla @nla [@FooNL, @BazNL, @BarNL] ["baz", "bar", "w"]
// CHECK: firrtl.module @BarNL
// CHECK: firrtl.wire {annotations = [{circt.nonlocal = @nla, class = "circt.test", nl = "nl"}]}
// CHECK: firrtl.instance @BarNL {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}], name = "bar"}
// CHECK: firrtl.instance @BazNL {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}], name = "baz"
firrtl.circuit "FooNL"  attributes {annotations = [
  {class = "circt.test", nl = "nl", target = "~FooNL|FooNL/baz:BazNL/bar:BarNL>w"},
  {class = "circt.test", nl = "nl2", target = "~FooNL|FooNL/baz:BazNL/bar:BarNL>w2.b[2]"}
  ]}  {
  firrtl.module @BarNL() {
    %w = firrtl.wire  : !firrtl.uint
    %w2 = firrtl.wire  : !firrtl.bundle<a: uint, b: vector<uint, 4>>
    firrtl.skip
  }
  firrtl.module @BazNL() {
    firrtl.instance @BarNL  {name = "bar"}
  }
  firrtl.module @FooNL() {
    firrtl.instance @BazNL  {name = "baz"}
  }
}


