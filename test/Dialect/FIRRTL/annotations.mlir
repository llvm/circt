// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-lower-annotations)' -split-input-file %s |FileCheck %s
// XFAIL: *

// circt.test copies the annotation to the target
// circt.testNT puts the targetless annotation on the circuit


// A non-local annotation should work.

// CHECK-LABEL: firrtl.circuit "FooNL"
// CHECK: firrtl.module @BarNL
// CHECK: firrtl.wire {annotations = [{circt.nonlocal = @"~FooNL|FooNL/baz:BazNL/bar:BarNL>w_NA_0", class = "circt.test", nl = "nl"}]}
// CHECK: firrtl.instance @BarNL {annotations = [{circt.nonlocal = @"~FooNL|FooNL/baz:BazNL/bar:BarNL>w_NA_0", class = "circt.nonlocal"}], name = "bar"}
// CHECK: firrtl.instance @BazNL {annotations = [{circt.nonlocal = @"~FooNL|FooNL/baz:BazNL/bar:BarNL>w_NA_0", class = "circt.nonlocal"}], name = "baz"
// CHECK: firrtl.nla @"~FooNL|FooNL/baz:BazNL/bar:BarNL>w_NA_0"
firrtl.circuit "FooNL"  attributes {annotations = [{class = "circt.test", nl = "nl", target = "~FooNL|FooNL/baz:BazNL/bar:BarNL>w"}]}  {
  firrtl.module @BarNL() {
    %w = firrtl.wire  : !firrtl.uint
    firrtl.skip
  }
  firrtl.module @BazNL() {
    firrtl.instance @BarNL  {name = "bar"}
  }
  firrtl.module @FooNL() {
    firrtl.instance @BazNL  {name = "baz"}
  }
}


