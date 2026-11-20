// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-grand-central{companion-mode=drop}))' -split-input-file %s | FileCheck %s

// A dropped companion is erased together with the modules below it.
// Paths rooted in an erased module are erased.
// https://github.com/llvm/circt/issues/11571

// CHECK-LABEL: firrtl.circuit "Chain"
// CHECK-NOT:     hw.hierpath
// CHECK-NOT:     firrtl.module private
// CHECK:         firrtl.module @Chain()
// CHECK-NEXT:    }
firrtl.circuit "Chain" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Interface", elements = [], id = 0 : i64, name = "View"},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct", filename = "gct/bindings.sv"}]} {
  hw.hierpath private @nla [@Companion::@a, @A::@b, @B]
  firrtl.module private @G() {}
  firrtl.module private @F() {
    firrtl.instance g @G()
  }
  firrtl.module private @E() {
    firrtl.instance f @F()
  }
  firrtl.module private @D() {
    firrtl.instance e @E()
  }
  firrtl.module private @C() {
    firrtl.instance d @D()
  }
  firrtl.module private @B() {
    %w = firrtl.wire {annotations = [{circt.nonlocal = @nla, class = "circt.test"}]} : !firrtl.uint<1>
    firrtl.instance c @C()
  }
  firrtl.module private @A() {
    firrtl.instance b sym @b @B()
  }
  firrtl.module private @Companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
       id = 0 : i64, name = "View"}]} {
    firrtl.instance a sym @a @A()
  }
  firrtl.module @Chain() {
    firrtl.instance companion @Companion()
  }
}
