// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-inject-dut-hier)' -split-input-file %s | FileCheck %s

// CHECK-LABEL: firrtl.circuit "Top"
firrtl.circuit "Top" attributes {
    annotations = [{class = "sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation", name = "Foo"}]
  } {
  // CHECK:      firrtl.module private @Foo()
  //
  // CHECK:      firrtl.module private @DUT
  // CHECK-SAME:   class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
  //
  // CHECK-NEXT:   firrtl.instance Foo {{.+}} @Foo()
  // CHECK-NEXT: }
  firrtl.module private @DUT() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {}

  // CHECK:      firrtl.module @Top
  // CHECK-NEXT:   firrtl.instance dut @DUT
  firrtl.module @Top() {
    firrtl.instance dut @DUT()
  }
}

// -----

// CHECK-LABEL: firrtl.circuit "NLARenaming"
firrtl.circuit "NLARenaming" attributes {
    annotations = [{class = "sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation", name = "Foo"}]
  } {
  // An NLA that is rooted at the DUT moves to the wrapper.
  //
  // CHECK:      firrtl.nla @nla_DUTRoot [#hw.innerNameRef<@Foo::@sub>, #hw.innerNameRef<@Sub::@a>]
  firrtl.nla @nla_DUTRoot [#hw.innerNameRef<@DUT::@sub>, #hw.innerNameRef<@Sub::@a>]

  // NLAs that end at the DUT or a DUT port are unmodified.
  //
  // CHECK-NEXT: firrtl.nla @nla_DUTLeafModule [#hw.innerNameRef<@NLARenaming::@dut>, @DUT]
  // CHECK-NEXT: firrtl.nla @nla_DUTLeafPort [#hw.innerNameRef<@NLARenaming::@dut>, #hw.innerNameRef<@DUT::@in>]
  firrtl.nla @nla_DUTLeafModule [#hw.innerNameRef<@NLARenaming::@dut>, @DUT]
  firrtl.nla @nla_DUTLeafPort [#hw.innerNameRef<@NLARenaming::@dut>, #hw.innerNameRef<@DUT::@in>]

  // NLAs that end inside the DUT get an extra level of hierarchy.
  //
  // CHECK-NEXT: firrtl.nla @nla_DUTLeafWire [#hw.innerNameRef<@NLARenaming::@dut>, #hw.innerNameRef<@DUT::@[[inst_sym:.+]]>, #hw.innerNameRef<@Foo::@w>]
  firrtl.nla @nla_DUTLeafWire [#hw.innerNameRef<@NLARenaming::@dut>, #hw.innerNameRef<@DUT::@w>]

  // An NLA that passes through the DUT gets an extra level of hierarchy.
  //
  // CHECK-NEXT: firrtl.nla @nla_DUTPassthrough [#hw.innerNameRef<@NLARenaming::@dut>, #hw.innerNameRef<@DUT::@[[inst_sym:.+]]>, #hw.innerNameRef<@Foo::@sub>, @Sub]
  firrtl.nla @nla_DUTPassthrough [#hw.innerNameRef<@NLARenaming::@dut>, #hw.innerNameRef<@DUT::@sub>, @Sub]
  firrtl.module private @Sub() attributes {annotations = [{circt.nonlocal = @nla_DUTPassthrough, class = "nla_DUTPassthrough"}]} {
    %a = firrtl.wire sym @a : !firrtl.uint<1>
  }

  // CHECK:     firrtl.module private @Foo
  // CHECK:     firrtl.module private @DUT
  // CHECK-NEXT   firrtl.instance Foo sym @[[inst_sym]]
  firrtl.module private @DUT(
    in %in: !firrtl.uint<1> sym @in [{circt.nonlocal = @nla_DUTLeafPort, class = "nla_DUTLeafPort"}]
  ) attributes {
    annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"},
      {circt.nonlocal = @nla_DUTLeafModule, class = "nla_DUTLeafModule"}]}
  {
    %w = firrtl.wire sym @w {
      annotations = [
        {circt.nonlocal = @nla_DUTPassthrough, class = "nla_DUT_LeafWire"}]
    } : !firrtl.uint<1>
    firrtl.instance sub sym @sub {
      annotations = [
        {circt.nonlocal = @nla_DUTRoot, class = "circt.nonlocal"},
        {circt.nonlocal = @nla_DUTPassthrough, class = "circt.nonlocal"}]} @Sub()
  }
  firrtl.module @NLARenaming() {
    %dut_in = firrtl.instance dut sym @dut {
      annotations = [
        {circt.nonlocal = @nla_DUTLeafModule, class = "circt.nonlocal"},
        {circt.nonlocal = @nla_DUTLeafPort, class = "circt.nonlocal"},
        {circt.nonlocal = @nla_DUTLeafWire, class = "circt.nonlocal"},
        {circt.nonlocal = @nla_DUTPassthrough, class = "circt.nonlocal"}]} @DUT(in in: !firrtl.uint<1>)
  }
}
