// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-inject-dut-hier))' -split-input-file %s | FileCheck %s

// CHECK-LABEL: firrtl.circuit "Top"
// CHECK-SAME:    {class = "sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation", name = "Foo"}
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

firrtl.circuit "Top" attributes {
    annotations = [{class = "sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation", name = "Foo", moveDut = true}]
  } {
  // CHECK:      firrtl.module private @DUT()
  // CHECK-SAME:   class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
  //
  // CHECK:      firrtl.module private @Foo
  // CHECK-NEXT:   firrtl.instance DUT {{.+}} @DUT()
  // CHECK-NEXT: }
  firrtl.module private @DUT() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {}

  // CHECK:      firrtl.module @Top
  // CHECK-NEXT:   firrtl.instance dut @Foo
  firrtl.module @Top() {
    firrtl.instance dut @DUT()
  }
}

// -----

// Test renaming when `moveDut=false`.  (This is the default behavior and
// doesn't need to be explicitly specified in the annotation.)
//
// CHECK-LABEL: firrtl.circuit "NLARenamingNewNLAs"
firrtl.circuit "NLARenamingNewNLAs" attributes {
    annotations = [{class = "sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation", name = "Foo"}]
  } {
  // An NLA that is rooted at the DUT moves to the wrapper.
  //
  // CHECK:      hw.hierpath private @nla_DUTRoot [@Foo::@sub, @Sub]
  // CHECK:      hw.hierpath private @nla_DUTRootRef [@Foo::@sub, @Sub::@a]
  hw.hierpath private @nla_DUTRoot [@DUT::@sub, @Sub]
  hw.hierpath private @nla_DUTRootRef [@DUT::@sub, @Sub::@a]

  // NLAs that end at the DUT or a DUT port are unmodified.  These should not be
  // cloned unless they have users.
  //
  // CHECK-NEXT: hw.hierpath private @nla_DUTLeafModule[[_:.+]] [@NLARenamingNewNLAs::@dut, @DUT]
  // CHECK-NEXT: hw.hierpath private @nla_DUTLeafModule [@NLARenamingNewNLAs::@dut, @DUT::@Foo, @Foo]
  // CHECK-NEXT: hw.hierpath private @[[nla_DUTLeafPort_clone:.+]] [@NLARenamingNewNLAs::@dut, @DUT]
  // CHECK-NEXT: hw.hierpath private @nla_DUTLeafPort [@NLARenamingNewNLAs::@dut, @DUT::@Foo, @Foo]
  hw.hierpath private @nla_DUTLeafModule [@NLARenamingNewNLAs::@dut, @DUT]
  hw.hierpath private @nla_DUTLeafPort [@NLARenamingNewNLAs::@dut, @DUT]

  // NLAs that end at the DUT are moved to a cloned path.  NLAs that end inside
  // the DUT keep the old path symbol which gets the added hierarchy.
  //
  // CHECK-NEXT: hw.hierpath private @nla_DUTLeafWire [@NLARenamingNewNLAs::@dut, @DUT::@[[inst_sym:.+]], @Foo]
  hw.hierpath private @nla_DUTLeafWire [@NLARenamingNewNLAs::@dut, @DUT]

  // An NLA that passes through the DUT gets an extra level of hierarchy.
  //
  // CHECK-NEXT: hw.hierpath private @nla_DUTPassthrough [@NLARenamingNewNLAs::@dut, @DUT::@[[inst_sym]], @Foo::@sub, @Sub]
  hw.hierpath private @nla_DUTPassthrough [@NLARenamingNewNLAs::@dut, @DUT::@sub, @Sub]
  firrtl.module private @Sub() attributes {annotations = [{circt.nonlocal = @nla_DUTPassthrough, class = "nla_DUTPassthrough"}]} {
    %a = firrtl.wire sym @a : !firrtl.uint<1>
  }

  // CHECK:      firrtl.module private @Foo
  // CHECK-NEXT:   %w = firrtl.wire
  // CHECK-SAME:     {annotations = [{circt.nonlocal = @nla_DUTLeafWire, class = "nla_DUT_LeafWire"}]}

  // CHECK:      firrtl.module private @DUT
  // CHECK-SAME:   in %in{{.+}} [{circt.nonlocal = @[[nla_DUTLeafPort_clone]], class = "nla_DUTLeafPort"}]
  // CHECK-NEXT:    firrtl.instance Foo sym @[[inst_sym]]
  firrtl.module private @DUT(
    in %in: !firrtl.uint<1> [{circt.nonlocal = @nla_DUTLeafPort, class = "nla_DUTLeafPort"}]
  ) attributes {
    annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"},
      {circt.nonlocal = @nla_DUTLeafModule, class = "nla_DUTLeafModule"}]}
  {
    %w = firrtl.wire {
      annotations = [
        {circt.nonlocal = @nla_DUTLeafWire, class = "nla_DUT_LeafWire"}]
    } : !firrtl.uint<1>
    firrtl.instance sub sym @sub @Sub()
  }
  firrtl.module @NLARenamingNewNLAs() {
    %dut_in = firrtl.instance dut sym @dut @DUT(in in: !firrtl.uint<1>)
  }
}

// -----

// Test renaming when `moveDut=true`.  This test is a copy of the one above it.
//
// CHECK-LABEL: firrtl.circuit "NLARenamingMoveDutTrue"
firrtl.circuit "NLARenamingMoveDutTrue" attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation",
        name = "Foo",
        moveDut = true
      }
    ]
  } {
  // An NLA that is rooted at the DUT moves to the WRAPPER.  However, because
  // the WRAPPER is renamed to the name of the DUT, these NLAs do not change.
  //
  // CHECK:      hw.hierpath private @nla_DUTRoot [@DUT::@sub, @Sub]
  // CHECK:      hw.hierpath private @nla_DUTRootRef [@DUT::@sub, @Sub::@a]
  hw.hierpath private @nla_DUTRoot [@DUT::@sub, @Sub]
  hw.hierpath private @nla_DUTRootRef [@DUT::@sub, @Sub::@a]

  // NLAs that end at the DUT or a DUT port are changed to end on the WRAPPER.
  // The WRAPPER takes the original DUT name.
  //
  // CHECK-NEXT: hw.hierpath private @nla_DUTLeafModule [@NLARenamingMoveDutTrue::@dut, @Foo::@DUT, @DUT]
  // CHECK-NEXT: hw.hierpath private @nla_DUTLeafPort [@NLARenamingMoveDutTrue::@dut, @Foo::@DUT, @DUT]
  hw.hierpath private @nla_DUTLeafModule [@NLARenamingMoveDutTrue::@dut, @DUT]
  hw.hierpath private @nla_DUTLeafPort [@NLARenamingMoveDutTrue::@dut, @DUT]

  // NLAs that end inside the DUT are moved to end inside the WRAPPER.  The
  // WRAPPER takes the original DUT name.
  //
  // CHECK-NEXT: hw.hierpath private @nla_DUTLeafWire [@NLARenamingMoveDutTrue::@dut, @Foo::@[[inst_sym:.+]], @DUT]
  hw.hierpath private @nla_DUTLeafWire [@NLARenamingMoveDutTrue::@dut, @DUT]

  // An NLA that passes through the DUT gets an extra level of hierarchy that
  // includes the WRAPPER.
  //
  // CHECK-NEXT: hw.hierpath private @nla_DUTPassthrough [@NLARenamingMoveDutTrue::@dut, @Foo::@[[inst_sym]], @DUT::@sub, @Sub]
  hw.hierpath private @nla_DUTPassthrough [@NLARenamingMoveDutTrue::@dut, @DUT::@sub, @Sub]
  firrtl.module private @Sub() attributes {annotations = [{circt.nonlocal = @nla_DUTPassthrough, class = "nla_DUTPassthrough"}]} {
    %a = firrtl.wire sym @a : !firrtl.uint<1>
  }

  // CHECK:      firrtl.module private @DUT
  // CHECK-SAME:   in %in{{.+}} [{circt.nonlocal = @nla_DUTLeafPort, class = "nla_DUTLeafPort"}]
  // CHECK-NEXT:   %w = firrtl.wire
  // CHECK-SAME:     {annotations = [{circt.nonlocal = @nla_DUTLeafWire, class = "nla_DUT_LeafWire"}]}

  // CHECK:      firrtl.module private @Foo
  // CHECK-NOT:    annotation
  // CHECK-NEXT:   firrtl.instance DUT sym @[[inst_sym]]
  firrtl.module private @DUT(
    in %in: !firrtl.uint<1> [{circt.nonlocal = @nla_DUTLeafPort, class = "nla_DUTLeafPort"}]
  ) attributes {
    annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"},
      {circt.nonlocal = @nla_DUTLeafModule, class = "nla_DUTLeafModule"}]}
  {
    %w = firrtl.wire {
      annotations = [
        {circt.nonlocal = @nla_DUTLeafWire, class = "nla_DUT_LeafWire"}]
    } : !firrtl.uint<1>
    firrtl.instance sub sym @sub @Sub()
  }
  firrtl.module @NLARenamingMoveDutTrue() {
    %dut_in = firrtl.instance dut sym @dut @DUT(in in: !firrtl.uint<1>)

    firrtl.path reference distinct[0]<>
  }
}

// -----

// Test that an object model path is updated to point at the new DUT when in
// `moveDut=true` mode.  This test is redundant with "NLARenamingNLA
//
// CHECK-LABEL: firrtl.circuit "ObjectModelDUT"
firrtl.circuit "ObjectModelDUT" attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation",
        name = "Foo",
        moveDut = true
      }
    ]
  } {

  // CHECK: hw.hierpath private @nla [@ObjectModelDUT::@dut, @Foo::@[[wrapperSym:.+]], @DUT]
  hw.hierpath private @nla [@ObjectModelDUT::@dut, @DUT]

  // CHECK:      firrtl.module private @DUT()
  // CHECK-SAME:   {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}
  // CHECK-SAME:   {circt.nonlocal = @nla, class = "circt.tracker", id = distinct[0]<>}

  // CHECK:     firrtl.module private @Foo()
  // CHECK-NOT:   "sifive.enterprise.firrtl.MarkDUTAnnotation"
  // CHECK-NOT:   circt.nonlocal
  // CHECK-NOT: firrtl.module
  // CHECK:       firrtl.instance {{.*}} sym @[[wrapperSym]] @DUT()
  firrtl.module private @DUT() attributes {
    annotations = [
      {
        class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
      },
      {
        circt.nonlocal = @nla,
        class = "circt.tracker",
        id = distinct[0]<>
      }
    ]
  } {}

  // CHECK:     firrtl.module @ObjectModelDUT()
  // CHECK-NOT: firrtl.module
  // CHECK:       firrtl.path reference distinct[0]<>
  firrtl.module @ObjectModelDUT() {
    firrtl.instance dut sym @dut @DUT()
    firrtl.path reference distinct[0]<>
  }

}

// -----

// CHECK-LABEL: firrtl.circuit "Refs"
firrtl.circuit "Refs" attributes {
    annotations = [{class = "sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation", name = "Foo"}]
  } {

  firrtl.module private @DUT(
    in %in: !firrtl.uint<1>, out %out: !firrtl.ref<uint<1>>
  ) attributes {
    annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}
    ]}
  {
    %ref = firrtl.ref.send %in : !firrtl.uint<1>
    firrtl.ref.define %out, %ref : !firrtl.ref<uint<1>>
  }
  firrtl.module @Refs() {
    %dut_in, %dut_tap = firrtl.instance dut sym @dut @DUT(in in: !firrtl.uint<1>, out out: !firrtl.ref<uint<1>>)
  }
}

// -----
// https://github.com/llvm/circt/issues/8552
// Check rwprobes are updated.

// CHECK-LABEL: firrtl.circuit "RWProbe"
firrtl.circuit "RWProbe" attributes {
    annotations = [{class = "sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation", name = "Foo"}]
  } {

  firrtl.module private @DUT(
    in %in: !firrtl.uint<1>, out %out: !firrtl.rwprobe<uint<1>>
  ) attributes {
    annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}
    ]}
  {
    %w = firrtl.wire sym @sym : !firrtl.uint<1>
    // CHECK: ref.rwprobe <@Foo::@sym>
    %rwp = firrtl.ref.rwprobe <@DUT::@sym> : !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %out, %rwp : !firrtl.rwprobe<uint<1>>
  }
  firrtl.module @RWProbe() {
    %dut_in, %dut_tap = firrtl.instance dut sym @dut @DUT(in in: !firrtl.uint<1>, out out: !firrtl.rwprobe<uint<1>>)
  }
}

// -----

// CHECK-LABEL: firrtl.circuit "Properties"
firrtl.circuit "Properties" attributes {
    annotations = [{class = "sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation", name = "Foo"}]
  } {

  firrtl.module private @DUT(
    in %in: !firrtl.integer, out %out: !firrtl.integer
  ) attributes {
    annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}
    ]}
  {
    // CHECK: [[IN:%.+]], [[OUT:%.+]] = firrtl.instance Foo
    // CHECK: firrtl.propassign [[IN]], %in
    // CHECK: firrtl.propassign %out, [[OUT]]
    firrtl.propassign %out, %in : !firrtl.integer
  }
  firrtl.module @Properties() {
    %dut_in, %dut_out = firrtl.instance dut sym @dut @DUT(in in: !firrtl.integer, out out: !firrtl.integer)
  }
}

// -----

firrtl.circuit "PublicMoveDutFalse" attributes {
  annotations = [
    {
      class = "sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation",
      name = "Foo",
      moveDut = false
    }
  ]
} {
  // CHECK: firrtl.module private @Foo()
  // CHECK: firrtl.module @DUT()
  firrtl.module @DUT() attributes {
    annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}
    ]
  } {}
  firrtl.module @PublicMoveDutFalse() {
    firrtl.instance dut @DUT()
  }
}

// -----

firrtl.circuit "PublicMoveDutTrue" attributes {
  annotations = [
    {
      class = "sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation",
      name = "Foo",
      moveDut = true
    }
  ]
} {
  // CHECK: firrtl.module @DUT()
  // CHECK: firrtl.module private @Foo()
  firrtl.module @DUT() attributes {
    annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}
    ]
  } {}
  firrtl.module @PublicMoveDutTrue() {
    firrtl.instance dut @DUT()
  }
}
