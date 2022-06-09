// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-dedup)' %s | FileCheck %s

// CHECK-LABEL: firrtl.circuit "Empty"
firrtl.circuit "Empty" {
  // CHECK: firrtl.module @Empty0
  firrtl.module @Empty0(in %i0: !firrtl.uint<1>) { }
  // CHECK-NOT: firrtl.module @Empty1
  firrtl.module @Empty1(in %i1: !firrtl.uint<1>) { }
  // CHECK-NOT: firrtl.module @Empty2
  firrtl.module @Empty2(in %i2: !firrtl.uint<1>) { }
  firrtl.module @Empty() {
    // CHECK: %e0_i0 = firrtl.instance e0  @Empty0
    // CHECK: %e1_i0 = firrtl.instance e1  @Empty0
    // CHECK: %e2_i0 = firrtl.instance e2  @Empty0
    %e0_i0 = firrtl.instance e0 @Empty0(in i0: !firrtl.uint<1>)
    %e1_i1 = firrtl.instance e1 @Empty1(in i1: !firrtl.uint<1>)
    %e2_i2 = firrtl.instance e2 @Empty2(in i2: !firrtl.uint<1>)
  }
}


// CHECK-LABEL: firrtl.circuit "Simple"
firrtl.circuit "Simple" {
  // CHECK: firrtl.module @Simple0
  firrtl.module @Simple0() {
    %a = firrtl.wire: !firrtl.bundle<a: uint<1>>
  }
  // CHECK-NOT: firrtl.module @Simple1
  firrtl.module @Simple1() {
    %b = firrtl.wire: !firrtl.bundle<b: uint<1>>
  }
  firrtl.module @Simple() {
    // CHECK: firrtl.instance simple0 @Simple0
    // CHECK: firrtl.instance simple1 @Simple0
    firrtl.instance simple0 @Simple0()
    firrtl.instance simple1 @Simple1()
  }
}


// Should pick a valid symbol when a wire has no name.
// CHECK-LABEL: firrtl.circuit "Top"
firrtl.circuit "Top"  {
  firrtl.module @Top() {
    %a1_x = firrtl.instance a1  @A(out x: !firrtl.uint<1>)
    %a2_x = firrtl.instance a2  @A_(out x: !firrtl.uint<1>)
  }
  firrtl.module @A(out %x: !firrtl.uint<1>) {
    // CHECK: %0 = firrtl.wire sym @inner_sym
    %0 = firrtl.wire  {annotations = [{class = "hello"}]} : !firrtl.uint<1>
  }
  firrtl.module @A_(out %x: !firrtl.uint<1>) {
    %0 = firrtl.wire  : !firrtl.uint<1>
  }
}

// CHECK-LABEL: firrtl.circuit "PrimOps"
firrtl.circuit "PrimOps" {
  // CHECK: firrtl.module @PrimOps0
  firrtl.module @PrimOps0(in %a: !firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>) {
    %a_a = firrtl.subfield %a(0): (!firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>) -> !firrtl.uint<2>
    %a_b = firrtl.subfield %a(1): (!firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>) -> !firrtl.uint<2>
    %a_c = firrtl.subfield %a(2): (!firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>) -> !firrtl.uint<2>
    %0 = firrtl.xor %a_a, %a_b: (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
    firrtl.connect %a_c, %a_b: !firrtl.uint<2>, !firrtl.uint<2>
  }
  // CHECK-NOT: firrtl.module @PrimOps1
  firrtl.module @PrimOps1(in %b: !firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>) {
    %b_a = firrtl.subfield %b(0): (!firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>) -> !firrtl.uint<2>
    %b_b = firrtl.subfield %b(1): (!firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>) -> !firrtl.uint<2>
    %b_c = firrtl.subfield %b(2): (!firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>) -> !firrtl.uint<2>
    %0 = firrtl.xor %b_a, %b_b: (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
    firrtl.connect %b_c, %b_b: !firrtl.uint<2>, !firrtl.uint<2>
  }
  firrtl.module @PrimOps() {
    // CHECK: firrtl.instance primops0 @PrimOps0
    // CHECK: firrtl.instance primops1 @PrimOps0
    %primops0 = firrtl.instance primops0 @PrimOps0(in a: !firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>)
    %primops1 = firrtl.instance primops1 @PrimOps1(in b: !firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>)
  }
}

// Check that when operations are recursively merged.
// CHECK-LABEL: firrtl.circuit "WhenOps"
firrtl.circuit "WhenOps" {
  // CHECK: firrtl.module @WhenOps0
  firrtl.module @WhenOps0(in %p : !firrtl.uint<1>) {
    // CHECK: firrtl.when %p {
    // CHECK:  %w = firrtl.wire : !firrtl.uint<8>
    // CHECK: }
    firrtl.when %p {
      %w = firrtl.wire : !firrtl.uint<8>
    }
  }
  // CHECK-NOT: firrtl.module @PrimOps1
  firrtl.module @WhenOps1(in %p : !firrtl.uint<1>) {
    firrtl.when %p {
      %w = firrtl.wire : !firrtl.uint<8>
    }
  }
  firrtl.module @WhenOps() {
    // CHECK: firrtl.instance whenops0 @WhenOps0
    // CHECK: firrtl.instance whenops1 @WhenOps0
    %whenops0 = firrtl.instance whenops0 @WhenOps0(in p : !firrtl.uint<1>)
    %whenops1 = firrtl.instance whenops1 @WhenOps1(in p : !firrtl.uint<1>)
  }
}

// CHECK-LABEL: firrtl.circuit "Annotations"
firrtl.circuit "Annotations" {
  // CHECK: firrtl.hierpath [[NLA3:@nla.*]] [@Annotations::@annotations1, @Annotations0]
  // CHECK: firrtl.hierpath [[NLA2:@nla.*]] [@Annotations::@annotations0, @Annotations0::@e]
  // CHECK: firrtl.hierpath [[NLA1:@nla.*]] [@Annotations::@annotations0, @Annotations0::@c]
  // CHECK: firrtl.hierpath [[NLA0:@nla.*]] [@Annotations::@annotations1, @Annotations0::@b]
  // CHECK: firrtl.hierpath @annos_nla0 [@Annotations::@annotations0, @Annotations0::@d]
  // CHECK: firrtl.hierpath @annos_nla1 [@Annotations::@annotations1, @Annotations0::@d]
  firrtl.hierpath @annos_nla0 [@Annotations::@annotations0, @Annotations0::@d]
  firrtl.hierpath @annos_nla1 [@Annotations::@annotations1, @Annotations1::@i]

  // CHECK: firrtl.module @Annotations0() attributes {annotations = [{circt.nonlocal = [[NLA3]], class = "one"}]}
  firrtl.module @Annotations0() {
    // Same annotation on both ops should stay local.
    // CHECK: %a = firrtl.wire  {annotations = [{class = "both"}]}
    %a = firrtl.wire {annotations = [{class = "both"}]} : !firrtl.uint<1>

    // Annotation from other module becomes non-local.
    // CHECK: %b = firrtl.wire sym @b  {annotations = [{circt.nonlocal = [[NLA0]], class = "one"}]}
    %b = firrtl.wire : !firrtl.uint<1>

    // Annotation from this module becomes non-local.
    // CHECK: %c = firrtl.wire sym @c  {annotations = [{circt.nonlocal = [[NLA1]], class = "one"}]}
    %c = firrtl.wire {annotations = [{class = "one"}]} : !firrtl.uint<1>

    // Two non-local annotations are unchanged, as they have enough context in the NLA already.
    // CHECK: %d = firrtl.wire sym @d  {annotations = [{circt.nonlocal = @annos_nla1, class = "NonLocal"}, {circt.nonlocal = @annos_nla0, class = "NonLocal"}]}
    %d = firrtl.wire sym @d {annotations = [{circt.nonlocal = @annos_nla0, class = "NonLocal"}]} : !firrtl.uint<1>

    // Subannotations should be handled correctly.
    // CHECK: %e = firrtl.wire sym @e  {annotations = [#firrtl.subAnno<fieldID = 1, {circt.nonlocal = [[NLA2]], class = "subanno"}>]}
    %e = firrtl.wire {annotations = [#firrtl.subAnno<fieldID = 1, {class = "subanno"}>]} : !firrtl.bundle<a: uint<1>>
  }
  // CHECK-NOT: firrtl.module @Annotations1
  firrtl.module @Annotations1() attributes {annotations = [{class = "one"}]} {
    %f = firrtl.wire {annotations = [{class = "both"}]} : !firrtl.uint<1>
    %g = firrtl.wire {annotations = [{class = "one"}]} : !firrtl.uint<1>
    %h = firrtl.wire : !firrtl.uint<1>
    %i = firrtl.wire sym @i {annotations = [{circt.nonlocal = @annos_nla1, class = "NonLocal"}]} : !firrtl.uint<1>
    %j = firrtl.wire : !firrtl.bundle<a: uint<1>>
  }
  firrtl.module @Annotations() {
    // CHECK: firrtl.instance annotations0 sym @annotations0  @Annotations0()
    // CHECK: firrtl.instance annotations1 sym @annotations1  @Annotations0()
    firrtl.instance annotations0 sym @annotations0 @Annotations0()
    firrtl.instance annotations1 sym @annotations1 @Annotations1()
  }
}

// Check that module and memory port annotations are merged correctly.
// CHECK-LABEL: firrtl.circuit "PortAnnotations"
firrtl.circuit "PortAnnotations" {
  // CHECK: firrtl.hierpath [[NLA3:@nla.*]] [@PortAnnotations::@portannos0, @PortAnnotations0::@a]
  // CHECK: firrtl.hierpath [[NLA2:@nla.*]] [@PortAnnotations::@portannos1, @PortAnnotations0::@a]
  // CHECK: firrtl.hierpath [[NLA1:@nla.*]] [@PortAnnotations::@portannos0, @PortAnnotations0::@bar]
  // CHECK: firrtl.hierpath [[NLA0:@nla.*]] [@PortAnnotations::@portannos1, @PortAnnotations0::@bar]
  // CHECK: firrtl.module @PortAnnotations0(in %a: !firrtl.uint<1> sym @a [
  // CHECK-SAME: {circt.nonlocal = [[NLA2]], class = "port1"},
  // CHECK-SAME: {circt.nonlocal = [[NLA3]], class = "port0"}]) {
  firrtl.module @PortAnnotations0(in %a : !firrtl.uint<1> [{class = "port0"}]) {
    // CHECK: %bar_r = firrtl.mem sym @bar
    // CHECK-SAME: portAnnotations =
    // CHECK-SAME:  {circt.nonlocal = [[NLA0]], class = "mem1"},
    // CHECK-SAME:  {circt.nonlocal = [[NLA1]], class = "mem0"}
    %bar_r = firrtl.mem Undefined  {depth = 16 : i64, name = "bar", portAnnotations = [[{class = "mem0"}]], portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
  }
  // CHECK-NOT: firrtl.module @PortAnnotations1
  firrtl.module @PortAnnotations1(in %b : !firrtl.uint<1> [{class = "port1"}])  {
    %bar_r = firrtl.mem Undefined  {depth = 16 : i64, name = "bar", portAnnotations = [[{class = "mem1"}]], portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
  }
  // CHECK: firrtl.module @PortAnnotations
  firrtl.module @PortAnnotations() {
    %portannos0_in = firrtl.instance portannos0 @PortAnnotations0(in a: !firrtl.uint<1>)
    %portannos1_in = firrtl.instance portannos1 @PortAnnotations1(in b: !firrtl.uint<1>)
  }
}

// Non-local annotations should have their path updated and bread crumbs should
// not be turned into non-local annotations. Note that this should not create
// totally new NLAs for the annotations, it should just update the existing
// ones.
// CHECK-LABEL: firrtl.circuit "Breadcrumb"
firrtl.circuit "Breadcrumb" {
  // CHECK:  @breadcrumb_nla0 [@Breadcrumb::@breadcrumb0, @Breadcrumb0::@crumb0, @Crumb::@in]
  firrtl.hierpath @breadcrumb_nla0 [@Breadcrumb::@breadcrumb0, @Breadcrumb0::@crumb0, @Crumb::@in]
  // CHECK:  @breadcrumb_nla1 [@Breadcrumb::@breadcrumb1, @Breadcrumb0::@crumb0, @Crumb::@in]
  firrtl.hierpath @breadcrumb_nla1 [@Breadcrumb::@breadcrumb1, @Breadcrumb1::@crumb1, @Crumb::@in]
  // CHECK:  @breadcrumb_nla2 [@Breadcrumb::@breadcrumb0, @Breadcrumb0::@crumb0, @Crumb::@w]
  firrtl.hierpath @breadcrumb_nla2 [@Breadcrumb::@breadcrumb0, @Breadcrumb0::@crumb0, @Crumb::@w]
  // CHECK:  @breadcrumb_nla3 [@Breadcrumb::@breadcrumb1, @Breadcrumb0::@crumb0, @Crumb::@w]
  firrtl.hierpath @breadcrumb_nla3 [@Breadcrumb::@breadcrumb1, @Breadcrumb1::@crumb1, @Crumb::@w]
  firrtl.module @Crumb(in %in: !firrtl.uint<1> sym @in [
      {circt.nonlocal = @breadcrumb_nla0, class = "port0"},
      {circt.nonlocal = @breadcrumb_nla1, class = "port1"}]) {
    %w = firrtl.wire sym @w {annotations = [
      {circt.nonlocal = @breadcrumb_nla2, class = "wire0"},
      {circt.nonlocal = @breadcrumb_nla3, class = "wire1"}]}: !firrtl.uint<1>
  }
  // CHECK: firrtl.module @Breadcrumb0()
  firrtl.module @Breadcrumb0() {
    // CHECK: %crumb0_in = firrtl.instance crumb0 sym @crumb0
    %crumb_in = firrtl.instance crumb0 sym @crumb0 @Crumb(in in : !firrtl.uint<1>)
  }
  // CHECK-NOT: firrtl.module @Breadcrumb1()
  firrtl.module @Breadcrumb1() {
    %crumb_in = firrtl.instance crumb1 sym @crumb1 @Crumb(in in : !firrtl.uint<1>)
  }
  // CHECK: firrtl.module @Breadcrumb()
  firrtl.module @Breadcrumb() {
    firrtl.instance breadcrumb0 sym @breadcrumb0 @Breadcrumb0()
    firrtl.instance breadcrumb1 sym @breadcrumb1 @Breadcrumb1()
  }
}

// Non-local annotations should be updated with additional context if the module
// at the root of the NLA is deduplicated.  The original NLA should be deleted,
// and the annotation should be cloned for each parent of the root module.
// CHECK-LABEL: firrtl.circuit "Context"
firrtl.circuit "Context" {
  // CHECK: firrtl.hierpath [[NLA3:@nla.*]] [@Context::@context1, @Context0::@c0, @ContextLeaf::@w]
  // CHECK: firrtl.hierpath [[NLA1:@nla.*]] [@Context::@context1, @Context0::@c0, @ContextLeaf::@in]
  // CHECK: firrtl.hierpath [[NLA2:@nla.*]] [@Context::@context0, @Context0::@c0, @ContextLeaf::@w]
  // CHECK: firrtl.hierpath [[NLA0:@nla.*]] [@Context::@context0, @Context0::@c0, @ContextLeaf::@in]
  // CHECK-NOT: @context_nla0
  // CHECK-NOT: @context_nla1
  // CHECK-NOT: @context_nla2
  // CHECK-NOT: @context_nla3
  firrtl.hierpath @context_nla0 [@Context0::@c0, @ContextLeaf::@in]
  firrtl.hierpath @context_nla1 [@Context0::@c0, @ContextLeaf::@w]
  firrtl.hierpath @context_nla2 [@Context1::@c1, @ContextLeaf::@in]
  firrtl.hierpath @context_nla3 [@Context1::@c1, @ContextLeaf::@w]

  // CHECK: firrtl.module @ContextLeaf(in %in: !firrtl.uint<1> sym @in [
  // CHECK-SAME: {circt.nonlocal = [[NLA0]], class = "port0"},
  // CHECK-SAME: {circt.nonlocal = [[NLA1]], class = "port1"}]
  firrtl.module @ContextLeaf(in %in : !firrtl.uint<1> sym @in [
      {circt.nonlocal = @context_nla0, class = "port0"},
      {circt.nonlocal = @context_nla2, class = "port1"}
    ]) {

    // CHECK: %w = firrtl.wire sym @w  {annotations = [
    // CHECK-SAME: {circt.nonlocal = [[NLA2]], class = "fake0"}
    // CHECK-SAME: {circt.nonlocal = [[NLA3]], class = "fake1"}
    %w = firrtl.wire sym @w {annotations = [
      {circt.nonlocal = @context_nla1, class = "fake0"},
      {circt.nonlocal = @context_nla3, class = "fake1"}]}: !firrtl.uint<3>
  }
  firrtl.module @Context0() {
    // CHECK: %leaf_in = firrtl.instance leaf sym @c0
    %leaf_in = firrtl.instance leaf sym @c0 @ContextLeaf(in in : !firrtl.uint<1>)
  }
  // CHECK-NOT: firrtl.module @Context1()
  firrtl.module @Context1() {
    %leaf_in = firrtl.instance leaf sym @c1 @ContextLeaf(in in : !firrtl.uint<1>)
  }
  firrtl.module @Context() {
    // CHECK: firrtl.instance context0 sym @context0
    firrtl.instance context0 @Context0()
    // CHECK: firrtl.instance context1 sym @context1
    firrtl.instance context1 @Context1()
  }
}


// External modules should dedup and fixup any NLAs.
// CHECK: firrtl.circuit "ExtModuleTest"
firrtl.circuit "ExtModuleTest" {
  // CHECK: firrtl.hierpath @ext_nla [@ExtModuleTest::@e1, @ExtMod0]
  firrtl.hierpath @ext_nla [@ExtModuleTest::@e1, @ExtMod1]
  // CHECK: firrtl.extmodule @ExtMod0() attributes {annotations = [{circt.nonlocal = @ext_nla}], defname = "a"}
  firrtl.extmodule @ExtMod0() attributes {defname = "a"}
  // CHECK-NOT: firrtl.extmodule @ExtMod1()
  firrtl.extmodule @ExtMod1() attributes {annotations = [{circt.nonlocal = @ext_nla}], defname = "a"}
  firrtl.module @ExtModuleTest() {
    // CHECK: firrtl.instance e0  @ExtMod0()
    firrtl.instance e0 @ExtMod0()
    // CHECK: firrtl.instance e1 sym @e1 @ExtMod0()
    firrtl.instance e1 sym @e1 @ExtMod1()
  }
}

// External modules with NLAs on ports should be properly rewritten.
// https://github.com/llvm/circt/issues/2713
// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "Foo"  {
  // CHECK: firrtl.hierpath @nla_1 [@Foo::@b, @A::@a]
  firrtl.hierpath @nla_1 [@Foo::@b, @B::@b]
  // CHECK: firrtl.extmodule @A(out a: !firrtl.clock sym @a [{circt.nonlocal = @nla_1}])
  firrtl.extmodule @A(out a: !firrtl.clock)
  firrtl.extmodule @B(out b: !firrtl.clock sym @b [{circt.nonlocal = @nla_1}])
  firrtl.module @Foo() {
    %b0_out = firrtl.instance a @A(out a: !firrtl.clock)
    // CHECK: firrtl.instance b sym @b  @A(out a: !firrtl.clock)
    %b1_out = firrtl.instance b sym @b @B(out b: !firrtl.clock)
  }
}

// Extmodules should properly hash port types and not dedup when they differ.
// CHECK-LABEL: firrtl.circuit "Foo"
firrtl.circuit "Foo"  {
  // CHECK: firrtl.extmodule @Bar
  firrtl.extmodule @Bar(
    in clock: !firrtl.clock, out io: !firrtl.bundle<a: clock>)
  // CHECK: firrtl.extmodule @Baz
  firrtl.extmodule @Baz(
    in clock: !firrtl.clock, out io: !firrtl.bundle<a flip: uint<1>, b flip: uint<16>, c: uint<1>>)
  firrtl.module @Foo() {
    %bar_clock, %bar_io = firrtl.instance bar @Bar(
      in clock: !firrtl.clock, out io: !firrtl.bundle<a: clock>)
    %baz_clock, %baz_io = firrtl.instance baz @Baz(
      in clock: !firrtl.clock, out io: !firrtl.bundle<a flip: uint<1>, b flip: uint<16>, c: uint<1>>)
  }
}

// As we dedup modules, the chain on NLAs should continuously grow.
// CHECK-LABEL: firrtl.circuit "Chain"
firrtl.circuit "Chain" {
  // CHECK: firrtl.hierpath [[NLA0:@nla.*]] [@Chain::@chainB1, @ChainB0::@chainA0, @ChainA0::@extchain0, @ExtChain0]
  // CHECK: firrtl.hierpath [[NLA1:@nla.*]] [@Chain::@chainB0, @ChainB0::@chainA0, @ChainA0::@extchain0, @ExtChain0]
  // CHECK: firrtl.module @ChainB0()
  firrtl.module @ChainB0() {
    firrtl.instance chainA0 @ChainA0()
  }
  // CHECK: firrtl.extmodule @ExtChain0() attributes {annotations = [
  // CHECK-SAME:  {circt.nonlocal = [[NLA0]], class = "1"},
  // CHECK-SAME:  {circt.nonlocal = [[NLA1]], class = "0"}], defname = "ExtChain"}
  firrtl.extmodule @ExtChain0() attributes {annotations = [{class = "0"}], defname = "ExtChain"}
  // CHECK-NOT: firrtl.extmodule @ExtChain1()
  firrtl.extmodule @ExtChain1() attributes {annotations = [{class = "1"}], defname = "ExtChain"}
  // CHECK: firrtl.module @ChainA0()
  firrtl.module @ChainA0()  {
    firrtl.instance extchain0 @ExtChain0()
  }
  // CHECK-NOT: firrtl.module @ChainB1()
  firrtl.module @ChainB1() {
    firrtl.instance chainA1 @ChainA1()
  }
  // CHECK-NOT: firrtl.module @ChainA1()
  firrtl.module @ChainA1()  {
    firrtl.instance extchain1 @ExtChain1()
  }
  firrtl.module @Chain() {
    // CHECK: firrtl.instance chainB0 sym @chainB0 @ChainB0()
    firrtl.instance chainB0 @ChainB0()
    // CHECK: firrtl.instance chainB1 sym @chainB1 @ChainB0()
    firrtl.instance chainB1 @ChainB1()
  }
}


// Check that we fixup subfields and connects, when an
// instance op starts returning a different bundle type.
// CHECK-LABEL: firrtl.circuit "Bundle"
firrtl.circuit "Bundle" {
  // CHECK: firrtl.module @Bundle0
  firrtl.module @Bundle0(out %a: !firrtl.bundle<b: bundle<c flip: uint<1>, d: uint<1>>>) { }
  // CHECK-NOT: firrtl.module @Bundle1
  firrtl.module @Bundle1(out %e: !firrtl.bundle<f: bundle<g flip: uint<1>, h: uint<1>>>) { }
  firrtl.module @Bundle() {
    // CHECK: firrtl.instance bundle0  @Bundle0
    %a = firrtl.instance bundle0 @Bundle0(out a: !firrtl.bundle<b: bundle<c flip: uint<1>, d: uint<1>>>)
    // CHECK: firrtl.instance bundle1  @Bundle0
    %e = firrtl.instance bundle1 @Bundle1(out e: !firrtl.bundle<f: bundle<g flip: uint<1>, h: uint<1>>>)

    // CHECK: [[B:%.+]] = firrtl.subfield %bundle0_a(0)
    %b = firrtl.subfield %a(0) : (!firrtl.bundle<b: bundle<c flip: uint<1>, d: uint<1>>>) -> !firrtl.bundle<c flip: uint<1>, d: uint<1>>
    // CHECK: [[F:%.+]] = firrtl.subfield %bundle1_a(0)
    %f = firrtl.subfield %e(0) : (!firrtl.bundle<f: bundle<g flip: uint<1>, h: uint<1>>>) -> !firrtl.bundle<g flip: uint<1>, h: uint<1>>

    // Check that we properly fixup connects when the field names change.
    %w0 = firrtl.wire : !firrtl.bundle<g flip: uint<1>, h: uint<1>>
    // CHECK: [[W0_G:%.+]] = firrtl.subfield %w0(0)
    // CHECK: [[F_G:%.+]] = firrtl.subfield [[F]](0)
    // CHECK: firrtl.connect [[F_G]], [[W0_G]]
    firrtl.connect %w0, %f : !firrtl.bundle<g flip: uint<1>, h: uint<1>>, !firrtl.bundle<g flip: uint<1>, h: uint<1>>
  }
}

// Make sure flipped fields are handled properly. This should pass flow
// verification checking.
// CHECK-LABEL: firrtl.circuit "Flip"
firrtl.circuit "Flip" {
  firrtl.module @Flip0(out %io: !firrtl.bundle<foo flip: uint<1>, fuzz: uint<1>>) {
    %0 = firrtl.subfield %io(0) : (!firrtl.bundle<foo flip: uint<1>, fuzz: uint<1>>) -> !firrtl.uint<1>
    %1 = firrtl.subfield %io(1) : (!firrtl.bundle<foo flip: uint<1>, fuzz: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %1, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  firrtl.module @Flip1(out %io: !firrtl.bundle<bar flip: uint<1>, buzz: uint<1>>) {
    %0 = firrtl.subfield %io(0) : (!firrtl.bundle<bar flip: uint<1>, buzz: uint<1>>) -> !firrtl.uint<1>
    %1 = firrtl.subfield %io(1) : (!firrtl.bundle<bar flip: uint<1>, buzz: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %1, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  firrtl.module @Flip(out %io: !firrtl.bundle<foo: bundle<foo flip: uint<1>, fuzz: uint<1>>, bar: bundle<bar flip: uint<1>, buzz: uint<1>>>) {
    %0 = firrtl.subfield %io(1) : (!firrtl.bundle<foo: bundle<foo flip: uint<1>, fuzz: uint<1>>, bar: bundle<bar flip: uint<1>, buzz: uint<1>>>) -> !firrtl.bundle<bar flip: uint<1>, buzz: uint<1>>
    %1 = firrtl.subfield %io(0) : (!firrtl.bundle<foo: bundle<foo flip: uint<1>, fuzz: uint<1>>, bar: bundle<bar flip: uint<1>, buzz: uint<1>>>) -> !firrtl.bundle<foo flip: uint<1>, fuzz: uint<1>>
    %foo_io = firrtl.instance foo  @Flip0(out io: !firrtl.bundle<foo flip: uint<1>, fuzz: uint<1>>)
    %bar_io = firrtl.instance bar  @Flip1(out io: !firrtl.bundle<bar flip: uint<1>, buzz: uint<1>>)
    firrtl.connect %1, %foo_io : !firrtl.bundle<foo flip: uint<1>, fuzz: uint<1>>, !firrtl.bundle<foo flip: uint<1>, fuzz: uint<1>>
    firrtl.connect %0, %bar_io : !firrtl.bundle<bar flip: uint<1>, buzz: uint<1>>, !firrtl.bundle<bar flip: uint<1>, buzz: uint<1>>
  }
}

// This is checking that the fixup phase due to changing bundle names does not
// block the deduplication of parent modules.
// CHECK-LABEL: firrtl.circuit "DelayedFixup"
firrtl.circuit "DelayedFixup"  {
  // CHECK: firrtl.extmodule @Foo
  firrtl.extmodule @Foo(out a: !firrtl.bundle<a: uint<1>>)
  // CHECK-NOT: firrtl.extmodule @Bar
  firrtl.extmodule @Bar(out b: !firrtl.bundle<b: uint<1>>)
  // CHECK: firrtl.module @Parent0
  firrtl.module @Parent0(out %a: !firrtl.bundle<a: uint<1>>, out %b: !firrtl.bundle<b: uint<1>>) {
    %foo_a = firrtl.instance foo  @Foo(out a: !firrtl.bundle<a: uint<1>>)
    firrtl.connect %a, %foo_a : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
    %bar_b = firrtl.instance bar  @Bar(out b: !firrtl.bundle<b: uint<1>>)
    firrtl.connect %b, %bar_b : !firrtl.bundle<b: uint<1>>, !firrtl.bundle<b: uint<1>>
  }
  // CHECK-NOT: firrtl.module @Parent1
  firrtl.module @Parent1(out %a: !firrtl.bundle<a: uint<1>>, out %b: !firrtl.bundle<b: uint<1>>) {
    %foo_a = firrtl.instance foo  @Foo(out a: !firrtl.bundle<a: uint<1>>)
    firrtl.connect %a, %foo_a : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
    %bar_b = firrtl.instance bar  @Bar(out b: !firrtl.bundle<b: uint<1>>)
    firrtl.connect %b, %bar_b : !firrtl.bundle<b: uint<1>>, !firrtl.bundle<b: uint<1>>
  }
  firrtl.module @DelayedFixup() {
    // CHECK: firrtl.instance parent0  @Parent0
    %parent0_a, %parent0_b = firrtl.instance parent0  @Parent0(out a: !firrtl.bundle<a: uint<1>>, out b: !firrtl.bundle<b: uint<1>>)
    // CHECK: firrtl.instance parent1  @Parent0
    %parent1_a, %parent1_b = firrtl.instance parent1  @Parent1(out a: !firrtl.bundle<a: uint<1>>, out b: !firrtl.bundle<b: uint<1>>)
  }
}

// Don't attach empty annotations onto ops without annotations.
// CHECK-LABEL: firrtl.circuit "NoEmptyAnnos"
firrtl.circuit "NoEmptyAnnos" {
  // CHECK-LABEL: @NoEmptyAnnos0()
  firrtl.module @NoEmptyAnnos0() {
    // CHECK: %w = firrtl.wire  : !firrtl.bundle<a: uint<1>>
    // CHECK: %0 = firrtl.subfield %w(0) : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
    %w = firrtl.wire : !firrtl.bundle<a: uint<1>>
    %0 = firrtl.subfield %w(0) : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
  }
  firrtl.module @NoEmptyAnnos1() {
    %w = firrtl.wire : !firrtl.bundle<a: uint<1>>
    %0 = firrtl.subfield %w(0) : (!firrtl.bundle<a: uint<1>>) -> !firrtl.uint<1>
  }
  firrtl.module @NoEmptyAnnos() {
    firrtl.instance empty0 @NoEmptyAnnos0()
    firrtl.instance empty1 @NoEmptyAnnos1()
  }
}


// Don't deduplicate modules with NoDedup.
// CHECK-LABEL: firrtl.circuit "NoDedup"
firrtl.circuit "NoDedup" {
  firrtl.module @Simple0() { }
  firrtl.module @Simple1() attributes {annotations = [{class = "firrtl.transforms.NoDedupAnnotation"}]} { }
  // CHECK: firrtl.module @NoDedup
  firrtl.module @NoDedup() {
    firrtl.instance simple0 @Simple0()
    firrtl.instance simple1 @Simple1()
  }
}

// Check that modules marked MustDedup have been deduped.
// CHECK-LABEL: firrtl.circuit "MustDedup"
firrtl.circuit "MustDedup" attributes {annotations = [{
    // The annotation should be removed.
    // CHECK-NOT: class = "firrtl.transforms.MustDeduplicateAnnotation"
    class = "firrtl.transforms.MustDeduplicateAnnotation",
    modules = ["~MustDedup|Simple0", "~MustDedup|Simple1"]}]
   } {
  // CHECK: @Simple0
  firrtl.module @Simple0() { }
  // CHECK-NOT: @Simple1
  firrtl.module @Simple1() { }
  // CHECK: firrtl.module @MustDedup
  firrtl.module @MustDedup() {
    firrtl.instance simple0 @Simple0()
    firrtl.instance simple1 @Simple1()
  }
}
