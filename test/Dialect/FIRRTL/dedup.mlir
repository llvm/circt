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

// CHECK-LABEL: firrtl.circuit "Annotations"
firrtl.circuit "Annotations" {
  // CHECK: firrtl.nla [[NLA3:@nla.*]] [#hw.innerNameRef<@Annotations::@annotations1>, @Annotations0]
  // CHECK: firrtl.nla [[NLA2:@nla.*]] [#hw.innerNameRef<@Annotations::@annotations0>, #hw.innerNameRef<@Annotations0::@e>]
  // CHECK: firrtl.nla [[NLA1:@nla.*]] [#hw.innerNameRef<@Annotations::@annotations0>, #hw.innerNameRef<@Annotations0::@c>]
  // CHECK: firrtl.nla [[NLA0:@nla.*]] [#hw.innerNameRef<@Annotations::@annotations1>, #hw.innerNameRef<@Annotations0::@b>]
  // CHECK: firrtl.nla @annos_nla0 [#hw.innerNameRef<@Annotations::@annotations0>, #hw.innerNameRef<@Annotations0::@d>]
  // CHECK: firrtl.nla @annos_nla1 [#hw.innerNameRef<@Annotations::@annotations1>, #hw.innerNameRef<@Annotations0::@d>]
  firrtl.nla @annos_nla0 [#hw.innerNameRef<@Annotations::@annotations0>, #hw.innerNameRef<@Annotations0::@d>]
  firrtl.nla @annos_nla1 [#hw.innerNameRef<@Annotations::@annotations1>, #hw.innerNameRef<@Annotations1::@i>]

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
    // CHECK: firrtl.instance annotations0 sym @annotations0  {annotations = [{circt.nonlocal = @annos_nla0, class = "circt.nonlocal"}, {circt.nonlocal = [[NLA1]], class = "circt.nonlocal"}, {circt.nonlocal = [[NLA2]], class = "circt.nonlocal"}]} @Annotations0()
    // CHECK: firrtl.instance annotations1 sym @annotations1  {annotations = [{circt.nonlocal = @annos_nla1, class = "circt.nonlocal"}, {circt.nonlocal = [[NLA0]], class = "circt.nonlocal"}, {circt.nonlocal = [[NLA3]], class = "circt.nonlocal"}]} @Annotations0()
    firrtl.instance annotations0 sym @annotations0 {annotations = [{circt.nonlocal = @annos_nla0, class = "circt.nonlocal"}]} @Annotations0()
    firrtl.instance annotations1 sym @annotations1 {annotations = [{circt.nonlocal = @annos_nla1, class = "circt.nonlocal"}]} @Annotations1()
  }
}

// Check that module and memory port annotations are merged correctly.
// CHECK-LABEL: firrtl.circuit "PortAnnotations"
firrtl.circuit "PortAnnotations" {
  // CHECK: firrtl.nla [[NLA3:@nla.*]] [#hw.innerNameRef<@PortAnnotations::@portannos0>, #hw.innerNameRef<@PortAnnotations0::@in>]
  // CHECK: firrtl.nla [[NLA2:@nla.*]] [#hw.innerNameRef<@PortAnnotations::@portannos1>, #hw.innerNameRef<@PortAnnotations0::@in>]
  // CHECK: firrtl.nla [[NLA1:@nla.*]] [#hw.innerNameRef<@PortAnnotations::@portannos0>, #hw.innerNameRef<@PortAnnotations0::@bar>]
  // CHECK: firrtl.nla [[NLA0:@nla.*]] [#hw.innerNameRef<@PortAnnotations::@portannos1>, #hw.innerNameRef<@PortAnnotations0::@bar>]
  // CHECK: firrtl.module @PortAnnotations0(in %in: !firrtl.uint<1> sym @in [
  // CHECK-SAME: {circt.nonlocal = [[NLA2]], class = "port1"},
  // CHECK-SAME: {circt.nonlocal = [[NLA3]], class = "port0"}]) {
  firrtl.module @PortAnnotations0(in %in : !firrtl.uint<1> [{class = "port0"}]) {
    // CHECK: %bar_r = firrtl.mem sym @bar
    // CHECK-SAME: portAnnotations =
    // CHECK-SAME:  {circt.nonlocal = [[NLA0]], class = "mem1"},
    // CHECK-SAME:  {circt.nonlocal = [[NLA1]], class = "mem0"}
    %bar_r = firrtl.mem Undefined  {depth = 16 : i64, name = "bar", portAnnotations = [[{class = "mem0"}]], portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
  }
  // CHECK-NOT: firrtl.module @PortAnnotations1
  firrtl.module @PortAnnotations1(in %in : !firrtl.uint<1> [{class = "port1"}])  {
    %bar_r = firrtl.mem Undefined  {depth = 16 : i64, name = "bar", portAnnotations = [[{class = "mem1"}]], portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
  }
  // CHECK: firrtl.module @PortAnnotations
  firrtl.module @PortAnnotations() {
    // CHECK: annotations = [{circt.nonlocal = [[NLA1]], class = "circt.nonlocal"}, {circt.nonlocal = [[NLA3]], class = "circt.nonlocal"}]
    %portannos0_in = firrtl.instance portannos0 @PortAnnotations0(in in: !firrtl.uint<1>)
    // CHECK: annotations = [{circt.nonlocal = [[NLA0]], class = "circt.nonlocal"}, {circt.nonlocal = [[NLA2]], class = "circt.nonlocal"}]
    %portannos1_in = firrtl.instance portannos1 @PortAnnotations1(in in: !firrtl.uint<1>)
  }
}


// Non-local annotations should have their path updated and bread crumbs should
// not be turned into non-local annotations. Note that this should not create
// totally new NLAs for the annotations, it should just update the existing
// ones.
// CHECK-LABEL: firrtl.circuit "Breadcrumb"
firrtl.circuit "Breadcrumb" {
  // CHECK:  @breadcrumb_nla0 [#hw.innerNameRef<@Breadcrumb::@breadcrumb0>, #hw.innerNameRef<@Breadcrumb0::@crumb0>, #hw.innerNameRef<@Crumb::@in>]
  firrtl.nla @breadcrumb_nla0 [#hw.innerNameRef<@Breadcrumb::@breadcrumb0>, #hw.innerNameRef<@Breadcrumb0::@crumb0>, #hw.innerNameRef<@Crumb::@in>]
  // CHECK:  @breadcrumb_nla1 [#hw.innerNameRef<@Breadcrumb::@breadcrumb1>, #hw.innerNameRef<@Breadcrumb0::@crumb0>, #hw.innerNameRef<@Crumb::@in>]
  firrtl.nla @breadcrumb_nla1 [#hw.innerNameRef<@Breadcrumb::@breadcrumb1>, #hw.innerNameRef<@Breadcrumb1::@crumb1>, #hw.innerNameRef<@Crumb::@in>]
  // CHECK:  @breadcrumb_nla2 [#hw.innerNameRef<@Breadcrumb::@breadcrumb0>, #hw.innerNameRef<@Breadcrumb0::@crumb0>, #hw.innerNameRef<@Crumb::@w>]
  firrtl.nla @breadcrumb_nla2 [#hw.innerNameRef<@Breadcrumb::@breadcrumb0>, #hw.innerNameRef<@Breadcrumb0::@crumb0>, #hw.innerNameRef<@Crumb::@w>]
  // CHECK:  @breadcrumb_nla3 [#hw.innerNameRef<@Breadcrumb::@breadcrumb1>, #hw.innerNameRef<@Breadcrumb0::@crumb0>, #hw.innerNameRef<@Crumb::@w>]
  firrtl.nla @breadcrumb_nla3 [#hw.innerNameRef<@Breadcrumb::@breadcrumb1>, #hw.innerNameRef<@Breadcrumb1::@crumb1>, #hw.innerNameRef<@Crumb::@w>]
  firrtl.module @Crumb(in %in: !firrtl.uint<1> sym @in [
      {circt.nonlocal = @breadcrumb_nla0, class = "port0"}, 
      {circt.nonlocal = @breadcrumb_nla1, class = "port1"}]) {
    %w = firrtl.wire sym @w {annotations = [
      {circt.nonlocal = @breadcrumb_nla2, class = "wire0"},
      {circt.nonlocal = @breadcrumb_nla3, class = "wire1"}]}: !firrtl.uint<1>
  }
  // CHECK: firrtl.module @Breadcrumb0()
  firrtl.module @Breadcrumb0() {
    // CHECK: %crumb0_in = firrtl.instance crumb0 sym @crumb0  {annotations = [
    // CHECK-SAME: {circt.nonlocal = @breadcrumb_nla1, class = "circt.nonlocal"},
    // CHECK-SAME: {circt.nonlocal = @breadcrumb_nla3, class = "circt.nonlocal"},
    // CHECK-SAME: {circt.nonlocal = @breadcrumb_nla0, class = "circt.nonlocal"},
    // CHECK-SAME: {circt.nonlocal = @breadcrumb_nla2, class = "circt.nonlocal"}]}
    %crumb_in = firrtl.instance crumb0 sym @crumb0 {annotations = [
      {circt.nonlocal = @breadcrumb_nla0, class = "circt.nonlocal"},
      {circt.nonlocal = @breadcrumb_nla2, class = "circt.nonlocal"}
    ]} @Crumb(in in : !firrtl.uint<1>)
  }
  // CHECK-NOT: firrtl.module @Breadcrumb1()
  firrtl.module @Breadcrumb1() {
    %crumb_in = firrtl.instance crumb1 sym @crumb1 {annotations = [
      {circt.nonlocal = @breadcrumb_nla1, class = "circt.nonlocal"},
      {circt.nonlocal = @breadcrumb_nla3, class = "circt.nonlocal"}
    ]} @Crumb(in in : !firrtl.uint<1>)
  }
  // CHECK: firrtl.module @Breadcrumb()
  firrtl.module @Breadcrumb() {
    // CHECK:     [{circt.nonlocal = @breadcrumb_nla0, class = "circt.nonlocal"},
    // CHECK-SAME: {circt.nonlocal = @breadcrumb_nla2, class = "circt.nonlocal"}]}
    firrtl.instance breadcrumb0 sym @breadcrumb0 {annotations = [
      {circt.nonlocal = @breadcrumb_nla0, class = "circt.nonlocal"},
      {circt.nonlocal = @breadcrumb_nla2, class = "circt.nonlocal"}
    ]} @Breadcrumb0()
    // CHECK:     [{circt.nonlocal = @breadcrumb_nla1, class = "circt.nonlocal"},
    // CHECK-SAME: {circt.nonlocal = @breadcrumb_nla3, class = "circt.nonlocal"}]}
    firrtl.instance breadcrumb1 sym @breadcrumb1 {annotations = [
      {circt.nonlocal = @breadcrumb_nla1, class = "circt.nonlocal"},
      {circt.nonlocal = @breadcrumb_nla3, class = "circt.nonlocal"}
    ]} @Breadcrumb1()
  }
}

// Non-local annotations should be updated with additional context if the module
// at the root of the NLA is deduplicated.  The original NLA should be deleted,
// and the annotation should be cloned for each parent of the root module.
// CHECK-LABEL: firrtl.circuit "Context"
firrtl.circuit "Context" {
  // CHECK: firrtl.nla [[NLA3:@nla.*]] [#hw.innerNameRef<@Context::@context1>, #hw.innerNameRef<@Context0::@c0>, #hw.innerNameRef<@ContextLeaf::@w>]
  // CHECK: firrtl.nla [[NLA1:@nla.*]] [#hw.innerNameRef<@Context::@context1>, #hw.innerNameRef<@Context0::@c0>, #hw.innerNameRef<@ContextLeaf::@in>]
  // CHECK: firrtl.nla [[NLA2:@nla.*]] [#hw.innerNameRef<@Context::@context0>, #hw.innerNameRef<@Context0::@c0>, #hw.innerNameRef<@ContextLeaf::@w>]
  // CHECK: firrtl.nla [[NLA0:@nla.*]] [#hw.innerNameRef<@Context::@context0>, #hw.innerNameRef<@Context0::@c0>, #hw.innerNameRef<@ContextLeaf::@in>]
  // CHECK-NOT: @context_nla0
  // CHECK-NOT: @context_nla1
  // CHECK-NOT: @context_nla2
  // CHECK-NOT: @context_nla3
  firrtl.nla @context_nla0 [#hw.innerNameRef<@Context0::@c0>, #hw.innerNameRef<@ContextLeaf::@in>]
  firrtl.nla @context_nla1 [#hw.innerNameRef<@Context0::@c0>, #hw.innerNameRef<@ContextLeaf::@w>]
  firrtl.nla @context_nla2 [#hw.innerNameRef<@Context1::@c1>, #hw.innerNameRef<@ContextLeaf::@in>]
  firrtl.nla @context_nla3 [#hw.innerNameRef<@Context1::@c1>, #hw.innerNameRef<@ContextLeaf::@w>]

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
    // CHECK: %leaf_in = firrtl.instance leaf sym @c0  {annotations = [
    // CHECK-SAME: {circt.nonlocal = [[NLA0]], class = "circt.nonlocal"},
    // CHECK-SAME: {circt.nonlocal = [[NLA2]], class = "circt.nonlocal"},
    // CHECK-SAME: {circt.nonlocal = [[NLA1]], class = "circt.nonlocal"},
    // CHECK-SAME: {circt.nonlocal = [[NLA3]], class = "circt.nonlocal"}]}
    %leaf_in = firrtl.instance leaf sym @c0 {annotations = [
      {circt.nonlocal = @context_nla0, class = "circt.nonlocal"},
      {circt.nonlocal = @context_nla1, class = "circt.nonlocal"}
    ]} @ContextLeaf(in in : !firrtl.uint<1>)
  }
  // CHECK-NOT: firrtl.module @Context1()
  firrtl.module @Context1() {
    %leaf_in = firrtl.instance leaf sym @c1 {annotations = [
      {circt.nonlocal = @context_nla2, class = "circt.nonlocal"},
      {circt.nonlocal = @context_nla3, class = "circt.nonlocal"}
    ]} @ContextLeaf(in in : !firrtl.uint<1>)
  }
  firrtl.module @Context() {
    // CHECK: firrtl.instance context0 sym @context0  {annotations = [
    // CHECK-SAME: {circt.nonlocal = [[NLA0]], class = "circt.nonlocal"},
    // CHECK-SAME: {circt.nonlocal = [[NLA2]], class = "circt.nonlocal"}]}
    firrtl.instance context0 @Context0()
    // CHECK: firrtl.instance context1 sym @context1  {annotations = [
    // CHECK-SAME: {circt.nonlocal = [[NLA1]], class = "circt.nonlocal"},
    // CHECK-SAME: {circt.nonlocal = [[NLA3]], class = "circt.nonlocal"}]}
    firrtl.instance context1 @Context1()
  }
}


// External modules should dedup and fixup any NLAs.
// CHECK: firrtl.circuit "ExtModuleTest"
firrtl.circuit "ExtModuleTest" {
  // CHECK: firrtl.nla @ext_nla [#hw.innerNameRef<@ExtModuleTest::@e1>, @ExtMod0]
  firrtl.nla @ext_nla [#hw.innerNameRef<@ExtModuleTest::@e1>, @ExtMod1]
  // CHECK: firrtl.extmodule @ExtMod0() attributes {annotations = [{circt.nonlocal = @ext_nla}], defname = "a"}
  firrtl.extmodule @ExtMod0() attributes {defname = "a"}
  // CHECK-NOT: firrtl.extmodule @ExtMod1()
  firrtl.extmodule @ExtMod1() attributes {annotations = [{circt.nonlocal = @ext_nla}], defname = "a"}
  firrtl.module @ExtModuleTest() {
    // CHECK: firrtl.instance e0  @ExtMod0()
    firrtl.instance e0 @ExtMod0()
    // CHECK: firrtl.instance e1 sym @e1  {annotations = [{circt.nonlocal = @ext_nla, class = "circt.nonlocal"}]} @ExtMod0()
    firrtl.instance e1 sym @e1 {annotations = [{circt.nonlocal = @ext_nla, class = "circt.nonlocal"}]} @ExtMod1()
  }
}

// As we dedup modules, the chain on NLAs should continuously grow.
// CHECK-LABEL: firrtl.circuit "Chain"
firrtl.circuit "Chain" {
  // CHECK: firrtl.nla [[NLA0:@nla.*]] [#hw.innerNameRef<@Chain::@chainB1>, #hw.innerNameRef<@ChainB0::@chainA0>, #hw.innerNameRef<@ChainA0::@extchain0>, @ExtChain0]
  // CHECK: firrtl.nla [[NLA1:@nla.*]] [#hw.innerNameRef<@Chain::@chainB0>, #hw.innerNameRef<@ChainB0::@chainA0>, #hw.innerNameRef<@ChainA0::@extchain0>, @ExtChain0]
  // CHECK: firrtl.module @ChainB0()
  firrtl.module @ChainB0() {
    // CHECK: {annotations = [{circt.nonlocal = [[NLA1]], class = "circt.nonlocal"}, {circt.nonlocal = [[NLA0]], class = "circt.nonlocal"}]}
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
    // CHECK: {circt.nonlocal = [[NLA1]], class = "circt.nonlocal"}, {circt.nonlocal = [[NLA0]], class = "circt.nonlocal"}
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
    // CHECK: firrtl.instance chainB0 sym @chainB0  {annotations = [{circt.nonlocal = [[NLA1]], class = "circt.nonlocal"}]} @ChainB0()
    firrtl.instance chainB0 @ChainB0()
    // CHECK: firrtl.instance chainB1 sym @chainB1  {annotations = [{circt.nonlocal = [[NLA0]], class = "circt.nonlocal"}]} @ChainB0()
    firrtl.instance chainB1 @ChainB1()
  }
}


// Check that we fixup subfields, partial connects, and connects, when an
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
    
    // Check that we properly fixup partial connects when the field names change.
    %w1 = firrtl.wire : !firrtl.bundle<g flip: uint<1>>
    // CHECK: [[F_G:%.+]] = firrtl.subfield [[F]](0)
    // CHECK: [[W1_G:%.+]] = firrtl.subfield %w1(0)
    // CHECK: firrtl.partialconnect [[F_G]], [[W1_G]]
    firrtl.partialconnect %w1, %f : !firrtl.bundle<g flip: uint<1>>, !firrtl.bundle<g flip: uint<1>, h: uint<1>>
  }
}

// This is testing an issue in partial connect fixup from a spelling mistake in
// the pass.
firrtl.circuit "PartialIssue" {
  firrtl.module @A(out %a: !firrtl.bundle<member: bundle<a: bundle<clock: clock, reset: asyncreset>>>) { }
  firrtl.module @B(out %b: !firrtl.bundle<member: bundle<b: bundle<clock: clock, reset: asyncreset>>>) { }
  firrtl.module @PartialIssue() {
    %a = firrtl.instance a @A(out a: !firrtl.bundle<member: bundle<a: bundle<clock: clock, reset: asyncreset>>>)
    %b = firrtl.instance b @B(out b: !firrtl.bundle<member: bundle<b: bundle<clock: clock, reset: asyncreset>>>)
    %wb = firrtl.wire : !firrtl.bundle<member: bundle<b: bundle<clock: clock, reset: asyncreset>>>
    firrtl.partialconnect %wb, %b : !firrtl.bundle<member: bundle<b: bundle<clock: clock, reset: asyncreset>>>, !firrtl.bundle<member: bundle<b: bundle<clock: clock, reset: asyncreset>>>
    // CHECK: %0 = firrtl.subfield %wb(0)
    // CHECK: %1 = firrtl.subfield %0(0)
    // CHECK: %2 = firrtl.subfield %b_a(0)
    // CHECK: %3 = firrtl.subfield %2(0)
    // CHECK: firrtl.partialconnect %1, %3
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
    firrtl.partialconnect %1, %foo_io : !firrtl.bundle<foo flip: uint<1>, fuzz: uint<1>>, !firrtl.bundle<foo flip: uint<1>, fuzz: uint<1>>
    firrtl.connect %0, %bar_io : !firrtl.bundle<bar flip: uint<1>, buzz: uint<1>>, !firrtl.bundle<bar flip: uint<1>, buzz: uint<1>>
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
