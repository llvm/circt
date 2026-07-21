// RUN: circt-opt --split-input-file --pass-pipeline='builtin.module(firrtl.circuit(firrtl-inliner))' -allow-unregistered-dialect --verify-diagnostics %s | FileCheck %s

// Regression tests for the FIRRTL ModuleInliner.
//
// Issue-paired circuits are reduced reproducers from GitHub issues, linked
// above each circuit.
// The remaining sections pin reduced crash and miscompile shapes the issue
// reproducers do not reach.
//
// When a hierpath survives in several contexts, the first keeps the symbol
// and later ones fork fresh names (@nla_0, @nla_1, ...).
// Colliding inner symbols are disambiguated with the same suffix scheme.

// -----
// https://github.com/llvm/circt/issues/3373
//
// The issue's own reproducer.  @Bar0 (private+inline, root of the module-leaf
// NLA @nla_3) is instantiated twice, so re-rooting must emit one context per site;
// @Bar1's wire carries both.
// Previously asserted "Module already renamed".
// CHECK-LABEL:  "Inliner"
firrtl.circuit "Inliner"  {
  // CHECK-NEXT:     hw.hierpath @nla_3 [@Inliner::@w_1, @Bar1]
  // CHECK-NEXT:     hw.hierpath private @nla_3_0 [@Inliner::@w_0, @Bar1]
  hw.hierpath @nla_3 [@Bar0::@w, @Bar1]
  // CHECK:          firrtl.module private @Bar1
  firrtl.module private @Bar1() {
    // CHECK-NEXT:       firrtl.wire sym @a
    // CHECK-SAME:         {annotations = [{circt.nonlocal = @nla_3, class = "test2"}, {circt.nonlocal = @nla_3_0, class = "test2"}]}
    %w = firrtl.wire sym @a {annotations = [{circt.nonlocal = @nla_3, class = "test2"}]}: !firrtl.uint<8>
  }
  firrtl.module private @Bar0() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance w sym @w @Bar1()
  }
  // CHECK:          firrtl.module @Inliner
  firrtl.module @Inliner() {
    // CHECK-NEXT:       firrtl.instance bar0_w sym @w_0 @Bar1()
    // CHECK-NEXT:       firrtl.instance bar1_w sym @w_1 @Bar1()
    firrtl.instance bar0 @Bar0()
    firrtl.instance bar1 @Bar0()
    %w = firrtl.wire sym @w : !firrtl.uint<8>
  }
}

// -----
// https://github.com/llvm/circt/issues/4921
//
// A chain of inline modules (@Bar1/@Bar2/@Bar3) carrying module-leaf NLAs is
// fully inlined into @Unreachable; the annotated wires become local there.
// Previously hit "UNREACHABLE ... default constructor for MutableNLA".
// @Bar1 is instantiated twice (directly and via @Bar2), so its wire lands twice.
// CHECK-LABEL:  "Unreachable"
firrtl.circuit "Unreachable" {
  hw.hierpath private @nla_5560 [@Unreachable::@w, @Bar2]
  hw.hierpath private @nla_5561 [@Bar1::@w, @Bar3]
  firrtl.module private @Bar2() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %x = firrtl.wire sym @x {annotations = [{circt.nonlocal = @nla_5560, class = "test0"}]} : !firrtl.uint<8>
    firrtl.instance no sym @no @Bar1()
  }
  firrtl.module private @Bar1() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance bar3 sym @w @Bar3()
  }
  firrtl.module private @Bar3() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %w = firrtl.wire sym @w {annotations = [{circt.nonlocal = @nla_5561, class = "test0"}]} : !firrtl.uint<8>
  }
  // Everything inlines into @Unreachable and the annotations become local.
  // CHECK:          firrtl.module @Unreachable
  // CHECK-NEXT:       firrtl.wire sym @w_0 {annotations = [{class = "test0"}]}
  // CHECK-NEXT:       firrtl.wire sym @x {annotations = [{class = "test0"}]}
  // CHECK-NEXT:       firrtl.wire sym @w_1 {annotations = [{class = "test0"}]}
  // CHECK-NOT:        circt.nonlocal
  firrtl.module @Unreachable() {
    firrtl.instance no sym @no @Bar1()
    firrtl.instance bar2 sym @w @Bar2()
  }
}

// -----
// https://github.com/llvm/circt/issues/3374
// https://github.com/llvm/circt/issues/10720
//
// The issue's own reproducer.  @Bar (private+inline, root of module-leaf NLA
// @nla1) is instantiated once via @Foo and twice directly under the top; @Foo is
// public so it is itself a live context.
// Every instantiation site becomes a distinct re-rooted context, all preserved
// on @Baz's wire.
// Previously produced a DictionaryAttr with duplicate keys (assert / n^2
// broken annotations off).
// CHECK-LABEL:  "CollidingSymbolsReTop"
firrtl.circuit "CollidingSymbolsReTop" {
  // CHECK-NEXT:     hw.hierpath @nla1 [@Bar::@baz, @Baz]
  // CHECK-NEXT:     hw.hierpath private @nla1_0 [@CollidingSymbolsReTop::@baz_0, @Baz]
  // CHECK-NEXT:     hw.hierpath private @nla1_1 [@CollidingSymbolsReTop::@baz, @Baz]
  // CHECK-NEXT:     hw.hierpath private @nla1_2 [@Foo::@baz, @Baz]
  hw.hierpath @nla1 [@Bar::@baz, @Baz]
  // CHECK:          firrtl.module @Baz
  firrtl.module @Baz() {
    // CHECK-NEXT:       firrtl.wire sym @a
    // CHECK-SAME:         {annotations = [{circt.nonlocal = @nla1, class = "hello"}, {circt.nonlocal = @nla1_0, class = "hello"}, {circt.nonlocal = @nla1_1, class = "hello"}, {circt.nonlocal = @nla1_2, class = "hello"}]}
    %a = firrtl.wire sym @a {annotations = [{circt.nonlocal = @nla1, class = "hello"}]} : !firrtl.uint<1>
  }
  firrtl.module @Bar() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance baz sym @baz @Baz()
  }
  firrtl.module @Foo() {
    firrtl.instance bar @Bar()
  }
  firrtl.module @CollidingSymbolsReTop() {
    firrtl.instance b @Bar()
    firrtl.instance c @Bar()
  }
}

// -----
// https://github.com/llvm/circt/issues/10588
//
// Diamond via a chain of inline modules: extmodule @D is the leaf of NLA @nla,
// rooted at @B; both @B and @C are inline, and @B is instantiated twice under the
// top.
// Both surviving contexts must annotate @D, and the leaf annotation on the
// FExtModuleOp must be duplicated so each context's hierpath has a consumer.
// Previously the hierpath retained only one instance; the other was left
// unannotated.
// CHECK-LABEL:  "Test"
firrtl.circuit "Test" {
  // CHECK-NEXT:     hw.hierpath private @nla [@Test::@d_0, @D]
  // CHECK-NEXT:     hw.hierpath private @nla_0 [@Test::@d, @D]
  hw.hierpath private @nla [@B::@c, @C::@d, @D]
  // CHECK:          firrtl.extmodule private @D
  // CHECK-SAME:       annotations = [{circt.nonlocal = @nla, class = "circt.test"}, {circt.nonlocal = @nla_0, class = "circt.test"}]
  firrtl.extmodule private @D() attributes {annotations = [{circt.nonlocal = @nla, class = "circt.test"}]}
  firrtl.module private @C() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance d sym @d @D()
  }
  firrtl.module private @B() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance c sym @c @C()
  }
  // CHECK:          firrtl.module @Test
  firrtl.module @Test() {
    // CHECK-NEXT:       firrtl.instance b1_c_d sym @d @D()
    // CHECK-NEXT:       firrtl.instance b2_c_d sym @d_0 @D()
    firrtl.instance b1 @B()
    firrtl.instance b2 @B()
  }
}

// -----
// https://github.com/llvm/circt/issues/10589
//
// Same diamond shape as #10588 but reached through an inline wrapper: @X (inline,
// NLA root) sits under @Wrapper (also inline), instantiated twice under the top.
// Previously crashed in MutableNLA::inlineModule ("unable to inline the root
// module").
// Both contexts must annotate extmodule @A.
// CHECK-LABEL:  "Test2"
firrtl.circuit "Test2" {
  // CHECK-NEXT:     hw.hierpath private @nla [@Test2::@a_0, @A]
  // CHECK-NEXT:     hw.hierpath private @nla_0 [@Test2::@a, @A]
  hw.hierpath private @nla [@X::@a, @A]
  // CHECK:          firrtl.extmodule private @A
  // CHECK-SAME:       annotations = [{circt.nonlocal = @nla, class = "circt.test"}, {circt.nonlocal = @nla_0, class = "circt.test"}]
  firrtl.extmodule private @A() attributes {annotations = [{circt.nonlocal = @nla, class = "circt.test"}]}
  firrtl.module private @X() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance a sym @a @A()
  }
  firrtl.module private @Wrapper() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance x @X()
  }
  // CHECK:          firrtl.module @Test2
  firrtl.module @Test2() {
    // CHECK-NEXT:       firrtl.instance w_x_a sym @a @A()
    // CHECK-NEXT:       firrtl.instance w2_x_a sym @a_0 @A()
    firrtl.instance w @Wrapper()
    firrtl.instance w2 @Wrapper()
  }
}

// -----
// https://github.com/llvm/circt/issues/10607
//
// Flatten from above the NLA root, active in only one path.  @B (NLA root's leaf
// module) is reached by a flattened parent (@FlattenParent) and a plain parent
// (@NonFlattenParent).
// The annotation targets only the @FlattenParent context;
// after flattening it must be local there and not leak onto @B (still live under
// @NonFlattenParent) or onto the @NonFlattenParent path.
// CHECK-LABEL:  "FlattenPartialLocalNLA"
firrtl.circuit "FlattenPartialLocalNLA" {
  // Retention keeps the collapsed @FlattenParent context (annotation localizes).
  // CHECK:          hw.hierpath private @nla [@FlattenParent::@w_sym]
  hw.hierpath private @nla [@FlattenParent::@b_flatten, @B::@w_sym]
  // @B survives (instantiated by @NonFlattenParent); its wire is now unannotated.
  // CHECK:          firrtl.module private @B
  // CHECK-NEXT:       firrtl.wire sym @w_sym : !firrtl.uint<1>
  firrtl.module private @B() {
    %w = firrtl.wire sym @w_sym {annotations = [{circt.nonlocal = @nla, class = "test"}]} : !firrtl.uint<1>
  }
  // @FlattenParent flattens @B in; the annotation becomes local here.
  // CHECK:          firrtl.module private @FlattenParent
  // CHECK-NEXT:       firrtl.wire sym @w_sym {annotations = [{class = "test"}]}
  firrtl.module private @FlattenParent() attributes {annotations = [{class = "firrtl.transforms.FlattenAnnotation"}]} {
    firrtl.instance w_flatten sym @b_flatten @B()
  }
  // CHECK:          firrtl.module private @NonFlattenParent
  // CHECK-NEXT:       firrtl.instance w sym @w_sym @B()
  firrtl.module private @NonFlattenParent() {
    firrtl.instance w sym @w_sym @B()
  }
  firrtl.module @FlattenPartialLocalNLA() {
    firrtl.instance ip @FlattenParent()
    firrtl.instance np @NonFlattenParent()
  }
}

// -----
// https://github.com/llvm/circt/issues/10608
//
// Flatten from above where the NLA root (@A) is instantiated multiple times under
// a single flattened top (re-rooting + flatten combined).
// Previously crashed in setInnerSym ("Mutable NLA did not contain symbol").
// @A's two paths (a, a2) each localize the leaf annotation; @B's path (b) is
// not part of the NLA and stays unannotated.
// CHECK-LABEL:  "FlattenNLA"
firrtl.circuit "FlattenNLA" {
  // Retention keeps the collapsed path (both @A contexts localize the leaf).
  // CHECK:          hw.hierpath private @nla [@FlattenNLA::@w_sym_0]
  hw.hierpath private @nla [@A::@l, @Leaf::@w_sym]
  firrtl.module private @Leaf() {
    %w = firrtl.wire sym @w_sym {annotations = [{circt.nonlocal = @nla, class = "test"}]} : !firrtl.uint<1>
  }
  firrtl.module private @A() {
    firrtl.instance l sym @l @Leaf()
  }
  firrtl.module private @B() {
    firrtl.instance l sym @l @Leaf()
  }
  // Everything flattens into @FlattenNLA: the two @A paths keep the annotation
  // locally, the @B path does not.
  // CHECK:          firrtl.module @FlattenNLA
  // CHECK-NEXT:       firrtl.wire sym @w_sym {annotations = [{class = "test"}]}
  // CHECK-NEXT:       firrtl.wire sym @w_sym_0 {annotations = [{class = "test"}]}
  // CHECK-NEXT:       firrtl.wire sym @w_sym_1 : !firrtl.uint<1>
  // CHECK-NOT:        circt.nonlocal
  firrtl.module @FlattenNLA() attributes {annotations = [{class = "firrtl.transforms.FlattenAnnotation"}]} {
    firrtl.instance a @A()
    firrtl.instance a2 @A()
    firrtl.instance b @B()
  }
}

// -----
// https://github.com/llvm/circt/issues/10678
//
// A module with both inline and flatten annotations (@Bar), NLA rooted at it and
// targeting a child port.
// Inline wins (the module disappears), the child instance becomes a wire, and
// the port annotation must survive as a local annotation on that wire.
// Previously the annotation was dropped entirely.
// CHECK-LABEL:  "Top"
firrtl.circuit "Top" {
  // Retention keeps the collapsed module-only path (annotation localizes).
  // CHECK:          hw.hierpath private @nla [@Foo]
  hw.hierpath private @nla [@Bar::@baz_inst, @Baz]
  // CHECK:          firrtl.module @Top
  // CHECK-NEXT:       firrtl.instance foo @Foo()
  firrtl.module @Top() {
    firrtl.instance foo @Foo()
  }
  // @Bar inlines+flattens into @Foo; @Baz's port becomes a local wire that keeps
  // the annotation.
  // CHECK:          firrtl.module private @Foo
  // CHECK-NEXT:       firrtl.wire {annotations = [{class = "test"}]}
  // CHECK-NOT:        circt.nonlocal
  firrtl.module private @Foo() {
    firrtl.instance bar @Bar()
  }
  firrtl.module private @Bar() attributes {annotations = [
    {class = "firrtl.passes.InlineAnnotation"},
    {class = "firrtl.transforms.FlattenAnnotation"}
  ]} {
    %baz_port = firrtl.instance baz sym @baz_inst @Baz(in port: !firrtl.uint<1>)
  }
  firrtl.module private @Baz(in %port: !firrtl.uint<1> [{circt.nonlocal = @nla, class = "test"}]) { }
}

// -----
// https://github.com/llvm/circt/issues/10684
// https://github.com/llvm/circt/issues/10685
//
// Nested inline where the root and its parent are both public+inline, so they
// persist and are inlined up the chain.  @nla (rooted at @Grandchild) must keep
// a context for @Grandchild, @Child, and @Parent -- all three instantiation
// contexts of @GreatGrandchild -- since making it local would be invalid for the
// public modules.
// Previously only one context was retained.
// CHECK-LABEL:  "Parent"
firrtl.circuit "Parent" {
  // CHECK-NEXT:     hw.hierpath private @nla [@Grandchild::@ggc_sym, @GreatGrandchild::@target]
  // CHECK-NEXT:     hw.hierpath private @nla_0 [@Child::@ggc_sym, @GreatGrandchild::@target]
  // CHECK-NEXT:     hw.hierpath private @nla_1 [@Parent::@ggc_sym, @GreatGrandchild::@target]
  hw.hierpath private @nla [@Grandchild::@ggc_sym, @GreatGrandchild::@target]
  firrtl.module @Parent() {
    firrtl.instance child sym @child_sym @Child()
  }
  firrtl.module @Child() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance gc sym @gc_sym @Grandchild()
  }
  firrtl.module @Grandchild() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance ggc sym @ggc_sym @GreatGrandchild()
  }
  // CHECK:          firrtl.module @GreatGrandchild
  firrtl.module @GreatGrandchild() {
    // CHECK-NEXT:       firrtl.wire sym @target
    // CHECK-SAME:         {annotations = [{circt.nonlocal = @nla, class = "test"}, {circt.nonlocal = @nla_0, class = "test"}, {circt.nonlocal = @nla_1, class = "test"}]}
    %w = firrtl.wire sym @target {annotations = [{circt.nonlocal = @nla, class = "test"}]} : !firrtl.uint<1>
  }
}

// -----
// https://github.com/llvm/circt/issues/10750
//
// A module marked both inline and flatten (@Bar), reached through an already-
// inlined ancestor (@Foo, inline) that is itself instantiated twice -- a diamond
// above the dual-annotated module, with the NLA threading through a grandchild
// port.
// This is the one shape not otherwise covered.
// Previously UNREACHABLE.
// Both top-level paths localize the leaf annotation.
// CHECK-LABEL:  "Top10750"
firrtl.circuit "Top10750" {
  // Retention keeps the collapsed module-only path (annotations localize).
  // CHECK:          hw.hierpath private @nla [@Top10750]
  hw.hierpath private @nla [@Bar::@baz_inst, @Baz]
  // CHECK:          firrtl.module @Top10750
  // CHECK-NEXT:       firrtl.wire {annotations = [{class = "test"}]}
  // CHECK-NEXT:       firrtl.wire {annotations = [{class = "test"}]}
  // CHECK-NOT:        circt.nonlocal
  firrtl.module @Top10750() {
    firrtl.instance foo1 @Foo()
    firrtl.instance foo2 @Foo()
  }
  firrtl.module private @Foo() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance bar @Bar()
  }
  firrtl.module private @Bar() attributes {annotations = [
    {class = "firrtl.passes.InlineAnnotation"},
    {class = "firrtl.transforms.FlattenAnnotation"}
  ]} {
    %baz_port = firrtl.instance baz sym @baz_inst @Baz(in port: !firrtl.uint<1>)
  }
  firrtl.module private @Baz(in %port: !firrtl.uint<1> [{circt.nonlocal = @nla, class = "test"}]) { }
}

//===----------------------------------------------------------------------===//
// Re-rooting + inner-symbol-conflict characterization (diamond bug class,
// #10588/#10589).
// These are not 1:1 issue reproducers -- they are reduced real-world crash
// shapes exercising the rename-on-conflict paths that the clean-symbol issue
// reproducers above do not reach: an inlined leaf whose inner sym collides in
// the target module must be renamed and every affected NLA context re-pointed
// at the new sym.
//===----------------------------------------------------------------------===//

// -----
// Inlining an NLA-root module into a parent that already owns a different
// instance with the same inner sym forces a rename; the (single-context) NLA
// must be rewritten to reference the renamed sym, not the original.
// CHECK-LABEL:  "InlineRetopSymConflict"
firrtl.circuit "InlineRetopSymConflict" {
  // CHECK-NEXT:     hw.hierpath private @nla [@InlineRetopSymConflict::@sym_0, @Leaf]
  hw.hierpath private @nla [@Inner::@sym, @Leaf]
  // CHECK:          firrtl.extmodule private @Leaf() attributes {annotations = [{circt.nonlocal = @nla, class = "test"}]}
  firrtl.extmodule private @Leaf() attributes {annotations = [{circt.nonlocal = @nla, class = "test"}]}
  firrtl.extmodule private @Other()
  firrtl.module private @Inner() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance leaf sym @sym @Leaf()
  }
  // CHECK:          firrtl.module @InlineRetopSymConflict
  firrtl.module @InlineRetopSymConflict() {
    // CHECK-NEXT:       firrtl.instance conflict sym @sym @Other()
    // CHECK-NEXT:       firrtl.instance i_leaf sym @sym_0 @Leaf()
    firrtl.instance conflict sym @sym @Other()
    firrtl.instance i @Inner()
  }
}

// -----
// Two-level inline chain (@Outer wraps @Inner, both inline) inlined into two
// distinct non-inline parents, one with a sym conflict.
// Each parent gets its own re-rooted NLA pointing at its own (renamed or kept)
// leaf instance, and the extmodule annotation references both.
// CHECK-LABEL:  "InlineRetopNestedTwoParents"
firrtl.circuit "InlineRetopNestedTwoParents" {
  // CHECK-NEXT:     hw.hierpath private @nla [@Root2::@sym, @Leaf]
  // CHECK-NEXT:     hw.hierpath private @nla_0 [@Root1::@sym_0, @Leaf]
  hw.hierpath private @nla [@Inner::@sym, @Leaf]
  // CHECK:          firrtl.extmodule private @Leaf()
  // CHECK-SAME:       annotations = [{circt.nonlocal = @nla, class = "test"}, {circt.nonlocal = @nla_0, class = "test"}]
  firrtl.extmodule private @Leaf() attributes {annotations = [{circt.nonlocal = @nla, class = "test"}]}
  firrtl.extmodule private @Other()
  firrtl.module private @Inner() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance leaf sym @sym @Leaf()
  }
  firrtl.module private @Outer() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance inner @Inner()
  }
  // CHECK:          firrtl.module private @Root1
  // CHECK-NEXT:       firrtl.instance conflict sym @sym @Other()
  // CHECK-NEXT:       firrtl.instance outer_inner_leaf sym @sym_0 @Leaf()
  firrtl.module private @Root1() {
    firrtl.instance conflict sym @sym @Other()
    firrtl.instance outer @Outer()
  }
  // CHECK:          firrtl.module private @Root2
  // CHECK-NEXT:       firrtl.instance inner_leaf sym @sym @Leaf()
  firrtl.module private @Root2() {
    firrtl.instance inner @Inner()
  }
  firrtl.module @InlineRetopNestedTwoParents() {
    firrtl.instance r1 @Root1()
    firrtl.instance r2 @Root2()
  }
}

// -----
// Port NLA on an inline module inlined into two non-inline parents, one with the
// port sym already occupied.
// Exercises port-to-wire lowering on the multi-context path: the port wire is
// renamed in the conflicting context and both wires carry the localized
// annotation once the NLA is fully inlined away.
// CHECK-LABEL:  "InlineRetopPortMultiParent"
firrtl.circuit "InlineRetopPortMultiParent" {
  // Retention keeps the collapsed path (both port wires localize the anno).
  // CHECK:          hw.hierpath private @nla [@NoConflict::@p_sym]
  hw.hierpath private @nla [@W::@b_inst, @B::@p_sym]
  firrtl.module private @B(
      in %p : !firrtl.uint<1> sym @p_sym [{circt.nonlocal = @nla, class = "test"}])
      attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
  }
  firrtl.extmodule private @Other()
  firrtl.module private @W() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance b sym @b_inst @B(in p : !firrtl.uint<1>)
  }
  // Conflicting context: port wire renamed @p_sym_0, annotation localized.
  // CHECK:          firrtl.module private @Conflict
  // CHECK-NEXT:       firrtl.instance other sym @p_sym @Other()
  // CHECK-NEXT:       firrtl.wire sym @p_sym_0 {annotations = [{class = "test"}]}
  firrtl.module private @Conflict() {
    firrtl.instance other sym @p_sym @Other()
    firrtl.instance w @W()
  }
  // Non-conflicting context: port wire keeps @p_sym.
  // CHECK:          firrtl.module private @NoConflict
  // CHECK-NEXT:       firrtl.wire sym @p_sym {annotations = [{class = "test"}]}
  // CHECK-NOT:        circt.nonlocal
  firrtl.module private @NoConflict() {
    firrtl.instance w @W()
  }
  firrtl.module @InlineRetopPortMultiParent() {
    firrtl.instance c @Conflict()
    firrtl.instance nc @NoConflict()
  }
}

// -----
// Two non-inline parents route to the same inline @Qux through the same shared
// inline @Wrapper.
// The @Qux instance in @Wrapper's body accumulates a context sym from each
// parent's inline pass -- exercising "last context sym wins" disambiguation
// across separate inlineInstances calls.
// One parent has a sym conflict (leaf renamed @sym_0); both contexts survive on
// @Bar.
// CHECK-LABEL:  "SharedWrapperTwoParents"
firrtl.circuit "SharedWrapperTwoParents" {
  // CHECK-NEXT:     hw.hierpath private @nla [@ParentNoConflict::@sym, @Bar]
  // CHECK-NEXT:     hw.hierpath private @nla_0 [@ParentConflict::@sym_0, @Bar]
  hw.hierpath private @nla [@Qux::@sym, @Bar]
  // CHECK:          firrtl.extmodule private @Bar()
  // CHECK-SAME:       annotations = [{circt.nonlocal = @nla, class = "test"}, {circt.nonlocal = @nla_0, class = "test"}]
  firrtl.extmodule private @Bar() attributes {annotations = [{circt.nonlocal = @nla, class = "test"}]}
  firrtl.extmodule private @Other()
  firrtl.module private @Qux() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance bar sym @sym @Bar()
  }
  firrtl.module private @Wrapper() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance qux @Qux()
  }
  // CHECK:          firrtl.module private @ParentConflict
  // CHECK-NEXT:       firrtl.instance other sym @sym @Other()
  // CHECK-NEXT:       firrtl.instance wrapper_qux_bar sym @sym_0 @Bar()
  firrtl.module private @ParentConflict() {
    firrtl.instance other sym @sym @Other()
    firrtl.instance wrapper @Wrapper()
  }
  // CHECK:          firrtl.module private @ParentNoConflict
  // CHECK-NEXT:       firrtl.instance wrapper_qux_bar sym @sym @Bar()
  firrtl.module private @ParentNoConflict() {
    firrtl.instance wrapper @Wrapper()
  }
  firrtl.module @SharedWrapperTwoParents() {
    firrtl.instance pc @ParentConflict()
    firrtl.instance pnc @ParentNoConflict()
  }
}

//===----------------------------------------------------------------------===//
// Context-enumeration shape tests.
// The pass enumerates a superset of the minimal set of contexts (the
// any-parent-flattens over-approximation); spurious upper hops are trimmed back
// to the minimal root and trim-equal contexts collapse onto one hierpath.
// These lock the collapsed outputs.
//===----------------------------------------------------------------------===//

// -----
// Single flatten-from-above the NLA root, partial across contexts.  @Mid (NLA
// root's parent) is reached by a flattened parent (@P1, localizes) and a plain
// parent (@P2, stays non-local).
// @Mid's only surviving instantiation is via @P2 (@P1 flattens @Mid away), so
// the @Mid-rooted context and the re-rooted @P2 context denote the same
// physical path.
// The non-evaporating @P2 prefix is trimmed back to @Mid and the two trim-equal
// contexts collapse to a single hierpath (no doubled annotation, no inner sym
// stamped on @P2's instance).
// CHECK-LABEL:  "FlattenFromAbove"
firrtl.circuit "FlattenFromAbove" {
  // CHECK-NEXT:     hw.hierpath private @[[NLA:[a-zA-Z0-9_]+]] [@Mid::@ci, @Child::@w]
  // CHECK-NOT:      hw.hierpath
  hw.hierpath private @nla [@Mid::@ci, @Child::@w]
  firrtl.module @FlattenFromAbove() {
    firrtl.instance p1 @P1()
    firrtl.instance p2 @P2()
  }
  // @P1 flattens @Mid in; the annotation becomes local here.
  // CHECK:          firrtl.module private @P1
  // CHECK-NEXT:       firrtl.wire sym @w {annotations = [{class = "test"}]}
  // CHECK-NOT:        circt.nonlocal
  firrtl.module private @P1() attributes {annotations = [{class = "firrtl.transforms.FlattenAnnotation"}]} {
    firrtl.instance mid @Mid()
  }
  // @P2's instance must not carry an inner sym once the upper hop is trimmed.
  // CHECK:          firrtl.module private @P2
  // CHECK-NEXT:       firrtl.instance mid @Mid()
  firrtl.module private @P2() {
    firrtl.instance mid @Mid()
  }
  firrtl.module private @Mid() {
    firrtl.instance c sym @ci @Child()
  }
  // CHECK:          firrtl.module private @Child
  // CHECK-NEXT:       firrtl.wire sym @w
  // CHECK-SAME:         {annotations = [{circt.nonlocal = @[[NLA]], class = "test"}]}
  // CHECK-NOT:        circt.nonlocal
  firrtl.module private @Child() {
    %w = firrtl.wire sym @w {annotations = [{circt.nonlocal = @nla, class = "test"}]} : !firrtl.uint<1>
  }
}

// -----
// A transit path (@nla1, already rooted at the top) and a re-rooted context
// (@nla2, rooted at inline @Y) that converge to the same namepath once @Y is
// inlined.
// Both annotations are distinct (test1/test2) so both must survive;
// Under retention both survive as their own primaries -- primaries never merge
// (either may have external references) -- so each annotation stays on its own
// symbol.
// IMDCE/Dedup may collapse the redundant pair later.
// CHECK-LABEL:  "TransitAndContextSameRef"
firrtl.circuit "TransitAndContextSameRef" {
  // CHECK:          hw.hierpath private @nla1 [@TransitAndContextSameRef::@leaf_sym, @Leaf]
  // CHECK-NEXT:     hw.hierpath private @nla2 [@TransitAndContextSameRef::@leaf_sym, @Leaf]
  hw.hierpath private @nla1 [@TransitAndContextSameRef::@y_sym, @Y::@leaf_sym, @Leaf]
  hw.hierpath private @nla2 [@Y::@leaf_sym, @Leaf]
  // CHECK:          firrtl.extmodule private @Leaf()
  // CHECK-SAME:       annotations = [{circt.nonlocal = @nla1, class = "test1"}, {circt.nonlocal = @nla2, class = "test2"}]
  firrtl.extmodule private @Leaf() attributes {
    annotations = [{circt.nonlocal = @nla1, class = "test1"},
                   {circt.nonlocal = @nla2, class = "test2"}]
  }
  firrtl.module private @Y() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance leaf sym @leaf_sym @Leaf()
  }
  // CHECK:          firrtl.module @TransitAndContextSameRef
  // CHECK-NEXT:       firrtl.instance y_leaf sym @leaf_sym @Leaf()
  firrtl.module @TransitAndContextSameRef() {
    firrtl.instance y sym @y_sym @Y()
  }
}

//===----------------------------------------------------------------------===//
// Per-field inner symbols as NLA leaves.
// An NLA whose leaf names a per-field inner symbol (fieldID != 0 on an
// aggregate) must have that leaf sym updated when inlining renames it,
// resolving the leaf by field ID (#5776).
// These pin the per-field leaf path for both wires and ports.
//===----------------------------------------------------------------------===//

// -----
// Per-field port sym as NLA leaf, module inlined with a collision on the field
// sym (the port-lowering path).  @nla targets field b (sym @pf) of @Child's bundle
// port; inlining lands the port as a wire, @pf collides with an existing @pf so
// it renames to @pf_0, and the localized annotation stays on fieldID 2.
// CHECK-LABEL:  "PerFieldPort"
firrtl.circuit "PerFieldPort" {
  // Retention keeps the collapsed path; the per-field leaf renamed to @pf_0.
  // CHECK:          hw.hierpath private @nla [@PerFieldPort::@pf_0]
  hw.hierpath private @nla [@PerFieldPort::@i, @Child::@pf]
  firrtl.module private @Child(
      in %x : !firrtl.bundle<a: uint<1>, b: uint<1>> sym [<@pg,1,public>,<@pf,2,public>]
        [{circt.nonlocal = @nla, class = "test", circt.fieldID = 2 : i64}])
      attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
  }
  // CHECK:          firrtl.module @PerFieldPort
  firrtl.module @PerFieldPort() {
    // CHECK-NEXT:       firrtl.wire sym @pf : !firrtl.uint<1>
    // CHECK-NEXT:       firrtl.wire sym [<@pg,1,public>, <@pf_0,2,public>] {annotations = [{circt.fieldID = 2 : i64, class = "test"}]}
    // CHECK-NOT:        circt.nonlocal
    %pf = firrtl.wire sym @pf : !firrtl.uint<1>
    %c_x = firrtl.instance i sym @i @Child(in x : !firrtl.bundle<a: uint<1>, b: uint<1>>)
  }
}

// -----
// Per-field port sym as NLA leaf carried through re-rooting: @Child (NLA leaf
// owner) is reached via inline @Mid instantiated twice, so the leaf stays
// non-local and the per-field leaf sym @pf is preserved in both contexts.
// CHECK-LABEL:  "PerFieldPortRetop"
firrtl.circuit "PerFieldPortRetop" {
  // CHECK-NEXT:     hw.hierpath private @nla [@PerFieldPortRetop::@i_0, @Child::@pf]
  // CHECK-NEXT:     hw.hierpath private @nla_0 [@PerFieldPortRetop::@i, @Child::@pf]
  hw.hierpath private @nla [@Mid::@i, @Child::@pf]
  // CHECK:          firrtl.module private @Child
  // CHECK-SAME:       sym [<@pg,1,public>, <@pf,2,public>]
  // CHECK-SAME:       [{circt.fieldID = 2 : i64, circt.nonlocal = @nla, class = "test"}, {circt.fieldID = 2 : i64, circt.nonlocal = @nla_0, class = "test"}]
  firrtl.module private @Child(
      in %x : !firrtl.bundle<a: uint<1>, b: uint<1>> sym [<@pg,1,public>,<@pf,2,public>]
        [{circt.nonlocal = @nla, class = "test", circt.fieldID = 2 : i64}]) {
  }
  firrtl.module private @Mid() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %c_x = firrtl.instance i sym @i @Child(in x : !firrtl.bundle<a: uint<1>, b: uint<1>>)
  }
  // CHECK:          firrtl.module @PerFieldPortRetop
  // CHECK-NEXT:       firrtl.instance m1_i sym @i @Child
  // CHECK-NEXT:       firrtl.instance m2_i sym @i_0 @Child
  firrtl.module @PerFieldPortRetop() {
    firrtl.instance m1 @Mid()
    firrtl.instance m2 @Mid()
  }
}

//===----------------------------------------------------------------------===//
// Determinism / multi-root.
// Writeback must emit HierPathOps -- and stamp disambiguated inner-symbol
// suffixes -- in a run-to-run stable order (iterate creation order, not DenseMap
// hash order).
// This circuit has four source NLAs across three distinct roots, three of which
// re-root into multiple contexts, so a nondeterministic writeback would
// reorder the emitted hierpaths and/or shuffle the @_N suffixes.
// The full CHECK-NEXT chains below pin both orders.
//===----------------------------------------------------------------------===//

// -----
// @nlaA/@nlaB are rooted at inline @Mid (instantiated 3x) -> three contexts each;
// @nlaD is rooted at inline @Mid2 (instantiated 2x) -> two contexts; @nlaT is a
// plain transit NLA -> one context.
// Nine hierpaths total, order fully pinned.
// CHECK-LABEL:  "MultiRootDeterminism"
firrtl.circuit "MultiRootDeterminism" {
  // CHECK-NEXT:     hw.hierpath private @nlaA [@MultiRootDeterminism::@a_1, @A::@w]
  // CHECK-NEXT:     hw.hierpath private @nlaA_0 [@MultiRootDeterminism::@a_0, @A::@w]
  // CHECK-NEXT:     hw.hierpath private @nlaA_1 [@MultiRootDeterminism::@a, @A::@w]
  // CHECK-NEXT:     hw.hierpath private @nlaB [@MultiRootDeterminism::@b_1, @B::@w]
  // CHECK-NEXT:     hw.hierpath private @nlaB_0 [@MultiRootDeterminism::@b_0, @B::@w]
  // CHECK-NEXT:     hw.hierpath private @nlaB_1 [@MultiRootDeterminism::@b, @B::@w]
  // CHECK-NEXT:     hw.hierpath private @nlaT [@MultiRootDeterminism::@t, @Trans::@c, @C::@w]
  // CHECK-NEXT:     hw.hierpath private @nlaD [@MultiRootDeterminism::@d_0, @D::@w]
  // CHECK-NEXT:     hw.hierpath private @nlaD_0 [@MultiRootDeterminism::@d, @D::@w]
  hw.hierpath private @nlaA [@Mid::@a, @A::@w]
  hw.hierpath private @nlaB [@Mid::@b, @B::@w]
  hw.hierpath private @nlaT [@MultiRootDeterminism::@t, @Trans::@c, @C::@w]
  hw.hierpath private @nlaD [@Mid2::@d, @D::@w]
  // CHECK:          firrtl.module private @A
  // CHECK-NEXT:       firrtl.wire sym @w
  // CHECK-SAME:         {circt.nonlocal = @nlaA, class = "test"}, {circt.nonlocal = @nlaA_0, class = "test"}, {circt.nonlocal = @nlaA_1, class = "test"}
  firrtl.module private @A() {
    %w = firrtl.wire sym @w {annotations = [{circt.nonlocal = @nlaA, class = "test"}]} : !firrtl.uint<1>
  }
  firrtl.module private @B() {
    %w = firrtl.wire sym @w {annotations = [{circt.nonlocal = @nlaB, class = "test"}]} : !firrtl.uint<1>
  }
  firrtl.module private @C() {
    %w = firrtl.wire sym @w {annotations = [{circt.nonlocal = @nlaT, class = "test"}]} : !firrtl.uint<1>
  }
  firrtl.module private @D() {
    %w = firrtl.wire sym @w {annotations = [{circt.nonlocal = @nlaD, class = "test"}]} : !firrtl.uint<1>
  }
  firrtl.module private @Trans() {
    firrtl.instance c sym @c @C()
  }
  firrtl.module private @Mid() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance a sym @a @A()
    firrtl.instance b sym @b @B()
  }
  firrtl.module private @Mid2() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance d sym @d @D()
  }
  // Inner-sym suffixes on the re-rooted instances are pinned too.
  // CHECK:          firrtl.module @MultiRootDeterminism
  // CHECK-NEXT:       firrtl.instance m1_a sym @a @A()
  // CHECK-NEXT:       firrtl.instance m1_b sym @b @B()
  // CHECK-NEXT:       firrtl.instance m2_a sym @a_0 @A()
  // CHECK-NEXT:       firrtl.instance m2_b sym @b_0 @B()
  // CHECK-NEXT:       firrtl.instance m3_a sym @a_1 @A()
  // CHECK-NEXT:       firrtl.instance m3_b sym @b_1 @B()
  // CHECK-NEXT:       firrtl.instance t sym @t @Trans()
  // CHECK-NEXT:       firrtl.instance n1_d sym @d @D()
  // CHECK-NEXT:       firrtl.instance n2_d sym @d_0 @D()
  firrtl.module @MultiRootDeterminism() {
    firrtl.instance m1 @Mid()
    firrtl.instance m2 @Mid()
    firrtl.instance m3 @Mid()
    firrtl.instance t sym @t @Trans()
    firrtl.instance n1 @Mid2()
    firrtl.instance n2 @Mid2()
  }
}
