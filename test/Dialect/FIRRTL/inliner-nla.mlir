// RUN: circt-opt --split-input-file --pass-pipeline='builtin.module(firrtl.circuit(firrtl-inliner))' -allow-unregistered-dialect --verify-diagnostics %s | FileCheck %s
//
// Behavior suite for the ModuleInliner's hierarchical-path (NLA) handling.
// Sections group related behaviors.
// All sections share the RUN line above; --split-input-file isolates them.

// -----
//===--- Annotation retargeting
//
// Annotations are rewritten against each context's final namepath, exactly
// once.
// These pin bugs in that contract:
//  * an annotation cloned mid-walk is already final and carries a
//    planner-minted symbol; the final writeback sweep must recognize it and
//    pass it through, not drop it as a dangling reference.
//  * a context's leaf symbol may be collision-renamed while its owner is
//    cloned; the namepath-keyed merge must see the renamed symbol, or a
//    genuinely different path that equals the stale key is wrongly merged
//    onto it (retargeting its annotation at the wrong op).
//  * a context is written only by the module its last hop lands in;
//    route-based activation can be broader than that ownership (a retained
//    public+inline body shares its original instance ops with the parent's
//    copy), and writing every active context duplicates annotations.

// -----

//===----------------------------------------------------------------------===//
// @B's wire @w is inlined into @A, colliding with @A's own @w and renamed
// @w_0 -- so @nla1 finalizes to [.., @A::@w_0]. @nla2 legitimately owns
// [.., @A::@w] (A's own wire).
// If the merge keyed on @nla1's pre-rename path it would fuse them and point
// @nla2's annotation at the wrong wire; they must stay two distinct hierpaths
// with their annotations on the right wires.
//===----------------------------------------------------------------------===//
// CHECK-LABEL:  firrtl.circuit "LeafSymCollision"
firrtl.circuit "LeafSymCollision" {
  // CHECK:          hw.hierpath private @nla1 [@LeafSymCollision::@iA, @A::@w_0]
  // CHECK-NEXT:     hw.hierpath private @nla2 [@LeafSymCollision::@iA, @A::@w]
  // CHECK-NOT:      hw.hierpath
  hw.hierpath private @nla1 [@LeafSymCollision::@iA, @A::@iB, @B::@w]
  hw.hierpath private @nla2 [@LeafSymCollision::@iA, @A::@w]
  firrtl.module private @B() attributes {annotations = [
      {class = "firrtl.passes.InlineAnnotation"}]} {
    %w = firrtl.wire sym @w {annotations = [
        {circt.nonlocal = @nla1, class = "testB"}]} : !firrtl.uint<1>
  }
  // CHECK:          firrtl.module private @A()
  // CHECK-NEXT:       firrtl.wire sym @w
  // CHECK-SAME:         annotations = [{circt.nonlocal = @nla2, class = "testA"}]
  // CHECK-NEXT:       firrtl.wire sym @w_0
  // CHECK-SAME:         annotations = [{circt.nonlocal = @nla1, class = "testB"}]
  firrtl.module private @A() {
    %w = firrtl.wire sym @w {annotations = [
        {circt.nonlocal = @nla2, class = "testA"}]} : !firrtl.uint<1>
    firrtl.instance iB sym @iB @B()
  }
  firrtl.module @LeafSymCollision() {
    firrtl.instance iA sym @iA @A()
  }
}

// -----

//===----------------------------------------------------------------------===//
// A collision-renamed port that is a context's leaf must update the surviving
// hierpath even when the port itself carries no annotations (the path is kept
// alive by an annotation on another op).
// The emitted path must name the renamed port wire @p_0, not @A's unrelated
// pre-existing @p.
//===----------------------------------------------------------------------===//
// CHECK-LABEL:  firrtl.circuit "PortLeafCollision"
firrtl.circuit "PortLeafCollision" {
  // CHECK:          hw.hierpath private @nla [@PortLeafCollision::@iA, @A::@p_0]
  hw.hierpath private @nla [@PortLeafCollision::@iA, @A::@iB, @B::@p]
  firrtl.module private @B(in %p: !firrtl.uint<1> sym @p) attributes {
      annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %z = firrtl.wire sym @z {annotations = [
        {circt.nonlocal = @nla, class = "keepalive"}]} : !firrtl.uint<1>
  }
  // CHECK:          firrtl.module private @A()
  // CHECK-NEXT:       firrtl.wire sym @p :
  // CHECK-NEXT:       firrtl.wire sym @p_0 :
  firrtl.module private @A() {
    %q = firrtl.wire sym @p : !firrtl.uint<1>
    %iB_p = firrtl.instance iB sym @iB @B(in p: !firrtl.uint<1>)
  }
  firrtl.module @PortLeafCollision() {
    firrtl.instance iA sym @iA @A()
  }
}

// -----

//===----------------------------------------------------------------------===//
// @R (inline) is reached from two live parents, so @nla forks into two
// fresh-named contexts. @L (inline) collapses into surviving @D, so the wire's
// annotation is rewritten during the walk to the forked symbols.
// The final sweep re-visits that wire and must keep both rewritten annotations.
//===----------------------------------------------------------------------===//
// CHECK-LABEL:  firrtl.circuit "ForkedThenCollapsed"
firrtl.circuit "ForkedThenCollapsed" {
  // CHECK:          hw.hierpath private @nla [@P2::@d, @D::@w]
  // CHECK-NEXT:     hw.hierpath private @nla_0 [@P1::@d, @D::@w]
  // CHECK-NOT:      hw.hierpath
  hw.hierpath private @nla [@R::@d, @D::@l, @L::@w]
  firrtl.module private @L() attributes {annotations = [
      {class = "firrtl.passes.InlineAnnotation"}]} {
    %w = firrtl.wire sym @w {annotations = [
        {circt.nonlocal = @nla, class = "test"}]} : !firrtl.uint<1>
  }
  // CHECK:          firrtl.module private @D()
  // CHECK-NEXT:       %l_w = firrtl.wire sym @w
  // CHECK-SAME:         annotations = [{circt.nonlocal = @nla, class = "test"}, {circt.nonlocal = @nla_0, class = "test"}]
  firrtl.module private @D() {
    firrtl.instance l sym @l @L()
  }
  firrtl.module private @R() attributes {annotations = [
      {class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance d sym @d @D()
  }
  firrtl.module private @P1() {
    firrtl.instance r @R()
  }
  firrtl.module private @P2() {
    firrtl.instance r @R()
  }
  firrtl.module @ForkedThenCollapsed() {
    firrtl.instance p1 @P1()
    firrtl.instance p2 @P2()
  }
}

// -----

//===----------------------------------------------------------------------===//
// @M is public (retained) and inline (copied into its parent), so the wire in
// inline @L lands twice: once in @M's retained body, once in the parent.
// Both contexts route through the same original instance @l, but each copy must
// get its localized annotation exactly once -- the parent-copy context must not
// also be written while @M's own body is processed.
//===----------------------------------------------------------------------===//
// CHECK-LABEL:  firrtl.circuit "RetainedBodyOwnership"
firrtl.circuit "RetainedBodyOwnership" {
  // Retention keeps the fully-collapsed path (annotation localizes onto @w).
  // CHECK:          hw.hierpath private @nla [@M::@w]
  hw.hierpath private @nla [@M::@l, @L::@w]
  firrtl.module private @L() attributes {annotations = [
      {class = "firrtl.passes.InlineAnnotation"}]} {
    %w = firrtl.wire sym @w {annotations = [
        {circt.nonlocal = @nla, class = "Test"}]} : !firrtl.uint<1>
  }
  // CHECK:          firrtl.module @M()
  // CHECK-NEXT:       firrtl.wire sym @w
  // CHECK-SAME:         annotations = [{class = "Test"}]
  firrtl.module @M() attributes {annotations = [
      {class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance l sym @l @L()
  }
  // CHECK:          firrtl.module @RetainedBodyOwnership()
  // CHECK-NEXT:       firrtl.wire sym @w
  // CHECK-SAME:         annotations = [{class = "Test"}]
  firrtl.module @RetainedBodyOwnership() {
    firrtl.instance m @M()
  }
}

// -----
//===--- Convergent forks: merging
//
// Late convergence: two hierpaths sourced from different roots can realize to a
// byte-identical namepath once inlining relocates their shared leaf.
// A hierpath is defined solely by its namepath, so the inliner materializes
// just one and retargets every converging annotation at it.
//
// @nla1 is a transit path already rooted at the top; @nla2 is a context rooted
// at the inline module @Y.
// Once @Y is inlined, both namepaths converge to
// [@LateConvergeMerge::@leaf_sym, @Leaf] -- identical.
// Retention keeps both:
// they are distinct source symbols (primaries), and primaries never merge --
// either may be referenced from outside the pass's view, so collapsing one
// would orphan that reference.
// Each annotation stays on its own symbol.
// The redundant pair is IMDCE/Dedup's to collapse later, with full user
// accounting.
// CHECK-LABEL:  firrtl.circuit "LateConvergeMerge"
firrtl.circuit "LateConvergeMerge" {
  // CHECK:          hw.hierpath private @nla1 [@LateConvergeMerge::@leaf_sym, @Leaf]
  // CHECK-NEXT:     hw.hierpath private @nla2 [@LateConvergeMerge::@leaf_sym, @Leaf]
  hw.hierpath private @nla1 [@LateConvergeMerge::@y_sym, @Y::@leaf_sym, @Leaf]
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
  firrtl.module @LateConvergeMerge() {
    firrtl.instance y sym @y_sym @Y()
  }
}

// -----

// @nla1 (the transit path) carries NO annotation; only @nla2 (rooted at inline
// @Y) does.
// After @Y inlines, both converge to
// [@ConvergeDupKeepsAlive::@leaf_sym, @Leaf].
// Under retention both survive as their own primaries: no cross-source-symbol
// merge, so @nla2 keeps test2 on its own symbol and @nla1 is a retained orphan
// (no annotation names it).
// IMDCE/Dedup may collapse the redundant pair later.
// CHECK-LABEL:  firrtl.circuit "ConvergeDupKeepsAlive"
firrtl.circuit "ConvergeDupKeepsAlive" {
  // CHECK:          hw.hierpath private @nla1 [@ConvergeDupKeepsAlive::@leaf_sym, @Leaf]
  // CHECK-NEXT:     hw.hierpath private @nla2 [@ConvergeDupKeepsAlive::@leaf_sym, @Leaf]
  hw.hierpath private @nla1 [@ConvergeDupKeepsAlive::@y_sym, @Y::@leaf_sym, @Leaf]
  hw.hierpath private @nla2 [@Y::@leaf_sym, @Leaf]
  // CHECK:          firrtl.extmodule private @Leaf()
  // CHECK-SAME:       annotations = [{circt.nonlocal = @nla2, class = "test2"}]
  firrtl.extmodule private @Leaf() attributes {
    annotations = [{circt.nonlocal = @nla2, class = "test2"}]
  }
  firrtl.module private @Y() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance leaf sym @leaf_sym @Leaf()
  }
  firrtl.module @ConvergeDupKeepsAlive() {
    firrtl.instance y sym @y_sym @Y()
  }
}

// -----
//===--- Convergent forks: kept distinct
//
// Canonicalization merges hierpaths that realize to the same namepath (see
// inliner-converge-merge.mlir).
// These guard the other direction: contexts that look identical up to their
// surviving hops but denote different physical paths must stay distinct.
// The discriminator is the resolved namepath, which only exists after the walk
// assigns each relocated instance its final sym -- so the merge keys on that,
// never on the pre-inlining hop identity.

// -----

//===----------------------------------------------------------------------===//
// Same inline module instantiated twice in one parent. @M is inlined into @Two
// via both @m1 and @m2, so the leaf instance @i lands in @Two twice with two
// distinct collision-resolved syms.
// The two contexts share the same source hop (@M::@i) and the same destination
// (@Two), differing only in the evaporated @m1/@m2 hop -- so a key over
// surviving hops (final instances still unresolved) would wrongly fuse them and
// point one annotation at the wrong instance.
// They must remain two hierpaths, one per instance.
//===----------------------------------------------------------------------===//
// CHECK-LABEL:  firrtl.circuit "TwoInst"
firrtl.circuit "TwoInst" {
  // CHECK:          hw.hierpath private @nla [@TwoInst::@i_sym_0, @Leaf]
  // CHECK-NEXT:     hw.hierpath private @nla_0 [@TwoInst::@i_sym, @Leaf]
  // CHECK-NOT:      hw.hierpath
  hw.hierpath private @nla [@M::@i_sym, @Leaf]
  // CHECK:          firrtl.extmodule private @Leaf()
  // CHECK-SAME:       annotations = [{circt.nonlocal = @nla, class = "test"}, {circt.nonlocal = @nla_0, class = "test"}]
  firrtl.extmodule private @Leaf() attributes {annotations = [{circt.nonlocal = @nla, class = "test"}]}
  firrtl.module private @M() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance i sym @i_sym @Leaf()
  }
  // CHECK:          firrtl.module @TwoInst
  // CHECK-NEXT:       firrtl.instance m1_i sym @i_sym @Leaf()
  // CHECK-NEXT:       firrtl.instance m2_i sym @i_sym_0 @Leaf()
  firrtl.module @TwoInst() {
    firrtl.instance m1 sym @m1 @M()
    firrtl.instance m2 sym @m2 @M()
  }
}

// -----

//===----------------------------------------------------------------------===//
// AnnotationSplit: an NLA rooted at inline @Mid, reached through two distinct
// live parents @A and @B. @Mid evaporates upward into each, splitting the path
// into two different-rooted namepaths ([@A::...] and [@B::...]).
// Both target the one shared @Leaf op via distinct hierarchies, so both must
// survive -- this is required, not redundant, and the differing root keeps them
// from merging.
//===----------------------------------------------------------------------===//
// CHECK-LABEL:  firrtl.circuit "AnnoSplit"
firrtl.circuit "AnnoSplit" {
  // CHECK:          hw.hierpath private @nla [@B::@leaf_sym, @Leaf]
  // CHECK-NEXT:     hw.hierpath private @nla_0 [@A::@leaf_sym, @Leaf]
  // CHECK-NOT:      hw.hierpath
  hw.hierpath private @nla [@Mid::@leaf_sym, @Leaf]
  // CHECK:          firrtl.extmodule private @Leaf()
  // CHECK-SAME:       annotations = [{circt.nonlocal = @nla, class = "test"}, {circt.nonlocal = @nla_0, class = "test"}]
  firrtl.extmodule private @Leaf() attributes {annotations = [{circt.nonlocal = @nla, class = "test"}]}
  firrtl.module private @Mid() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance leaf sym @leaf_sym @Leaf()
  }
  firrtl.module private @A() {
    firrtl.instance m sym @a_m @Mid()
  }
  firrtl.module private @B() {
    firrtl.instance m sym @b_m @Mid()
  }
  firrtl.module @AnnoSplit() {
    firrtl.instance a @A()
    firrtl.instance b @B()
  }
}

// -----
//===--- Flatten: non-regular terminals
//
// Flatten only absorbs regular-module subtrees: an instance of a non-regular
// module (extmodule here) survives, relocated into the flattening module, and
// a hierpath terminating at that module must relocate with it -- not be
// treated as fully collapsed (which silently dropped the extmodule's
// annotation and the path).

// -----

// CHECK-LABEL:  firrtl.circuit "FlattenExtTerminal"
firrtl.circuit "FlattenExtTerminal" {
  // CHECK:          hw.hierpath private @nla [@FlattenExtTerminal::@k, @K]
  hw.hierpath private @nla [@M::@k, @K]
  // CHECK:          firrtl.extmodule private @K()
  // CHECK-SAME:       annotations = [{circt.nonlocal = @nla, class = "test"}]
  firrtl.extmodule private @K() attributes {annotations = [
      {circt.nonlocal = @nla, class = "test"}]}
  firrtl.module private @M() {
    firrtl.instance k sym @k @K()
  }
  firrtl.module @FlattenExtTerminal() attributes {annotations = [
      {class = "firrtl.transforms.FlattenAnnotation"}]} {
    firrtl.instance m1 @M()
  }
}

// -----

// Two copies of the subtree under one flatten: the path forks, one context per
// relocated extmodule instance.
// CHECK-LABEL:  firrtl.circuit "FlattenExtTerminalDup"
firrtl.circuit "FlattenExtTerminalDup" {
  // CHECK:          hw.hierpath private @nla [@FlattenExtTerminalDup::@k_0, @K]
  // CHECK-NEXT:     hw.hierpath private @nla_0 [@FlattenExtTerminalDup::@k, @K]
  // CHECK-NOT:      hw.hierpath
  hw.hierpath private @nla [@M::@k, @K]
  // CHECK:          firrtl.extmodule private @K()
  // CHECK-SAME:       annotations = [{circt.nonlocal = @nla, class = "test"}, {circt.nonlocal = @nla_0, class = "test"}]
  firrtl.extmodule private @K() attributes {annotations = [
      {circt.nonlocal = @nla, class = "test"}]}
  firrtl.module private @M() {
    firrtl.instance k sym @k @K()
  }
  // CHECK:          firrtl.module @FlattenExtTerminalDup()
  // CHECK-NEXT:       firrtl.instance m1_k sym @k @K()
  // CHECK-NEXT:       firrtl.instance m2_k sym @k_0 @K()
  firrtl.module @FlattenExtTerminalDup() attributes {annotations = [
      {class = "firrtl.transforms.FlattenAnnotation"}]} {
    firrtl.instance m1 @M()
    firrtl.instance m2 @M()
  }
}

// -----
//===--- Flatten: extmodule liveness
//
// @M carries a FlattenAnnotation and instantiates the extmodule @Leaf.
// Flatten dissolves @M's regular-module subtree but must stop at @Leaf
// (extmodules are blackboxes -- there is nothing to inline), keeping the
// instance in place and marking @Leaf live.
firrtl.circuit "Top" {
  // CHECK-LABEL:  firrtl.circuit "Top"
  // CHECK:          firrtl.extmodule private @Leaf()
  firrtl.extmodule private @Leaf()
  // CHECK:          firrtl.module private @M()
  // CHECK-NOT:        FlattenAnnotation
  // CHECK:            firrtl.instance leaf @Leaf()
  firrtl.module private @M() attributes {annotations = [{class = "firrtl.transforms.FlattenAnnotation"}]} {
    firrtl.instance leaf @Leaf()
  }
  // CHECK:          firrtl.module @Top()
  // CHECK-NEXT:       firrtl.instance m @M()
  firrtl.module @Top() {
    firrtl.instance m @M()
  }
}

// -----
//===--- Inline/flatten combinations
//
// A module carrying both inline and flatten annotations.
// The two compose: the module is inlined into its parent (inline), and its
// whole subtree is absorbed as it goes (flatten). @M is inline+flatten, so
// @M -- and everything below it (@A, @B) -- collapses into @Top.
// This differs from either alone: pure inline would leave @A/@B as instances in
// @Top; pure flatten would keep @M as a module with its subtree absorbed into
// it.

// CHECK-LABEL:  firrtl.circuit "InlineFlattenCombo"
firrtl.circuit "InlineFlattenCombo" {
  // No modules other than the top survive.
  // CHECK:          firrtl.module @InlineFlattenCombo(
  // CHECK-NOT:      firrtl.module
  // CHECK-NOT:        firrtl.instance
  // The subtree is flattened in, names prefixed by the instance chain, and the
  // leaf inner symbol is preserved.
  // CHECK:            %m_z = firrtl.wire
  // CHECK:            %m_a_y = firrtl.wire
  // CHECK:            %m_a_b_x = firrtl.wire
  // CHECK:            %m_a_b_w = firrtl.wire sym @w
  firrtl.module private @B(in %x: !firrtl.uint<1>) {
    %w = firrtl.wire sym @w : !firrtl.uint<1>
  }
  firrtl.module private @A(in %y: !firrtl.uint<1>) {
    %b_x = firrtl.instance b sym @b @B(in x: !firrtl.uint<1>)
    firrtl.matchingconnect %b_x, %y : !firrtl.uint<1>
  }
  firrtl.module private @M(in %z: !firrtl.uint<1>) attributes {annotations = [
      {class = "firrtl.passes.InlineAnnotation"},
      {class = "firrtl.transforms.FlattenAnnotation"}]} {
    %a_y = firrtl.instance a sym @a @A(in y: !firrtl.uint<1>)
    firrtl.matchingconnect %a_y, %z : !firrtl.uint<1>
  }
  firrtl.module @InlineFlattenCombo(in %p: !firrtl.uint<1>) {
    %m_z = firrtl.instance m sym @m @M(in z: !firrtl.uint<1>)
    firrtl.matchingconnect %m_z, %p : !firrtl.uint<1>
  }
}

// -----
//===--- instance_choice
//
// An inline-marked module instantiated by firrtl.instance_choice cannot be
// inlined at that site (the choice is resolved later): warn, inline it into
// any plain-instance parents, retain the module, and consume the annotation.

// CHECK-LABEL:  firrtl.circuit "ChoiceOnly"
firrtl.circuit "ChoiceOnly" {
  firrtl.option @Opt {
    firrtl.option_case @A
  }
  firrtl.module private @Default() {}
  // CHECK:          firrtl.module private @Impl() {
  // expected-warning @below {{module marked inline is also instantiated by an operation that cannot be inlined; it is inlined only into its 'firrtl.instance' parents and retained}}
  firrtl.module private @Impl() attributes {annotations = [
      {class = "firrtl.passes.InlineAnnotation"}]} {
    %w = firrtl.wire : !firrtl.uint<1>
  }
  firrtl.module @ChoiceOnly() {
    // expected-note @below {{instantiated here}}
    firrtl.instance_choice inst @Default alternatives @Opt { @A -> @Impl }()
  }
}

// -----

// Mixed instantiation: the plain instance is inlined away; the choice keeps
// the module alive (with its body intact and the annotation consumed).
// CHECK-LABEL:  firrtl.circuit "ChoiceAndInstance"
firrtl.circuit "ChoiceAndInstance" {
  firrtl.option @Opt {
    firrtl.option_case @A
  }
  firrtl.module private @Default() {}
  // CHECK:          firrtl.module private @Impl() {
  // CHECK-NEXT:       firrtl.wire
  // expected-warning @below {{module marked inline is also instantiated by an operation that cannot be inlined; it is inlined only into its 'firrtl.instance' parents and retained}}
  firrtl.module private @Impl() attributes {annotations = [
      {class = "firrtl.passes.InlineAnnotation"}]} {
    %w = firrtl.wire : !firrtl.uint<1>
  }
  // CHECK:          firrtl.module @ChoiceAndInstance()
  // CHECK-NEXT:       firrtl.instance_choice inst @Default alternatives @Opt { @A -> @Impl }
  // CHECK-NEXT:       %direct_w = firrtl.wire
  firrtl.module @ChoiceAndInstance() {
    // expected-note @below {{instantiated here}}
    firrtl.instance_choice inst @Default alternatives @Opt { @A -> @Impl }()
    firrtl.instance direct @Impl()
  }
}

// -----

// A hierpath hopping through an instance_choice, untouched by inlining:
// survives verbatim.
// CHECK-LABEL:  firrtl.circuit "ChoiceHopUntouched"
firrtl.circuit "ChoiceHopUntouched" {
  firrtl.option @Opt {
    firrtl.option_case @A
  }
  // CHECK:          hw.hierpath private @nla [@ChoiceHopUntouched::@sel, @L::@w]
  hw.hierpath private @nla [@ChoiceHopUntouched::@sel, @L::@w]
  firrtl.module private @L() {
    // CHECK:            annotations = [{circt.nonlocal = @nla, class = "test"}]
    %w = firrtl.wire sym @w {annotations = [
        {circt.nonlocal = @nla, class = "test"}]} : !firrtl.uint<1>
  }
  firrtl.module private @LA() {}
  firrtl.module private @Gone() attributes {annotations = [
      {class = "firrtl.passes.InlineAnnotation"}]} {}
  firrtl.module @ChoiceHopUntouched() {
    firrtl.instance_choice sel sym @sel @L alternatives @Opt { @A -> @LA }()
    firrtl.instance g @Gone()
  }
}

// -----

// A hierpath through a choice whose leaf is the inline-marked default target:
// retention keeps the path verbatim and the annotation on the retained wire.
// CHECK-LABEL:  firrtl.circuit "ChoiceHopInlineTarget"
firrtl.circuit "ChoiceHopInlineTarget" {
  firrtl.option @Opt {
    firrtl.option_case @A
  }
  // CHECK:          hw.hierpath private @nla [@ChoiceHopInlineTarget::@sel, @T::@w]
  hw.hierpath private @nla [@ChoiceHopInlineTarget::@sel, @T::@w]
  // CHECK:          firrtl.module private @T() {
  // expected-warning @below {{module marked inline is also instantiated by an operation that cannot be inlined; it is inlined only into its 'firrtl.instance' parents and retained}}
  firrtl.module private @T() attributes {annotations = [
      {class = "firrtl.passes.InlineAnnotation"}]} {
    // CHECK:            annotations = [{circt.nonlocal = @nla, class = "test"}]
    %w = firrtl.wire sym @w {annotations = [
        {circt.nonlocal = @nla, class = "test"}]} : !firrtl.uint<1>
  }
  firrtl.module private @TA() {}
  firrtl.module @ChoiceHopInlineTarget() {
    // expected-note @below {{instantiated here}}
    firrtl.instance_choice sel sym @sel @T alternatives @Opt { @A -> @TA }()
  }
}

// -----

// The module holding the choice hop is itself inlined: the choice op
// relocates into the parent and the hop is retargeted with it -- an
// instance_choice is never absorbed, so the hop never evaporates.
// CHECK-LABEL:  firrtl.circuit "ChoiceHopParentInlined"
firrtl.circuit "ChoiceHopParentInlined" {
  firrtl.option @Opt {
    firrtl.option_case @A
  }
  // CHECK:          hw.hierpath private @nla [@ChoiceHopParentInlined::@sel, @L::@w]
  hw.hierpath private @nla [@ChoiceHopParentInlined::@m, @M::@sel, @L::@w]
  firrtl.module private @L() {
    // CHECK:            annotations = [{circt.nonlocal = @nla, class = "test"}]
    %w = firrtl.wire sym @w {annotations = [
        {circt.nonlocal = @nla, class = "test"}]} : !firrtl.uint<1>
  }
  firrtl.module private @LA() {}
  firrtl.module private @M() attributes {annotations = [
      {class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance_choice sel sym @sel @L alternatives @Opt { @A -> @LA }()
  }
  // CHECK:          firrtl.module @ChoiceHopParentInlined()
  // CHECK-NEXT:       firrtl.instance_choice m_sel sym @sel @L
  firrtl.module @ChoiceHopParentInlined() {
    firrtl.instance m sym @m @M()
  }
}

// -----

// A hierpath rooted at an inline module that a choice keeps alive: the
// retained definition keeps the original path (it covers the choice
// instantiation, which still references it), and each plain-instance copy
// forks its own context.
// No context is enumerated "through" the choice; that instantiation is never
// absorbed.
// CHECK-LABEL:  firrtl.circuit "ChoiceRootRetained"
firrtl.circuit "ChoiceRootRetained" {
  firrtl.option @Opt {
    firrtl.option_case @A
  }
  // CHECK:          hw.hierpath private @nla [@R::@l, @L::@w]
  // CHECK-NEXT:     hw.hierpath private @nla_0 [@ChoiceRootRetained::@l, @L::@w]
  // CHECK-NOT:      hw.hierpath
  hw.hierpath private @nla [@R::@l, @L::@w]
  firrtl.module private @L() {
    // CHECK:            annotations = [{circt.nonlocal = @nla, class = "test"}, {circt.nonlocal = @nla_0, class = "test"}]
    %w = firrtl.wire sym @w {annotations = [
        {circt.nonlocal = @nla, class = "test"}]} : !firrtl.uint<1>
  }
  firrtl.module private @LA() {}
  // expected-warning @below {{module marked inline is also instantiated by an operation that cannot be inlined; it is inlined only into its 'firrtl.instance' parents and retained}}
  firrtl.module private @R() attributes {annotations = [
      {class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance l sym @l @L()
  }
  firrtl.module @ChoiceRootRetained() {
    // expected-note @below {{instantiated here}}
    firrtl.instance_choice c sym @c @R alternatives @Opt { @A -> @LA }()
    firrtl.instance p @R()
  }
}

// -----

// Flatten must stop at an instance_choice: the choice target is retained (never
// absorbed), so a subtree reached through a choice is not localized by an
// ancestor's flatten, and a hierpath routing through the choice survives
// verbatim.
// CHECK-LABEL:  firrtl.circuit "FlattenChoiceHopRetained"
firrtl.circuit "FlattenChoiceHopRetained" {
  firrtl.option @Opt { firrtl.option_case @A }
  // CHECK:          hw.hierpath private @nla [@F::@sel, @M::@l, @L::@w]
  hw.hierpath private @nla [@F::@sel, @M::@l, @L::@w]
  firrtl.module @FlattenChoiceHopRetained() {
    firrtl.instance f sym @f @F()
  }
  firrtl.module private @F() attributes {annotations = [
      {class = "firrtl.transforms.FlattenAnnotation"}]} {
    firrtl.instance_choice sel sym @sel @M alternatives @Opt { @A -> @Alt }()
  }
  firrtl.module private @Alt() {}
  firrtl.module private @M() {
    firrtl.instance l sym @l @L()
  }
  firrtl.module private @L() {
    // CHECK:            annotations = [{circt.nonlocal = @nla, class = "test.fake"}]
    %w = firrtl.wire sym @w {annotations = [
        {circt.nonlocal = @nla, class = "test.fake"}]} : !firrtl.uint<1>
  }
}

// -----

// Same, one level deeper and with the flatten module also inlined: the path
// crosses a choice, a retained subtree, and a second choice.
// The relocated namepath stays a valid instance path (a bare module symbol in a
// non-leaf position fails the hierpath verifier).
// CHECK-LABEL:  firrtl.circuit "FlattenInlineChoiceDeep"
firrtl.circuit "FlattenInlineChoiceDeep" {
  firrtl.option @Opt { firrtl.option_case @A }
  // CHECK:          hw.hierpath private @nla [@FlattenInlineChoiceDeep::@s9, @M2::@s11, @M3::@s5, @M5::@w5]
  hw.hierpath private @nla [@FlattenInlineChoiceDeep::@s10, @M1::@s9, @M2::@s11, @M3::@s5, @M5::@w5]
  firrtl.module @FlattenInlineChoiceDeep() {
    firrtl.instance i11 sym @s10 @M1()
  }
  firrtl.module private @M1() attributes {annotations = [
      {class = "firrtl.passes.InlineAnnotation"}, {class = "firrtl.transforms.FlattenAnnotation"}]} {
    firrtl.instance_choice i10 sym @s9 @M2 alternatives @Opt { @A -> @M5 }()
  }
  firrtl.module private @M2() {
    firrtl.instance i3 sym @s11 @M3()
  }
  firrtl.module private @M3() {
    firrtl.instance_choice i6 sym @s5 @M5 alternatives @Opt { @A -> @M4 }()
  }
  firrtl.module private @M4() {}
  firrtl.module private @M5() {
    %w = firrtl.wire sym @w5 {annotations = [{circt.nonlocal = @nla, class = "test.fake"}]} : !firrtl.uint<1>
  }
}

// -----
//===--- Port-annotation users
//
// `circt.nonlocal` users inside the per-port annotations of memories (and
// instances) are annotation users like any other: an untouched hierpath they
// reference must survive, and one that localizes must be rewritten in place.
// The hazard: erasing the hierpath as unused while a port annotation names it.

// -----

// CHECK-LABEL:  firrtl.circuit "MemPortAnnoUntouched"
firrtl.circuit "MemPortAnnoUntouched" {
  // CHECK:          hw.hierpath private @nla [@MemPortAnnoUntouched::@c, @Child::@mem]
  hw.hierpath private @nla [@MemPortAnnoUntouched::@c, @Child::@mem]
  firrtl.module private @Child() {
    // CHECK:            portAnnotations = {{\[\[}}{circt.nonlocal = @nla, class = "test"}]]
    %mem_r = firrtl.mem sym @mem Undefined {depth = 8 : i64, name = "mem",
        portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32,
        portAnnotations = [[{circt.nonlocal = @nla, class = "test"}]]} :
      !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>
  }
  firrtl.module private @Gone() attributes {annotations = [
      {class = "firrtl.passes.InlineAnnotation"}]} {}
  firrtl.module @MemPortAnnoUntouched() {
    firrtl.instance c sym @c @Child()
    firrtl.instance g @Gone()
  }
}

// -----

// CHECK-LABEL:  firrtl.circuit "MemPortAnnoLocalized"
firrtl.circuit "MemPortAnnoLocalized" {
  // Retention keeps the collapsed path (the mem port anno localizes).
  // CHECK:          hw.hierpath private @nla [@MemPortAnnoLocalized::@mem]
  hw.hierpath private @nla [@MemPortAnnoLocalized::@m, @M::@mem]
  firrtl.module private @M() attributes {annotations = [
      {class = "firrtl.passes.InlineAnnotation"}]} {
    %mem_r = firrtl.mem sym @mem Undefined {depth = 8 : i64, name = "mem",
        portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32,
        portAnnotations = [[{circt.nonlocal = @nla, class = "test"}]]} :
      !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>
  }
  // CHECK:          firrtl.module @MemPortAnnoLocalized()
  // CHECK-NEXT:       firrtl.mem sym @mem
  // CHECK-SAME:         portAnnotations = {{\[\[}}{class = "test"}]]
  firrtl.module @MemPortAnnoLocalized() {
    firrtl.instance m sym @m @M()
  }
}

// -----
//===--- Public hierpath forks
//
// A public hierpath is externally visible, so a fork must not replace it with
// fresh private copies: the first surviving context retargets the original op
// in place -- keeping its symbol and its public visibility -- and only the
// remaining contexts fork private copies.

// CHECK-LABEL:  firrtl.circuit "PublicFork"
firrtl.circuit "PublicFork" {
  // CHECK:          hw.hierpath @nla [@P2::@l, @L::@w]
  // CHECK-NEXT:     hw.hierpath private @nla_0 [@P1::@l, @L::@w]
  // CHECK-NOT:      hw.hierpath
  hw.hierpath @nla [@R::@l, @L::@w]
  firrtl.module private @L() {
    // CHECK:            annotations = [{circt.nonlocal = @nla, class = "test"}, {circt.nonlocal = @nla_0, class = "test"}]
    %w = firrtl.wire sym @w {annotations = [
        {circt.nonlocal = @nla, class = "test"}]} : !firrtl.uint<1>
  }
  firrtl.module private @R() attributes {annotations = [
      {class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance l sym @l @L()
  }
  firrtl.module private @P1() {
    firrtl.instance r @R()
  }
  firrtl.module private @P2() {
    firrtl.instance r @R()
  }
  firrtl.module @PublicFork() {
    firrtl.instance p1 @P1()
    firrtl.instance p2 @P2()
  }
}

// -----
//===--- Trimming: equal-path collapse
//
// Regression target for prefix-trimming / trim-equal collapse.
// Asserts the minimal output.
//
// The companion `FlattenFromAbove` in inliner-issues.mlir exercises the same
// shape in the issue suite; both now collapse to a single hierpath.
//
// @Mid (the NLA root's parent) is reached by a flattening parent @P1 (localizes)
// and a plain parent @P2 (stays non-local).
// Because @P1 flattens @Mid away,
// @Mid's only surviving instantiation is via @P2, so the @Mid-rooted context and
// the re-rooted @P2 context denote the same physical path.
// Today the pass emits both (the any-parent-flattens over-approximation);
// trimming removes the non-evaporating @P2 prefix back to the original root
// @Mid, dedups the two trim-equal contexts into one, reuses the original sym
// (group stays size 1), and never stamps the now-unneeded @sym on @P2's
// instance.
//
// Minimal target:
//   - exactly one hierpath, rooted at @Mid (source-symbol reuse);
//   - @Child's wire carries exactly one nonlocal annotation;
//   - @P2's instance has NO inner sym (the trimmed upper hop needs none).
firrtl.circuit "TrimEqualCollapse" {
  // CHECK-LABEL:  firrtl.circuit "TrimEqualCollapse"
  // CHECK:          hw.hierpath private @[[NLA:[a-zA-Z0-9_]+]] [@Mid::@ci, @Child::@w]
  // CHECK-NOT:      hw.hierpath
  hw.hierpath private @nla [@Mid::@ci, @Child::@w]
  firrtl.module @TrimEqualCollapse() {
    firrtl.instance p1 @P1()
    firrtl.instance p2 @P2()
  }
  // @P1 flattens @Mid in; the annotation becomes local here.
  // CHECK:          firrtl.module private @P1
  // CHECK-NEXT:       firrtl.wire sym @w {annotations = [{class = "test"}]}
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
  // Exactly one surviving nonlocal annotation, referencing the sole hierpath.
  // CHECK:          firrtl.module private @Child
  // CHECK-NEXT:       firrtl.wire sym @w {annotations = [{circt.nonlocal = @[[NLA]], class = "test"}]}
  // CHECK-NOT:        circt.nonlocal
  firrtl.module private @Child() {
    %w = firrtl.wire sym @w {annotations = [{circt.nonlocal = @nla, class = "test"}]} : !firrtl.uint<1>
  }
}

// -----
//===--- Trimming: root boundaries
//
// The inliner traces an NLA's root upward through instantiation contexts, then
// trims leading upper hops that don't actually evaporate under inlining.
// The trim boundary must consult the root module's own fate -- not just the
// upper hops -- because the trace may have been triggered by a
// context-insensitive
// "is-flattened-somewhere" over-approximation that doesn't hold on the concrete
// path in question.
// These two circuits pin both sides of that boundary.

//===----------------------------------------------------------------------===//
// (1) The root survives on a concrete non-flattening path: trim back to it.
//
// @R is reached by a flattening parent @F (where it evaporates -> local) and by
// a plain two-level chain @Keep -> @Mid2 -> @R (where it survives).
// On the surviving path the trace above @R was spurious, so the whole [@Keep,
// @Mid2] prefix trims back to @R: one hierpath rooted at @R, a single local
// annotation in @F, and no inner syms stamped on the trimmed-through instances.
//===----------------------------------------------------------------------===//
// CHECK-LABEL:  firrtl.circuit "RootSurvives"
firrtl.circuit "RootSurvives" {
  // CHECK:          hw.hierpath private @[[NLA:[a-zA-Z0-9_]+]] [@R::@ci, @Child::@w]
  // CHECK-NOT:      hw.hierpath
  hw.hierpath private @nla [@R::@ci, @Child::@w]
  firrtl.module @RootSurvives() {
    firrtl.instance f @F()
    firrtl.instance keep @Keep()
  }
  // Flatten copy: @R evaporates here, annotation becomes local.
  // CHECK:          firrtl.module private @F
  // CHECK-NEXT:       firrtl.wire sym @w {annotations = [{class = "test"}]}
  // CHECK-NOT:        circt.nonlocal
  firrtl.module private @F() attributes {annotations = [{class = "firrtl.transforms.FlattenAnnotation"}]} {
    firrtl.instance r @R()
  }
  // Trimmed-through instances keep no inner sym.
  // CHECK:          firrtl.module private @Keep
  // CHECK-NEXT:       firrtl.instance m2 @Mid2()
  firrtl.module private @Keep() {
    firrtl.instance m2 @Mid2()
  }
  // CHECK:          firrtl.module private @Mid2
  // CHECK-NEXT:       firrtl.instance r @R()
  firrtl.module private @Mid2() {
    firrtl.instance r @R()
  }
  firrtl.module private @R() {
    firrtl.instance c sym @ci @Child()
  }
  // CHECK:          firrtl.module private @Child
  // CHECK-NEXT:       firrtl.wire sym @w {annotations = [{circt.nonlocal = @[[NLA]], class = "test"}]}
  // CHECK-NOT:        circt.nonlocal
  firrtl.module private @Child() {
    %w = firrtl.wire sym @w {annotations = [{circt.nonlocal = @nla, class = "test"}]} : !firrtl.uint<1>
  }
}

// -----

//===----------------------------------------------------------------------===//
// (2) The root itself is inline: do not trim to it -- it is deleted.
//
// @R is the NLA root and carries an InlineAnnotation, so it evaporates into its
// parent @Keep.
// The trace above @R was founded (@R really is gone), so the trim must stop at
// @Keep and re-root the path there.
// Rooting at @R would leave a hierpath naming a module that inlining removed ->
// invalid IR.
//===----------------------------------------------------------------------===//
// CHECK-LABEL:  firrtl.circuit "InlineRootNotTrimmed"
firrtl.circuit "InlineRootNotTrimmed" {
  // CHECK:          hw.hierpath private @nla [@Keep::@ci, @Child::@w]
  hw.hierpath private @nla [@R::@ci, @Child::@w]
  firrtl.module @InlineRootNotTrimmed() {
    firrtl.instance keep @Keep()
  }
  // @R inlined in: the @ci-sym'd instance now lives in @Keep, and the hierpath
  // above re-roots there instead of at the deleted @R.
  // CHECK:          firrtl.module private @Keep
  // CHECK-NEXT:       firrtl.instance r_c sym @ci @Child()
  firrtl.module private @Keep() {
    firrtl.instance r @R()
  }
  firrtl.module private @R() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance c sym @ci @Child()
  }
  // CHECK:          firrtl.module private @Child
  // CHECK-NEXT:       firrtl.wire sym @w {annotations = [{circt.nonlocal = @nla, class = "test"}]}
  firrtl.module private @Child() {
    %w = firrtl.wire sym @w {annotations = [{circt.nonlocal = @nla, class = "test"}]} : !firrtl.uint<1>
  }
}

// -----
//===--- Trimming: through inline modules
//
// Trimming an NLA's spurious upper prefix must pass through inline modules but
// stop at flatten modules -- they evaporate differently.
// Flatten localizes a subtree into the flattening module (nothing below
// survives to root at);
// inline merely relocates a body up into its parent (everything below keeps its
// identity and survives, just re-parented).
// So an inline module on the upper path is transparent to the trim, while a
// flatten pins the root.

// -----

//===----------------------------------------------------------------------===//
// Inline module between the root and solid ground: trim through it.
//
// @P (and so @R) are conservatively treated as flattened via @Flat.
// On the concrete path @S -> @I -> @P -> @R, @I is inline (relocates @P up into
// @S) and nothing flattens @R, so @R survives and is the minimal root.
// The @S-rooted context the trace produces must collapse into the single
// @R-rooted path -- one hierpath, one annotation on the leaf -- not linger as a
// redundant longer path.
//===----------------------------------------------------------------------===//
// CHECK-LABEL:  firrtl.circuit "InlineOnUpperPath"
firrtl.circuit "InlineOnUpperPath" {
  // CHECK:          hw.hierpath private @[[NLA:[a-zA-Z0-9_]+]] [@R::@ci, @Child::@w]
  // CHECK-NOT:      hw.hierpath
  hw.hierpath private @nla [@R::@ci, @Child::@w]
  firrtl.module @InlineOnUpperPath() {
    firrtl.instance flat @Flat()
    firrtl.instance s @S()
  }
  // Over-approximation source: flattens @P/@R, marking them as under a flatten.
  firrtl.module private @Flat() attributes {annotations = [{class = "firrtl.transforms.FlattenAnnotation"}]} {
    firrtl.instance p @P()
  }
  firrtl.module private @S() {
    firrtl.instance i @I()
  }
  firrtl.module private @I() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance p @P()
  }
  firrtl.module private @P() {
    firrtl.instance r @R()
  }
  firrtl.module private @R() {
    firrtl.instance c sym @ci @Child()
  }
  // CHECK:          firrtl.module private @Child
  // CHECK-NEXT:       firrtl.wire sym @w {annotations = [{circt.nonlocal = @[[NLA]], class = "test"}]}
  // CHECK-NOT:        circt.nonlocal
  firrtl.module private @Child() {
    %w = firrtl.wire sym @w {annotations = [{circt.nonlocal = @nla, class = "test"}]} : !firrtl.uint<1>
  }
}

// -----

//===----------------------------------------------------------------------===//
// Flatten below an inline on the upper path: trim through the inline, stop at
// the flatten. @I (inline) is transparent, but @MidFlat flattens @R away, so the
// leaf localizes into @MidFlat -- the path is rooted there, not at @R.
//===----------------------------------------------------------------------===//
// CHECK-LABEL:  firrtl.circuit "FlattenBelowInline"
firrtl.circuit "FlattenBelowInline" {
  // The leaf flattens into @MidFlat and localizes; retention keeps the
  // collapsed one-hop path.
  // CHECK:          hw.hierpath private @nla [@MidFlat::@w]
  hw.hierpath private @nla [@R::@ci, @Child::@w]
  firrtl.module @FlattenBelowInline() {
    firrtl.instance i @I()
  }
  firrtl.module private @I() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance m @MidFlat()
  }
  // @MidFlat flattens its subtree (@R, @Child) into itself.
  // CHECK:          firrtl.module private @MidFlat
  // CHECK-NEXT:       firrtl.wire sym @w {annotations = [{class = "test"}]}
  // CHECK-NOT:        circt.nonlocal
  firrtl.module private @MidFlat() attributes {annotations = [{class = "firrtl.transforms.FlattenAnnotation"}]} {
    firrtl.instance r @R()
  }
  firrtl.module private @R() {
    firrtl.instance c sym @ci @Child()
  }
  firrtl.module private @Child() {
    %w = firrtl.wire sym @w {annotations = [{circt.nonlocal = @nla, class = "test"}]} : !firrtl.uint<1>
  }
}

// -----

// Module-targeting NLA with its annotation on the module op itself.
// Inlining the middle module retargets the path; the annotation stays.
// CHECK-LABEL:  firrtl.circuit "ModuleAnnoRetarget"
firrtl.circuit "ModuleAnnoRetarget" {
  // CHECK:          hw.hierpath private @nla [@ModuleAnnoRetarget::@c, @Child]
  hw.hierpath private @nla [@ModuleAnnoRetarget::@m, @Mid::@c, @Child]
  // CHECK:          firrtl.module private @Child
  // CHECK-SAME:       circt.nonlocal = @nla
  firrtl.module private @Child() attributes {annotations = [{circt.nonlocal = @nla, class = "circt.test"}]} {}
  firrtl.module private @Mid() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance c sym @c @Child()
  }
  firrtl.module @ModuleAnnoRetarget() {
    firrtl.instance m sym @m @Mid()
  }
}

// -----

// Inlining the module a module-targeting NLA ends at.
// The module-op annotation localizes; retention keeps the collapsed one-hop
// path (an orphan here -- IMDCE GCs it downstream).
// CHECK-LABEL:  firrtl.circuit "ModuleAnnoTargetInlined"
// CHECK:          hw.hierpath private @nla [@ModuleAnnoTargetInlined]
firrtl.circuit "ModuleAnnoTargetInlined" {
  hw.hierpath private @nla [@ModuleAnnoTargetInlined::@c, @Child]
  firrtl.module private @Child() attributes {annotations = [{circt.nonlocal = @nla, class = "circt.test"}, {class = "firrtl.passes.InlineAnnotation"}]} {}
  firrtl.module @ModuleAnnoTargetInlined() {
    firrtl.instance c sym @c @Child()
  }
}

// -----

// A hierpath with no users at all: retention keeps it (collapsed one-hop
// orphan) rather than GC'ing -- IMDCE removes it downstream.
// CHECK-LABEL:  firrtl.circuit "ModuleOnlyGC"
// CHECK:          hw.hierpath private @nla [@ModuleOnlyGC]
firrtl.circuit "ModuleOnlyGC" {
  hw.hierpath private @nla [@ModuleOnlyGC::@c, @Child]
  firrtl.module private @Child() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {}
  firrtl.module @ModuleOnlyGC() {
    firrtl.instance c sym @c @Child()
  }
}

// -----

// An unused, untouched hierpath: retention leaves it as-is (like the previous
// inliner did); GC of a truly-dead path is IMDeadCodeElim's job, not ours.
// CHECK-LABEL:  firrtl.circuit "ModuleOnlyGCUntouched"
// CHECK:          hw.hierpath private @nla [@ModuleOnlyGCUntouched::@c, @Child]
firrtl.circuit "ModuleOnlyGCUntouched" {
  hw.hierpath private @nla [@ModuleOnlyGCUntouched::@c, @Child]
  firrtl.module private @Child() {}
  firrtl.module @ModuleOnlyGCUntouched() {
    firrtl.instance c sym @c @Child()
  }
}

// -----

// A hierpath rooted in a module that is dead-erased is itself erased: the dead
// root is never live, so traceUpUntilSurviving discovers no context, the source
// path has no surviving target, and the writeback removes it
// (num-hierpaths-erased).
// Pins the one remaining erase path.
// CHECK-LABEL:  firrtl.circuit "DeadRootedHierPath"
firrtl.circuit "DeadRootedHierPath" {
  // CHECK-NOT:      hw.hierpath
  hw.hierpath private @p [@Unreachable::@w]
  // CHECK-NOT:        @Unreachable
  firrtl.module private @Unreachable() {
    %w = firrtl.wire sym @w : !firrtl.uint<1>
  }
  // CHECK:          firrtl.module @DeadRootedHierPath()
  // CHECK-NOT:        @Unreachable
  firrtl.module @DeadRootedHierPath() {}
}
