// RUN: circt-opt --split-input-file --pass-pipeline='builtin.module(firrtl.circuit(firrtl-inliner))' -allow-unregistered-dialect --verify-diagnostics %s | FileCheck %s

// Test that an external module as the main module works.
firrtl.circuit "main_extmodule" {
  firrtl.extmodule @main_extmodule()
  firrtl.module private @unused () { }
}
// CHECK-LABEL: firrtl.circuit "main_extmodule" {
// CHECK-NEXT:   firrtl.extmodule @main_extmodule()
// CHECK-NEXT: }

// Test that unused modules are deleted.
firrtl.circuit "delete_dead_modules" {
firrtl.module @delete_dead_modules () {
  firrtl.instance used @used()
  firrtl.instance used @used_ext()
}
firrtl.module private @unused () { }
firrtl.module private @used () { }
firrtl.extmodule private @unused_ext ()
firrtl.extmodule private @used_ext ()
}
// CHECK-LABEL: firrtl.circuit "delete_dead_modules" {
// CHECK-NEXT:   firrtl.module @delete_dead_modules() {
// CHECK-NEXT:     firrtl.instance used @used()
// CHECK-NEXT:     firrtl.instance used @used_ext
// CHECK-NEXT:   }
// CHECK-NEXT:   firrtl.module private @used() {
// CHECK-NEXT:   }
// CHECK-NEXT:   firrtl.extmodule private @used_ext()
// CHECK-NEXT: }


// Test basic inlining
firrtl.circuit "inlining" {
firrtl.module @inlining() {
  firrtl.instance test1 @test1()
}
firrtl.module private @test1()
  attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
  %test_wire = firrtl.wire : !firrtl.uint<2>
  firrtl.instance test2 @test2()
}
firrtl.module private @test2()
  attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
  %test_wire = firrtl.wire : !firrtl.uint<2>
}
}
// CHECK-LABEL: firrtl.circuit "inlining" {
// CHECK-NEXT:   firrtl.module @inlining() {
// CHECK-NEXT:     %test1_test_wire = firrtl.wire : !firrtl.uint<2>
// CHECK-NEXT:     %test1_test2_test_wire = firrtl.wire : !firrtl.uint<2>
// CHECK-NEXT:   }
// CHECK-NEXT: }


// Test basic flattening:
//   1. All instances under the flattened module are inlined.
//   2. The flatten annotation is removed.
firrtl.circuit "flattening" {
firrtl.module @flattening()
  attributes {annotations = [{class = "firrtl.transforms.FlattenAnnotation"}]} {
  firrtl.instance test1 @test1()
}
firrtl.module private @test1() {
  %test_wire = firrtl.wire : !firrtl.uint<2>
  firrtl.instance test2 @test2()
}
firrtl.module private @test2() {
  %test_wire = firrtl.wire : !firrtl.uint<2>
}
}
// CHECK-LABEL: firrtl.circuit "flattening"
// CHECK-NEXT:   firrtl.module @flattening()
// CHECK-NOT:      annotations
// CHECK-NEXT:     %test1_test_wire = firrtl.wire : !firrtl.uint<2>
// CHECK-NEXT:     %test1_test2_test_wire = firrtl.wire : !firrtl.uint<2>
// CHECK-NOT:    firrtl.module private @test1
// CHECK-NOT:    firrtl.module private @test2


// Test that inlining and flattening compose well.
firrtl.circuit "compose" {
firrtl.module @compose() {
  firrtl.instance test1 @test1()
  firrtl.instance test2 @test2()
  firrtl.instance test3 @test3()
}
firrtl.module private @test1() attributes {annotations =
        [{class = "firrtl.transforms.FlattenAnnotation"},
         {class = "firrtl.passes.InlineAnnotation"}]} {
  %test_wire = firrtl.wire : !firrtl.uint<2>
  firrtl.instance test2 @test2()
  firrtl.instance test3 @test3()
}
firrtl.module private @test2() attributes {annotations =
        [{class = "firrtl.passes.InlineAnnotation"}]} {
  %test_wire = firrtl.wire : !firrtl.uint<2>
  firrtl.instance test3 @test3()
}
firrtl.module private @test3() {
  %test_wire = firrtl.wire : !firrtl.uint<2>
}
}
// CHECK-LABEL: firrtl.circuit "compose" {
// CHECK-NEXT:   firrtl.module @compose() {
// CHECK-NEXT:     %test1_test_wire = firrtl.wire  : !firrtl.uint<2>
// CHECK-NEXT:     %test1_test2_test_wire = firrtl.wire  : !firrtl.uint<2>
// CHECK-NEXT:     %test1_test2_test3_test_wire = firrtl.wire  : !firrtl.uint<2>
// CHECK-NEXT:     %test1_test3_test_wire = firrtl.wire  : !firrtl.uint<2>
// CHECK-NEXT:     %test2_test_wire = firrtl.wire  : !firrtl.uint<2>
// CHECK-NEXT:     firrtl.instance test2_test3 @test3()
// CHECK-NEXT:     firrtl.instance test3 @test3()
// CHECK-NEXT:   }
// CHECK-NEXT:   firrtl.module private @test3() {
// CHECK-NEXT:     %test_wire = firrtl.wire  : !firrtl.uint<2>
// CHECK-NEXT:   }
// CHECK-NEXT: }

// Test behavior inlining a flattened module into multiple parents
firrtl.circuit "TestInliningFlatten" {
firrtl.module @TestInliningFlatten() {
  firrtl.instance test1 @test1()
  firrtl.instance test2 @test2()
}
firrtl.module private @test1() {
  %test_wire = firrtl.wire : !firrtl.uint<2>
  firrtl.instance fi @flatinline()
}
firrtl.module private @test2() {
  %test_wire = firrtl.wire : !firrtl.uint<2>
  firrtl.instance fi @flatinline()
}
firrtl.module private @flatinline() attributes {annotations =
        [{class = "firrtl.transforms.FlattenAnnotation"},
         {class = "firrtl.passes.InlineAnnotation"}]} {
  %test_wire = firrtl.wire : !firrtl.uint<2>
  firrtl.instance leaf @leaf()
}
firrtl.module private @leaf() {
  %test_wire = firrtl.wire : !firrtl.uint<2>
}
}
// CHECK-LABEL: firrtl.circuit "TestInliningFlatten"
// CHECK-NEXT:    firrtl.module @TestInliningFlatten
// inlining a flattened module should not contain 'instance's:
// CHECK:       firrtl.module private @test1()
// CHECK-NOT:     firrtl.instance
// inlining a flattened module should not contain 'instance's:
// CHECK:       firrtl.module private @test2()
// CHECK-NOT:     firrtl.instance
// These should be removed
// CHECK-NOT:   @flatinline
// CHECK-NOT:   @leaf

// Test behavior retaining public modules but not their annotations
firrtl.circuit "TestPubAnno" {
firrtl.module @TestPubAnno() {
  firrtl.instance fi @flatinline()
}
firrtl.module @flatinline() attributes {annotations =
        [{class = "firrtl.transforms.FlattenAnnotation"},
         {class = "firrtl.passes.InlineAnnotation"}]} {
  %test_wire = firrtl.wire : !firrtl.uint<2>
  firrtl.instance leaf @leaf()
}
firrtl.module private @leaf() {
  %test_wire = firrtl.wire : !firrtl.uint<2>
}
}
// CHECK-LABEL: firrtl.circuit "TestPubAnno"
// CHECK-NEXT:    firrtl.module @TestPubAnno
// CHECK-NOT: annotation
// This is preserved, public
// CHECK:         firrtl.module @flatinline
// CHECK-NOT: annotation
// CHECK-NOT: @leaf

// This is testing that connects are properly replaced when inlining.
// This is also testing that deep-cloning and value remapping work correctly.
firrtl.circuit "TestConnections" {
firrtl.module @InlineMe0(in %in0: !firrtl.uint<4>, in %in1: !firrtl.uint<4>,
                         out %out0: !firrtl.uint<4>, out %out1: !firrtl.uint<4>)
        attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
  %0 = firrtl.and %in0, %in1 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out0, %0 : !firrtl.uint<4>, !firrtl.uint<4>
  %1 = firrtl.and %in0, %in1 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  firrtl.connect %out1, %1 : !firrtl.uint<4>, !firrtl.uint<4>
}
firrtl.module @InlineMe1(in %in0: !firrtl.uint<4>, in %in1: !firrtl.uint<4>,
                   out %out0: !firrtl.uint<4>, out %out1: !firrtl.uint<4>)
        attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
  %a_in0, %a_in1, %a_out0, %a_out1 = firrtl.instance a @InlineMe0(in in0: !firrtl.uint<4>, in in1: !firrtl.uint<4>, out out0: !firrtl.uint<4>, out out1: !firrtl.uint<4>)
  firrtl.connect %a_in0, %in0 : !firrtl.uint<4>, !firrtl.uint<4>
  firrtl.connect %a_in1, %in1 : !firrtl.uint<4>, !firrtl.uint<4>
  firrtl.connect %out0, %a_out0 : !firrtl.uint<4>, !firrtl.uint<4>
  firrtl.connect %out1, %a_out1 : !firrtl.uint<4>, !firrtl.uint<4>
}
firrtl.module @TestConnections(in %in0: !firrtl.uint<4>, in %in1: !firrtl.uint<4>,
                   out %out0: !firrtl.uint<4>, out %out1: !firrtl.uint<4>) {
  %b_in0, %b_in1, %b_out0, %b_out1 = firrtl.instance b @InlineMe1(in in0: !firrtl.uint<4>, in in1: !firrtl.uint<4>, out out0: !firrtl.uint<4>, out out1: !firrtl.uint<4>)
  firrtl.connect %b_in0, %in0 : !firrtl.uint<4>, !firrtl.uint<4>
  firrtl.connect %b_in1, %in1 : !firrtl.uint<4>, !firrtl.uint<4>
  firrtl.connect %out0, %b_out0 : !firrtl.uint<4>, !firrtl.uint<4>
  firrtl.connect %out1, %b_out1 : !firrtl.uint<4>, !firrtl.uint<4>
}
}
// CHECK-LABEL: firrtl.module @TestConnections(in %in0: !firrtl.uint<4>, in %in1: !firrtl.uint<4>, out %out0: !firrtl.uint<4>, out %out1: !firrtl.uint<4>) {
// CHECK-NEXT:   %b_in0 = firrtl.wire  : !firrtl.uint<4>
// CHECK-NEXT:   %b_in1 = firrtl.wire  : !firrtl.uint<4>
// CHECK-NEXT:   %b_out0 = firrtl.wire  : !firrtl.uint<4>
// CHECK-NEXT:   %b_out1 = firrtl.wire  : !firrtl.uint<4>
// CHECK-NEXT:   %b_a_in0 = firrtl.wire  : !firrtl.uint<4>
// CHECK-NEXT:   %b_a_in1 = firrtl.wire  : !firrtl.uint<4>
// CHECK-NEXT:   %b_a_out0 = firrtl.wire  : !firrtl.uint<4>
// CHECK-NEXT:   %b_a_out1 = firrtl.wire  : !firrtl.uint<4>
// CHECK-NEXT:   %0 = firrtl.and %b_a_in0, %b_a_in1 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
// CHECK-NEXT:   firrtl.connect %b_a_out0, %0 : !firrtl.uint<4>
// CHECK-NEXT:   %1 = firrtl.and %b_a_in0, %b_a_in1 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
// CHECK-NEXT:   firrtl.connect %b_a_out1, %1 : !firrtl.uint<4>
// CHECK-NEXT:   firrtl.connect %b_a_in0, %b_in0 : !firrtl.uint<4>
// CHECK-NEXT:   firrtl.connect %b_a_in1, %b_in1 : !firrtl.uint<4>
// CHECK-NEXT:   firrtl.connect %b_out0, %b_a_out0 : !firrtl.uint<4>
// CHECK-NEXT:   firrtl.connect %b_out1, %b_a_out1 : !firrtl.uint<4>
// CHECK-NEXT:   firrtl.connect %b_in0, %in0 : !firrtl.uint<4>
// CHECK-NEXT:   firrtl.connect %b_in1, %in1 : !firrtl.uint<4>
// CHECK-NEXT:   firrtl.connect %out0, %b_out0 : !firrtl.uint<4>
// CHECK-NEXT:   firrtl.connect %out1, %b_out1 : !firrtl.uint<4>
// CHECK-NEXT: }


// This is testing that bundles with flip types are handled properly by the inliner.
firrtl.circuit "TestBulkConnections" {
firrtl.module @InlineMe0(in %in0: !firrtl.bundle<a: uint<4>, b flip: uint<4>>,
                         out %out0: !firrtl.bundle<a: uint<4>, b flip: uint<4>>)
        attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
  firrtl.connect %out0, %in0 : !firrtl.bundle<a: uint<4>, b flip: uint<4>>, !firrtl.bundle<a: uint<4>, b flip: uint<4>>
}
firrtl.module @TestBulkConnections(in %in0: !firrtl.bundle<a: uint<4>, b flip: uint<4>>,
                                   out %out0: !firrtl.bundle<a: uint<4>, b flip: uint<4>>) {
  %i_in0, %i_out0 = firrtl.instance i @InlineMe0(in in0: !firrtl.bundle<a: uint<4>, b flip: uint<4>>, out out0: !firrtl.bundle<a: uint<4>, b flip: uint<4>>)
  firrtl.connect %i_in0, %in0 : !firrtl.bundle<a: uint<4>, b flip: uint<4>>, !firrtl.bundle<a: uint<4>, b flip: uint<4>>
  firrtl.connect %out0, %i_out0 : !firrtl.bundle<a: uint<4>, b flip: uint<4>>, !firrtl.bundle<a: uint<4>, b flip: uint<4>>
// CHECK: %i_in0 = firrtl.wire  : !firrtl.bundle<a: uint<4>, b flip: uint<4>>
// CHECK: %i_out0 = firrtl.wire  : !firrtl.bundle<a: uint<4>, b flip: uint<4>>
// CHECK: firrtl.connect %i_out0, %i_in0 : !firrtl.bundle<a: uint<4>, b flip: uint<4>>
// CHECK: firrtl.connect %i_in0, %in0 : !firrtl.bundle<a: uint<4>, b flip: uint<4>>
// CHECK: firrtl.connect %out0, %i_out0 : !firrtl.bundle<a: uint<4>, b flip: uint<4>>
}
}

// Test that all operations with names are renamed.
firrtl.circuit "renaming" {
firrtl.module @renaming() {
  %0, %1, %2 = firrtl.instance myinst @declarations(in clock : !firrtl.clock, in u8 : !firrtl.uint<8>, in reset : !firrtl.asyncreset)
}
firrtl.module @declarations(in %clock : !firrtl.clock, in %u8 : !firrtl.uint<8>, in %reset : !firrtl.asyncreset) attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
  %c0_ui8 = firrtl.constant 0 : !firrtl.uint<8>
  // CHECK: %myinst_cmem = chirrtl.combmem : !chirrtl.cmemory<uint<8>, 8>
  %cmem = chirrtl.combmem : !chirrtl.cmemory<uint<8>, 8>
  // CHECK: %myinst_mem_read = firrtl.mem Undefined {depth = 1 : i64, name = "myinst_mem", portNames = ["read"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: sint<42>>
  %mem_read = firrtl.mem Undefined {depth = 1 : i64, name = "mem", portNames = ["read"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: sint<42>>
  // CHECK: %myinst_memoryport_data, %myinst_memoryport_port = chirrtl.memoryport Read %myinst_cmem {name = "myinst_memoryport"} : (!chirrtl.cmemory<uint<8>, 8>) -> (!firrtl.uint<8>, !chirrtl.cmemoryport)
  %memoryport_data, %memoryport_port = chirrtl.memoryport Read %cmem {name = "memoryport"} : (!chirrtl.cmemory<uint<8>, 8>) -> (!firrtl.uint<8>, !chirrtl.cmemoryport)
  chirrtl.memoryport.access %memoryport_port[%u8], %clock : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
  // CHECK: %myinst_node = firrtl.node %myinst_u8  : !firrtl.uint<8>
  %node = firrtl.node %u8 {name = "node"} : !firrtl.uint<8>
  // CHECK: %myinst_reg = firrtl.reg %myinst_clock : !firrtl.clock, !firrtl.uint<8>
  %reg = firrtl.reg %clock {name = "reg"} : !firrtl.clock, !firrtl.uint<8>
  // CHECK: %myinst_regreset = firrtl.regreset %myinst_clock, %myinst_reset, %c0_ui8 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<8>, !firrtl.uint<8>
  %regreset = firrtl.regreset %clock, %reset, %c0_ui8 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<8>, !firrtl.uint<8>
  // CHECK: %myinst_smem = chirrtl.seqmem Undefined : !chirrtl.cmemory<uint<8>, 8>
  %smem = chirrtl.seqmem Undefined : !chirrtl.cmemory<uint<8>, 8>
  // CHECK: %myinst_wire = firrtl.wire  : !firrtl.uint<1>
  %wire = firrtl.wire : !firrtl.uint<1>
  firrtl.when %wire : !firrtl.uint<1> {
    // CHECK:  %myinst_inwhen = firrtl.wire  : !firrtl.uint<1>
    %inwhen = firrtl.wire : !firrtl.uint<1>
  }
}

// Test that non-module operations should not be deleted.
firrtl.circuit "PreserveUnknownOps" {
firrtl.module @PreserveUnknownOps() { }
// CHECK: sv.verbatim "hello"
sv.verbatim "hello"
}

}

// Test NLA handling during inlining for situations involving NLAs where the NLA
// begins at the main module.
// There are four behaviors being tested:
//
//   1) @nla1: Targeting a module should be updated
//   2) @nla2: Targeting a component should be updated
//   3) @nla3: Targeting a module port should be updated
//   4) @nla4: Targeting an inlined module should be dropped
//   5) @nla5: NLAs targeting a component should promote to local annotations
//   6) @nla5: NLAs targeting a port should promote to local annotations
//
// CHECK-LABEL: firrtl.circuit "NLAInlining"
firrtl.circuit "NLAInlining" {
  // CHECK-NEXT: hw.hierpath private @nla1 [@NLAInlining::@bar, @Bar]
  // CHECK-NEXT: hw.hierpath private @nla2 [@NLAInlining::@bar, @Bar::@a]
  // CHECK-NEXT: hw.hierpath private @nla3 [@NLAInlining::@bar, @Bar::@port]
  // Retention keeps the localized paths as one-hop hierpaths; the
  // annotations still localize onto the ops.
  // CHECK-NEXT: hw.hierpath private @nla4 [@NLAInlining]
  // CHECK-NEXT: hw.hierpath private @nla5 [@NLAInlining::@b]
  // CHECK-NEXT: hw.hierpath private @nla6 [@NLAInlining::@port]
  hw.hierpath private @nla1 [@NLAInlining::@foo, @Foo::@bar, @Bar]
  hw.hierpath private @nla2 [@NLAInlining::@foo, @Foo::@bar, @Bar::@a]
  hw.hierpath private @nla3 [@NLAInlining::@foo, @Foo::@bar, @Bar::@port]
  hw.hierpath private @nla4 [@NLAInlining::@foo, @Foo]
  hw.hierpath private @nla5 [@NLAInlining::@foo, @Foo::@b]
  hw.hierpath private @nla6 [@NLAInlining::@foo, @Foo::@port]
  // CHECK-NEXT: firrtl.module private @Bar
  // CHECK-SAME: %port: {{.+}} sym @port [{circt.nonlocal = @nla3, class = "nla3"}]
  // CHECK-SAME: [{circt.nonlocal = @nla1, class = "nla1"}]
  firrtl.module private @Bar(
    in %port: !firrtl.uint<1> sym @port [{circt.nonlocal = @nla3, class = "nla3"}]
  ) attributes {annotations = [{circt.nonlocal = @nla1, class = "nla1"}]} {
    %a = firrtl.wire sym @a {annotations = [{circt.nonlocal = @nla2, class = "nla2"}]} : !firrtl.uint<1>
  }
  firrtl.module private @Foo(in %port: !firrtl.uint<1> sym @port [{circt.nonlocal = @nla6, class = "nla6"}]) attributes {annotations = [
  {class = "firrtl.passes.InlineAnnotation"}, {circt.nonlocal = @nla4, class = "nla4"}]} {
    %bar_port = firrtl.instance bar sym @bar @Bar(in port: !firrtl.uint<1>)
    %b = firrtl.wire sym @b {annotations = [{circt.nonlocal = @nla5, class = "nla5"}]} : !firrtl.uint<1>
  }
  // CHECK: firrtl.module @NLAInlining
  firrtl.module @NLAInlining() {
    %foo_port = firrtl.instance foo sym @foo @Foo(in port: !firrtl.uint<1>)
    // CHECK-NEXT: %foo_port = firrtl.wire {{.+}} [{class = "nla6"}]
    // CHECK-NEXT: firrtl.instance foo_bar {{.+}}
    // CHECK-NEXT: %foo_b = firrtl.wire {{.+}} [{class = "nla5"}]
  }
}

// Test NLA handling during inlining where the NLA does NOT start at the root.
// This checks that the NLA, on either a component or a port, is properly copied
// for each new instantiation.
//
// CHECK-LABEL: firrtl.circuit "NLAInliningNotMainRoot"
firrtl.circuit "NLAInliningNotMainRoot" {
  // CHECK-NEXT: hw.hierpath private @nla1 [@NLAInliningNotMainRoot::@baz, @Baz::@a]
  // CHECK-NEXT: hw.hierpath private @nla1_0 [@Foo::@baz, @Baz::@a]
  // CHECK-NEXT: hw.hierpath private @nla2 [@NLAInliningNotMainRoot::@baz, @Baz::@port]
  // CHECK-NEXT: hw.hierpath private @nla2_0 [@Foo::@baz, @Baz::@port]
  hw.hierpath private @nla1 [@Bar::@baz, @Baz::@a]
  hw.hierpath private @nla2 [@Bar::@baz, @Baz::@port]
  // CHECK-NEXT: firrtl.module private @Baz
  // CHECK-SAME: %port: {{.+}} [{circt.nonlocal = @nla2, class = "nla2"}, {circt.nonlocal = @nla2_0, class = "nla2"}]
  firrtl.module private @Baz(
    in %port: !firrtl.uint<1> sym @port [{circt.nonlocal = @nla2, class = "nla2"}]
  ) {
    // CHECK-NEXT: firrtl.wire {{.+}} [{circt.nonlocal = @nla1, class = "hello"}, {circt.nonlocal = @nla1_0, class = "hello"}]
    %a = firrtl.wire sym @a {annotations = [{circt.nonlocal = @nla1, class = "hello"}]} : !firrtl.uint<1>
  }
  firrtl.module private @Bar() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %baz_port = firrtl.instance baz sym @baz @Baz(in port: !firrtl.uint<1>)
  }
  // CHECK: firrtl.module private @Foo
  firrtl.module private @Foo() {
    // CHECK-NEXT: firrtl.instance bar_baz {{.+}}
    firrtl.instance bar @Bar()
  }
  // CHECK: firrtl.module @NLAInliningNotMainRoot
  firrtl.module @NLAInliningNotMainRoot() {
    firrtl.instance foo @Foo()
    // CHECK: firrtl.instance bar_baz {{.+}}
    firrtl.instance bar @Bar()
    %baz_port = firrtl.instance baz @Baz(in port: !firrtl.uint<1>)
  }
}

// Test NLA handling during flattening for situations where the root of an NLA
// is the flattened module or an ancestor of the flattened module.
// This is testing the following conditions:
//
//   1) @nla1: Targeting a reference should be updated.
//   2) @nla1: Targeting a port should be updated.
//   3) @nla3: Targeting a module should be dropped.
//   4) @nla4: Targeting a reference should be promoted to local.
//
// CHECK-LABEL: firrtl.circuit "NLAFlattening"
firrtl.circuit "NLAFlattening" {
  // CHECK-NEXT: hw.hierpath private @nla1 [@NLAFlattening::@foo, @Foo::@a]
  // CHECK-NEXT: hw.hierpath private @nla2 [@NLAFlattening::@foo, @Foo::@port]
  // Retention keeps @nla3/@nla4 (annotations localize onto the flattened ops).
  // CHECK-NEXT: hw.hierpath private @nla3 [@NLAFlattening::@foo, @Foo]
  // CHECK-NEXT: hw.hierpath private @nla4 [@Foo::@b]
  hw.hierpath private @nla1 [@NLAFlattening::@foo, @Foo::@bar, @Bar::@baz, @Baz::@a]
  hw.hierpath private @nla2 [@NLAFlattening::@foo, @Foo::@bar, @Bar::@baz, @Baz::@port]
  hw.hierpath private @nla3 [@NLAFlattening::@foo, @Foo::@bar, @Bar::@baz, @Baz]
  hw.hierpath private @nla4 [@Foo::@bar, @Bar::@b]
  // CHECK-NEXT: firrtl.module @Baz
  // CHECK-NOT: @nla
  // CHECK-NOT: nonlocal
  firrtl.module @Baz(
    in %port: !firrtl.uint<1> sym @port [{circt.nonlocal = @nla2, class = "nla2"}]
  ) attributes {annotations = [{circt.nonlocal = @nla3, class = "nla3"}]} {
    %a = firrtl.wire sym @a {annotations = [{circt.nonlocal = @nla1, class = "nla1"}]} : !firrtl.uint<1>
  }
  firrtl.module @Bar() {
    firrtl.instance baz sym @baz @Baz(in port: !firrtl.uint<1>)
    %b = firrtl.wire sym @b {annotations = [{circt.nonlocal = @nla4, class = "nla4"}]} : !firrtl.uint<1>
  }
  // CHECK: firrtl.module @Foo
  firrtl.module @Foo() attributes {annotations = [{class = "firrtl.transforms.FlattenAnnotation"}]} {
    firrtl.instance bar sym @bar @Bar()
    // CHECK-NEXT: %bar_baz_port = firrtl.wire sym @port {{.+}} [{circt.nonlocal = @nla2, class = "nla2"}]
    // CHECK-NEXT: %bar_baz_a = firrtl.wire {{.+}} [{circt.nonlocal = @nla1, class = "nla1"}]
    // CHECK-NEXT: %bar_b = firrtl.wire {{.+}} [{class = "nla4"}]
  }
  // CHECK: firrtl.module @NLAFlattening
  firrtl.module @NLAFlattening() {
    // CHECK-NEXT: firrtl.instance foo {{.+}}
    firrtl.instance foo sym @foo @Foo()
  }
}

// Test NLA handling during flattening when the NLA root is a child of the
// flattened module:
//
//   1) @nla1: component path localizes; retained as a one-hop hierpath.
//   2) @nla2: port path localizes; retained as a one-hop hierpath.
//   3) @nla3: component path survives, trim-equal contexts collapse into it.
//   4) @nla4: port path survives, trim-equal contexts collapse into it.
//
// CHECK-LABEL: firrtl.circuit "NLAFlatteningChildRoot"
firrtl.circuit "NLAFlatteningChildRoot" {
  // @Baz's only surviving instance is under the top; @Foo flattens away.
  // The top-rooted contexts are therefore trim-equal with the @Baz-rooted
  // @nla3/@nla4 and collapse into them -- one hierpath each, one annotation
  // each.
  // @nla1/@nla2 localize into the flattened @Foo copy; retention keeps them as
  // one-hop hierpaths (annotations localized onto the ops).
  // CHECK-NEXT: hw.hierpath private @nla1 [@Foo::@a]
  // CHECK-NEXT: hw.hierpath private @nla2 [@Foo::@Qux_port]
  // CHECK-NEXT: hw.hierpath private @nla3 [@Baz::@quz, @Quz::@b]
  // CHECK-NEXT: hw.hierpath private @nla4 [@Baz::@quz, @Quz::@Quz_port]
  // CHECK-NEXT: firrtl.module private @Quz
  hw.hierpath private @nla1 [@Bar::@qux, @Qux::@a]
  hw.hierpath private @nla2 [@Bar::@qux, @Qux::@Qux_port]
  hw.hierpath private @nla3 [@Baz::@quz, @Quz::@b]
  hw.hierpath private @nla4 [@Baz::@quz, @Quz::@Quz_port]
  // CHECK-SAME: in %port: {{.+}} [{circt.nonlocal = @nla4, class = "nla4"}]
  firrtl.module private @Quz(
    in %port: !firrtl.uint<1> sym @Quz_port [{circt.nonlocal = @nla4, class = "nla4"}]
  ) {
    // CHECK-NEXT: firrtl.wire {{.+}} [{circt.nonlocal = @nla3, class = "nla3"}]
    %b = firrtl.wire sym @b {annotations = [{circt.nonlocal = @nla3, class = "nla3"}]} : !firrtl.uint<1>
  }
  firrtl.module private @Qux(
    in %port: !firrtl.uint<1> sym @Qux_port [{circt.nonlocal = @nla2, class = "nla2"}]
  ) {
    %a = firrtl.wire sym @a {annotations = [{circt.nonlocal = @nla1, class = "nla1"}]} : !firrtl.uint<1>
  }
  // CHECK: firrtl.module private @Baz
  firrtl.module private @Baz() {
    // CHECK-NEXT: firrtl.instance {{.+}}
    firrtl.instance quz sym @quz @Quz(in port: !firrtl.uint<1>)
  }
  firrtl.module private @Bar() {
    firrtl.instance qux sym @qux @Qux(in port: !firrtl.uint<1>)
  }
  // CHECK: firrtl.module private @Foo
  firrtl.module private @Foo() attributes {annotations = [{class = "firrtl.transforms.FlattenAnnotation"}]} {
    // CHECK-NEXT: %bar_qux_port = firrtl.wire sym @Qux_port {{.+}} [{class = "nla2"}]
    // CHECK-NEXT: %bar_qux_a = firrtl.wire {{.+}} [{class = "nla1"}]
    // CHECK-NEXT: %baz_quz_port = firrtl.wire sym @Quz_port {{.+}} [{class = "nla4"}]
    // CHECK-NEXT: %baz_quz_b = firrtl.wire {{.+}} [{class = "nla3"}]
    firrtl.instance bar @Bar()
    firrtl.instance baz @Baz()
  }
  firrtl.module @NLAFlatteningChildRoot() {
    firrtl.instance foo @Foo()
    firrtl.instance baz @Baz()
  }
}

// Test that symbols are uniqued due to collisions.
//
//   1) An inlined symbol is uniqued.
//   2) An inlined symbol that participates in an NLA is uniqued
//
// CHECK-LABEL: CollidingSymbols
firrtl.circuit "CollidingSymbols" {
  // CHECK-NEXT: hw.hierpath private @nla1 [@CollidingSymbols::@[[FoobarSym:[_a-zA-Z0-9]+]], @Bar]
  hw.hierpath private @nla1 [@CollidingSymbols::@foo, @Foo::@bar, @Bar]
  firrtl.module @Bar() attributes {annotations = [{circt.nonlocal = @nla1, class = "nla1"}]} {}
  firrtl.module @Foo() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %b = firrtl.wire sym @b : !firrtl.uint<1>
    firrtl.instance bar sym @bar @Bar()
  }
  // CHECK:      firrtl.module @CollidingSymbols
  // CHECK-NEXT:   firrtl.wire sym @[[inlinedSymbol:[_a-zA-Z0-9]+]]
  // CHECK-NEXT:   firrtl.instance foo_bar sym @[[FoobarSym]]
  // CHECK-NOT:    firrtl.wire sym @[[inlinedSymbol]]
  // CHECK-NOT:    firrtl.wire sym @[[FoobarSym]]
  firrtl.module @CollidingSymbols() {
    firrtl.instance foo sym @foo @Foo()
    %collision_b = firrtl.wire sym @b : !firrtl.uint<1>
    %collision_bar = firrtl.wire sym @bar : !firrtl.uint<1>
  }
}

// Test that port symbols are uniqued due to a collision.
//
//   1) An inlined port is uniqued and the NLA is updated.
//
// CHECK-LABEL: CollidingSymbolsPort
firrtl.circuit "CollidingSymbolsPort" {
  // CHECK-NEXT: hw.hierpath private @nla1 [@CollidingSymbolsPort::@foo, @Foo::@[[BarbSym:[_a-zA-Z0-9]+]]]
  hw.hierpath private @nla1 [@CollidingSymbolsPort::@foo, @Foo::@bar, @Bar::@b]
  // CHECK-NOT: firrtl.module private @Bar
  firrtl.module private @Bar(
    in %b: !firrtl.uint<1> sym @b [{circt.nonlocal = @nla1, class = "nla1"}]
  ) attributes {annotations = [
    {class = "firrtl.passes.InlineAnnotation"}
  ]} {}
  // CHECK-NEXT: firrtl.module private @Foo
  firrtl.module private @Foo() {
    // CHECK-NEXT: firrtl.wire sym @[[BarbSym]] {annotations = [{circt.nonlocal = @nla1, class = "nla1"}]}
    firrtl.instance bar sym @bar @Bar(in b: !firrtl.uint<1>)
    // CHECK-NEXT: firrtl.wire sym @b
    %colliding_b = firrtl.wire sym @b : !firrtl.uint<1>
  }
  firrtl.module @CollidingSymbolsPort() {
    firrtl.instance foo sym @foo @Foo()
  }
}

// Test that colliding symbols that originate from the root of an inlined module
// are properly duplicated and renamed.
//
//   1) The symbol @baz becomes @baz_0 in the top module (as @baz is used)
//   2) The symbol @baz becomes @baz_1 in Foo (as @baz and @baz_0 are both used)
//
// CHECK-LABEL: firrtl.circuit "CollidingSymbolsReTop"
firrtl.circuit "CollidingSymbolsReTop" {
  // CHECK-NEXT: hw.hierpath private @nla1 [@Bar::@baz, @Baz::@a]
  // CHECK-NEXT: hw.hierpath private @nla1_0 [@CollidingSymbolsReTop::@[[TopbazSym:[_a-zA-Z0-9]+]], @Baz::@a]
  // CHECK-NEXT: hw.hierpath private @nla1_1 [@Foo::@[[FoobazSym:[_a-zA-Z0-9]+]], @Baz::@a]
  // CHECK-NEXT: firrtl.module @Baz
  hw.hierpath private @nla1 [@Bar::@baz, @Baz::@a]
  firrtl.module @Baz() {
    // CHECK-NEXT: firrtl.wire {{.+}} [{circt.nonlocal = @nla1, class = "hello"}, {circt.nonlocal = @nla1_0, class = "hello"}, {circt.nonlocal = @nla1_1, class = "hello"}]
    %a = firrtl.wire sym @a {annotations = [{circt.nonlocal = @nla1, class = "hello"}]} : !firrtl.uint<1>
  }
  firrtl.module @Bar() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance baz sym @baz @Baz()
  }
  // CHECK: firrtl.module @Foo
  firrtl.module @Foo() {
    // CHECK-NEXT: firrtl.instance bar_baz sym @[[FoobazSym]] {{.+}}
    firrtl.instance bar @Bar()
    %colliding_baz = firrtl.wire sym @baz : !firrtl.uint<1>
    %colliding_baz_0 = firrtl.wire sym @baz_0 : !firrtl.uint<1>
  }
  // CHECK: firrtl.module @CollidingSymbolsReTop
  firrtl.module @CollidingSymbolsReTop() {
    firrtl.instance foo @Foo()
    // CHECK: firrtl.instance bar_baz sym @[[TopbazSym]]{{.+}}
    firrtl.instance bar @Bar()
    firrtl.instance baz @Baz()
    %colliding_baz = firrtl.wire sym @baz : !firrtl.uint<1>
  }
}

// Test that when inlining two instances of a module and the port names collide,
// that the NLA is properly updated.
// Specifically in this test case, the second instance inlined should be
// renamed, and it should *not* update the NLA.
// CHECK-LABEL: firrtl.circuit "CollidingSymbolsNLAFixup"
firrtl.circuit "CollidingSymbolsNLAFixup" {
  // CHECK: hw.hierpath private @nla0 [@Foo::@bar, @Bar::@io]
  hw.hierpath private @nla0 [@Foo::@bar, @Bar::@baz0, @Baz::@io]

  // CHECK: hw.hierpath private @nla1 [@Foo::@bar, @Bar::@w]
  hw.hierpath private @nla1 [@Foo::@bar, @Bar::@baz0, @Baz::@w]

  firrtl.module @Baz(out %io: !firrtl.uint<1> sym @io [{circt.nonlocal = @nla0, class = "test"}])
       attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %w = firrtl.wire sym @w {annotations = [{circt.nonlocal = @nla1, class = "test"}]} : !firrtl.uint<1>
  }

  // CHECK: firrtl.module @Bar()
  firrtl.module @Bar() {
    // CHECK: %baz0_io = firrtl.wire sym @io  {annotations = [{circt.nonlocal = @nla0, class = "test"}]}
    // CHECK: %baz0_w = firrtl.wire sym @w  {annotations = [{circt.nonlocal = @nla1, class = "test"}]}
    %0 = firrtl.instance baz0 sym @baz0 @Baz(out io : !firrtl.uint<1>)

    // CHECK: %baz1_io = firrtl.wire sym @io_0
    // CHECK: %baz1_w = firrtl.wire sym @w
    %1 = firrtl.instance baz1 sym @baz1 @Baz(out io : !firrtl.uint<1>)
  }

  firrtl.module @Foo() {
    firrtl.instance bar sym @bar @Bar()
  }

  firrtl.module @CollidingSymbolsNLAFixup() {
    firrtl.instance system sym @system @Foo()
  }
}

// Test that anything with a "name" will be renamed, even things that FIRRTL
// Dialect doesn't understand.
//
// CHECK-LABEL: firrtl.circuit "RenameAnything"
firrtl.circuit "RenameAnything" {
  firrtl.module private @Foo() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    "some_unknown_dialect.op"() { name = "world" } : () -> ()
  }
  // CHECK-NEXT: firrtl.module @RenameAnything
  firrtl.module @RenameAnything() {
    // CHECK-NEXT: "some_unknown_dialect.op"(){{.+}}name = "hello_world"
    firrtl.instance hello @Foo()
  }
}

// Test that when an op is inlined into two locations and an annotation on it
// becomes local, that the local annotation is only copied to the clone that
// corresponds to the original NLA path.
// CHECK-LABEL: firrtl.circuit "AnnotationSplit0"
firrtl.circuit "AnnotationSplit0" {
hw.hierpath private @nla_5560 [@Bar0::@leaf, @Leaf::@w]
hw.hierpath private @nla_5561 [@Bar1::@leaf, @Leaf::@w]
firrtl.module @Leaf() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
  %w = firrtl.wire sym @w {annotations = [
    {circt.nonlocal = @nla_5560, class = "test0"},
    {circt.nonlocal = @nla_5561, class = "test1"}]} : !firrtl.uint<8>
}
// CHECK: firrtl.module @Bar0
firrtl.module @Bar0() {
  // CHECK: %leaf_w = firrtl.wire sym @w  {annotations = [{class = "test0"}]}
  firrtl.instance leaf sym @leaf  @Leaf()
}
// CHECK: firrtl.module @Bar1
firrtl.module @Bar1() {
  // CHECK: %leaf_w = firrtl.wire sym @w  {annotations = [{class = "test1"}]}
  firrtl.instance leaf sym @leaf  @Leaf()
}
firrtl.module @AnnotationSplit0() {
  firrtl.instance bar0 @Bar0()
  firrtl.instance bar1 @Bar1()
}
}

// Test that when an operation is inlined into two locations and an annotation
// on it should only be copied to a specific clone.
// This differs from the test above in that the annotation does not become a
// regular local annotation.
// CHECK-LABEL: firrtl.circuit "AnnotationSplit1"
firrtl.circuit "AnnotationSplit1" {
hw.hierpath private @nla_5560 [@AnnotationSplit1::@bar0, @Bar0::@leaf, @Leaf::@w]
hw.hierpath private @nla_5561 [@AnnotationSplit1::@bar1, @Bar1::@leaf, @Leaf::@w]
firrtl.module @Leaf() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
  %w = firrtl.wire sym @w {annotations = [
    {circt.nonlocal = @nla_5560, class = "test0"},
    {circt.nonlocal = @nla_5561, class = "test1"}]} : !firrtl.uint<8>
}
// CHECK: firrtl.module @Bar0
firrtl.module @Bar0() {
  // CHECK: %leaf_w = firrtl.wire sym @w  {annotations = [{circt.nonlocal = @nla_5560, class = "test0"}]}
  firrtl.instance leaf sym @leaf  @Leaf()
}
// CHECK: firrtl.module @Bar1
firrtl.module @Bar1() {
  // CHECK: %leaf_w = firrtl.wire sym @w  {annotations = [{circt.nonlocal = @nla_5561, class = "test1"}]}
  firrtl.instance leaf sym @leaf  @Leaf()
}
firrtl.module @AnnotationSplit1() {
  firrtl.instance bar0 sym @bar0 @Bar0()
  firrtl.instance bar1 sym @bar1 @Bar1()
}
}

// Test Rename of InstanceOps.
// https://github.com/llvm/circt/issues/3307
firrtl.circuit "Inline"  {
  // CHECK: firrtl.circuit "Inline"
  hw.hierpath private @nla_2 [@Inline::@bar, @Bar::@i]
  hw.hierpath private @nla_1 [@Inline::@foo, @Foo::@bar, @Bar::@i]
  // CHECK:   hw.hierpath private @nla_2 [@Inline::@bar, @Bar::@i]
  // CHECK:   hw.hierpath private @nla_1 [@Inline::@[[bar_0:.+]], @Bar::@i]
  firrtl.module @Inline(in %i: !firrtl.uint<1>, out %o: !firrtl.uint<1>) {
    %foo_i, %foo_o = firrtl.instance foo sym @foo  @Foo(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    // CHECK:  = firrtl.instance foo_bar sym @[[bar_0]]  @Bar(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    %bar_i, %bar_o = firrtl.instance bar sym @bar  @Bar(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    // CHECK:  = firrtl.instance bar sym @bar  @Bar(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    firrtl.matchingconnect %foo_i, %bar_i : !firrtl.uint<1>
    firrtl.matchingconnect %bar_i, %i : !firrtl.uint<1>
    firrtl.matchingconnect %o, %foo_o : !firrtl.uint<1>
  }
  firrtl.module private @Bar(in %i: !firrtl.uint<1> sym @i [{circt.nonlocal = @nla_1, class = "test_1"}, {circt.nonlocal = @nla_2, class = "test_2"}], out %o: !firrtl.uint<1>) {
    firrtl.matchingconnect %o, %i : !firrtl.uint<1>
  }
  firrtl.module private @Foo(in %i: !firrtl.uint<1>, out %o: !firrtl.uint<1>) attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %bar_i, %bar_o = firrtl.instance bar sym @bar  @Bar(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    firrtl.matchingconnect %bar_i, %i : !firrtl.uint<1>
    firrtl.matchingconnect %o, %bar_o : !firrtl.uint<1>
  }
}

firrtl.circuit "Inline2"  {
  // CHECK-LABEL: firrtl.circuit "Inline2"
  hw.hierpath private @nla_1 [@Inline2::@foo, @Foo::@bar, @Bar::@i]
  // CHECK:  hw.hierpath private @nla_1 [@Inline2::@[[bar_0:.+]], @Bar::@i]
  firrtl.module @Inline2(in %i: !firrtl.uint<1>, out %o: !firrtl.uint<1>) {
    %foo_i, %foo_o = firrtl.instance foo sym @foo  @Foo(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    %bar = firrtl.wire sym @bar  : !firrtl.uint<1>
    firrtl.matchingconnect %foo_i, %bar : !firrtl.uint<1>
    firrtl.matchingconnect %bar, %i : !firrtl.uint<1>
    firrtl.matchingconnect %o, %foo_o : !firrtl.uint<1>
  }
  firrtl.module private @Bar(in %i: !firrtl.uint<1> sym @i [{circt.nonlocal = @nla_1, class = "testing"}], out %o: !firrtl.uint<1>) {
    firrtl.matchingconnect %o, %i : !firrtl.uint<1>
  }
  firrtl.module private @Foo(in %i: !firrtl.uint<1>, out %o: !firrtl.uint<1>) attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %bar_i, %bar_o = firrtl.instance bar sym @bar  @Bar(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    // CHECK:  = firrtl.instance foo_bar sym @[[bar_0]]  @Bar(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    firrtl.matchingconnect %bar_i, %i : !firrtl.uint<1>
    firrtl.matchingconnect %o, %bar_o : !firrtl.uint<1>
  }
}

// CHECK-LABEL: firrtl.circuit "Issue3334"
firrtl.circuit "Issue3334" {
  // CHECK: hw.hierpath private @path_component_old
  // CHECK: hw.hierpath private @path_component_new
  hw.hierpath private @path_component_old [@Issue3334::@foo, @Foo::@bar1, @Bar::@b]
  hw.hierpath private @path_component_new [@Issue3334::@foo, @Foo::@bar1, @Bar]
  firrtl.module private @Bar() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %b = firrtl.wire sym @b {annotations = [
      {circt.nonlocal = @path_component_old, "path_component_old"},
      {circt.nonlocal = @path_component_new, "path_component_new"}
    ]} : !firrtl.uint<1>
  }
  firrtl.module private @Foo() {
    firrtl.instance bar1 sym @bar1 @Bar()
    firrtl.instance bar2 sym @bar2 @Bar()
  }
  firrtl.module @Issue3334() {
    firrtl.instance foo sym @foo @Foo()
  }
}

// CHECK-LABEL: firrtl.circuit "Issue3334_flatten"
firrtl.circuit "Issue3334_flatten" {
  // CHECK: hw.hierpath private @path_component_old
  // CHECK: hw.hierpath private @path_component_new
  hw.hierpath private @path_component_old [@Issue3334_flatten::@foo, @Foo::@bar1, @Bar::@b]
  hw.hierpath private @path_component_new [@Issue3334_flatten::@foo, @Foo::@bar1, @Bar]
  firrtl.module private @Bar() {
    %b = firrtl.wire sym @b {annotations = [
      {circt.nonlocal = @path_component_old, "path_component_old"},
      {circt.nonlocal = @path_component_new, "path_component_new"}
    ]} : !firrtl.uint<1>
  }
  firrtl.module private @Foo() attributes {annotations = [{class = "firrtl.transforms.FlattenAnnotation"}]} {
    firrtl.instance bar1 sym @bar1 @Bar()
    firrtl.instance bar2 sym @bar2 @Bar()
  }
  firrtl.module @Issue3334_flatten() {
    firrtl.instance foo sym @foo @Foo()
  }
}

firrtl.circuit "instNameRename"  {
  hw.hierpath private @nla_5560 [@instNameRename::@bar0, @Bar0::@w, @Bar2::@w, @Bar1]
  // CHECK:  hw.hierpath private @nla_5560 [@instNameRename::@[[w_1:.+]], @Bar2::@w, @Bar1]
  hw.hierpath private @nla_5560_1 [@instNameRename::@bar1, @Bar0::@w, @Bar2::@w, @Bar1]
  // CHECK:  hw.hierpath private @nla_5560_1 [@instNameRename::@[[w_2:.+]], @Bar2::@w, @Bar1]
  firrtl.module @Bar1() {
    %w = firrtl.wire   {annotations = [{circt.nonlocal = @nla_5560, class = "test0"}, {circt.nonlocal = @nla_5560_1, class = "test1"}]} : !firrtl.uint<8>
  }
  firrtl.module @Bar2() {
    firrtl.instance leaf sym @w  @Bar1()
  }
  firrtl.module @Bar0() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance leaf sym @w  @Bar2()
  }
  firrtl.module @instNameRename() {
    firrtl.instance no sym @no  @Bar0()
    firrtl.instance bar0 sym @bar0  @Bar0()
    firrtl.instance bar1 sym @bar1  @Bar0()
    // CHECK:  firrtl.instance bar0_leaf sym @[[w_1]]  @Bar2()
    // CHECK:  firrtl.instance bar1_leaf sym @[[w_2]]  @Bar2()
    %w = firrtl.wire sym @w   : !firrtl.uint<8>
  }
}

// This test checks for context sensitive Hierpath update.
// The following inlining causes 4 instances of @Baz being added to @Foo1,
// but only two of them should have valid HierPathOps.
firrtl.circuit "CollidingSymbolsMultiInline" {

  hw.hierpath private @nla1 [@Foo1::@bar1, @Foo2::@bar, @Foo::@bar, @Bar::@w, @Baz::@w]
  // CHECK: hw.hierpath private @nla1 [@Foo1::@w_0, @Baz::@w]
  hw.hierpath private @nla2 [@Foo1::@bar2, @Foo2::@bar1, @Foo::@bar, @Bar::@w, @Baz::@w]
  // CHECK:  hw.hierpath private @nla2 [@Foo1::@w_7, @Baz::@w]

  firrtl.module @Baz(out %io: !firrtl.uint<1> )
        {
    %w = firrtl.wire sym @w {annotations = [{circt.nonlocal = @nla1, class = "test"}, {circt.nonlocal = @nla2, class = "test"}]} : !firrtl.uint<1>
  }

  firrtl.module @Bar() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]}{
    %0 = firrtl.instance baz0 sym @w    @Baz(out io : !firrtl.uint<1>)
  }

  firrtl.module @Foo() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]}{
    firrtl.instance bar sym @bar @Bar()
    %w = firrtl.wire sym @w : !firrtl.uint<1>
  }

  firrtl.module @Foo2() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]}{
    firrtl.instance bar sym @bar @Foo()
    firrtl.instance bar sym @bar1 @Foo()
    %w = firrtl.wire sym @w : !firrtl.uint<1>
  }

  firrtl.module @Foo1() {
    firrtl.instance bar sym @bar1 @Foo2()
    firrtl.instance bar sym @bar2 @Foo2()
    %w = firrtl.wire sym @bar : !firrtl.uint<1>
    %w1 = firrtl.wire sym @w : !firrtl.uint<1>
    // CHECK:  %bar_bar_bar_baz0_io = firrtl.instance bar_bar_bar_baz0 sym @w_0  @Baz(out io: !firrtl.uint<1>)
    // CHECK:  %bar_bar_bar_baz0_io_0 = firrtl.instance bar_bar_bar_baz0 sym @w_2  @Baz(out io: !firrtl.uint<1>)
    // CHECK:  %bar_bar_bar_baz0_io_2 = firrtl.instance bar_bar_bar_baz0 sym @w_5  @Baz(out io: !firrtl.uint<1>)
    // CHECK:  %bar_bar_bar_baz0_io_4 = firrtl.instance bar_bar_bar_baz0 sym @w_7  @Baz(out io: !firrtl.uint<1>)
  }

  firrtl.module @CollidingSymbolsMultiInline() {
    firrtl.instance system sym @system @Foo1()
  }
}

// CHECK-LABEL: firrtl.circuit "TrackInliningInDebugInfo"
firrtl.circuit "TrackInliningInDebugInfo" {
  // CHECK: firrtl.module @TrackInliningInDebugInfo
  firrtl.module @TrackInliningInDebugInfo() {
    // CHECK: [[SCOPE_FOO:%.+]] = dbg.scope "foo", "Foo"
    // CHECK: [[SCOPE_BAR:%.+]] = dbg.scope "bar", "Bar" scope [[SCOPE_FOO]]
    // CHECK: dbg.variable "a", {{%.+}} scope [[SCOPE_BAR]]
    // CHECK: [[SCOPE_IMPL:%.+]] = dbg.scope "impl", "Bugu" scope [[SCOPE_BAR]]
    // CHECK: dbg.variable "b", {{%.+}} scope [[SCOPE_IMPL]]
    firrtl.instance foo @Foo()
  }
  // CHECK-NOT: @Foo
  firrtl.module private @Foo() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance bar @Bar()
  }
  // CHECK-NOT: @Bar
  firrtl.module private @Bar() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %wire = firrtl.wire : !firrtl.uint<1>
    dbg.variable "a", %wire : !firrtl.uint<1>
    firrtl.when %wire : !firrtl.uint<1> {
      %0 = dbg.scope "impl", "Bugu"
      dbg.variable "b", %wire scope %0 : !firrtl.uint<1>
    }
  }
}

// CHECK-LABEL: firrtl.circuit "TrackFlatteningInDebugInfo"
firrtl.circuit "TrackFlatteningInDebugInfo" {
  // CHECK: firrtl.module @TrackFlatteningInDebugInfo
  firrtl.module @TrackFlatteningInDebugInfo() attributes {annotations = [{class = "firrtl.transforms.FlattenAnnotation"}]} {
    // CHECK: [[SCOPE_FOO:%.+]] = dbg.scope "foo", "Foo"
    // CHECK: [[SCOPE_BAR:%.+]] = dbg.scope "bar", "Bar" scope [[SCOPE_FOO]]
    // CHECK: dbg.variable "a", {{%.+}} scope [[SCOPE_BAR]]
    // CHECK: [[SCOPE_IMPL:%.+]] = dbg.scope "impl", "Bugu" scope [[SCOPE_BAR]]
    // CHECK: dbg.variable "b", {{%.+}} scope [[SCOPE_IMPL]]
    firrtl.instance foo @Foo()
  }
  // CHECK-NOT: @Foo
  firrtl.module private @Foo() {
    firrtl.instance bar @Bar()
  }
  // CHECK-NOT: @Bar
  firrtl.module private @Bar() {
    %wire = firrtl.wire : !firrtl.uint<1>
    dbg.variable "a", %wire : !firrtl.uint<1>
    firrtl.when %wire : !firrtl.uint<1> {
      %0 = dbg.scope "impl", "Bugu"
      dbg.variable "b", %wire scope %0 : !firrtl.uint<1>
    }
  }
}

// -----

// Test proper hierarchical inlining of RefType
// CHECK-LABEL: firrtl.circuit "HierarchicalRefType" {
firrtl.circuit "HierarchicalRefType" {
  firrtl.module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]}{
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %1 = firrtl.ref.send %zero : !firrtl.uint<1>
    firrtl.ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  // CHECK: module @Bar(
  firrtl.module @Bar(out %_a: !firrtl.probe<uint<1>>) attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]}{
    %xmr   = firrtl.instance bar sym @barXMR @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    firrtl.ref.define %_a, %xmr   : !firrtl.probe<uint<1>>
    // CHECK:      %bar__a = firrtl.wire : !firrtl.probe<uint<1>>
    // CHECK-NEXT: %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK-NEXT: %0 = firrtl.ref.send %c0_ui1 : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.ref.define %bar__a, %0 : !firrtl.probe<uint<1>>
    // CHECK-NEXT: firrtl.ref.define %_a, %bar__a : !firrtl.probe<uint<1>>
  }
  // CHECK: module @HierarchicalRefType(
  firrtl.module @HierarchicalRefType() {
    %bar_a = firrtl.instance bar sym @bar  @Bar(out _a: !firrtl.probe<uint<1>>)
    %a = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.probe<uint<1>>
    firrtl.matchingconnect %a, %0 : !firrtl.uint<1>
    // CHECK:       %bar__a = firrtl.wire : !firrtl.probe<uint<1>>
    // CHECK-NEXT:  %bar_bar__a = firrtl.wire : !firrtl.probe<uint<1>>
    // CHECK-NEXT:  %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK-NEXT:  %0 = firrtl.ref.send %c0_ui1 : !firrtl.uint<1>
    // CHECK-NEXT:  firrtl.ref.define %bar_bar__a, %0 : !firrtl.probe<uint<1>>
    // CHECK-NEXT:  firrtl.ref.define %bar__a, %bar_bar__a : !firrtl.probe<uint<1>>
    // CHECK-NEXT:  %a = firrtl.wire : !firrtl.uint<1>
    // CHECK-NEXT:  %1 = firrtl.ref.resolve %bar__a : !firrtl.probe<uint<1>>
    // CHECK-NEXT:  firrtl.matchingconnect %a, %1 : !firrtl.uint<1>
  }
}

// -----

// Test proper inlining of RefSend to Ports of RefType
// CHECK-LABEL: firrtl.circuit "InlineRefSend" {
firrtl.circuit "InlineRefSend" {
  firrtl.module @XmrSrcMod(in %pa: !firrtl.uint<1>, out %_a: !firrtl.probe<uint<1>>) attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]}{
    %1 = firrtl.ref.send %pa : !firrtl.uint<1>
    firrtl.ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  firrtl.module @Bar(out %_a: !firrtl.probe<uint<1>>)  attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]}{
    %pa, %xmr   = firrtl.instance bar sym @barXMR @XmrSrcMod(in pa: !firrtl.uint<1>, out _a: !firrtl.probe<uint<1>>)
    // CHECK:      %bar_pa = firrtl.wire : !firrtl.uint<1>
    // CHECK-NEXT: %bar__a = firrtl.wire : !firrtl.probe<uint<1>>
    // CHECK-NEXT: %0 = firrtl.ref.send %bar_pa : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.ref.define %bar__a, %0 : !firrtl.probe<uint<1>>
    // CHECK-NEXT: firrtl.ref.define %_a, %bar__a : !firrtl.probe<uint<1>>
    firrtl.ref.define %_a, %xmr   : !firrtl.probe<uint<1>>
  }
  firrtl.module @InlineRefSend() {
    %bar_a = firrtl.instance bar sym @bar  @Bar(out _a: !firrtl.probe<uint<1>>)
    %a = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.probe<uint<1>>
    firrtl.matchingconnect %a, %0 : !firrtl.uint<1>
    // CHECK:      %bar__a = firrtl.wire : !firrtl.probe<uint<1>>
    // CHECK-NEXT: %bar_bar_pa = firrtl.wire : !firrtl.uint<1>
    // CHECK-NEXT: %bar_bar__a = firrtl.wire : !firrtl.probe<uint<1>>
    // CHECK-NEXT: %0 = firrtl.ref.send %bar_bar_pa : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.ref.define %bar_bar__a, %0 : !firrtl.probe<uint<1>>
    // CHECK-NEXT: firrtl.ref.define %bar__a, %bar_bar__a : !firrtl.probe<uint<1>>
    // CHECK-NEXT: %a = firrtl.wire : !firrtl.uint<1>
    // CHECK-NEXT: %1 = firrtl.ref.resolve %bar__a : !firrtl.probe<uint<1>>
  }
}

// -----

// Test for multiple readers and multiple instances of RefType
// CHECK-LABEL: firrtl.circuit "MultReadInstRefType" {
firrtl.circuit "MultReadInstRefType" {
  firrtl.module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) {
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    %1 = firrtl.ref.send %zero : !firrtl.uint<1>
    firrtl.ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  firrtl.module @Foo(out %_a: !firrtl.probe<uint<1>>) {
    %xmr   = firrtl.instance bar sym @fooXMR @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    firrtl.ref.define %_a, %xmr   : !firrtl.probe<uint<1>>
    %0 = firrtl.ref.resolve %xmr   : !firrtl.probe<uint<1>>
    %a = firrtl.wire : !firrtl.uint<1>
    firrtl.matchingconnect %a, %0 : !firrtl.uint<1>
  }
  firrtl.module @Bar(out %_a: !firrtl.probe<uint<1>>) {
    %xmr   = firrtl.instance bar sym @barXMR @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    firrtl.ref.define %_a, %xmr   : !firrtl.probe<uint<1>>
    %0 = firrtl.ref.resolve %xmr   : !firrtl.probe<uint<1>>
    %a = firrtl.wire : !firrtl.uint<1>
    firrtl.matchingconnect %a, %0 : !firrtl.uint<1>
  }
  firrtl.module @MultReadInstRefType() attributes {annotations = [{class = "firrtl.transforms.FlattenAnnotation"}]}{
    %bar_a = firrtl.instance bar sym @bar  @Bar(out _a: !firrtl.probe<uint<1>>)
    // CHECK:      %bar__a = firrtl.wire : !firrtl.probe<uint<1>>
    // CHECK-NEXT: %bar_bar__a = firrtl.wire : !firrtl.probe<uint<1>>
    // CHECK-NEXT: %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK-NEXT: %0 = firrtl.ref.send %c0_ui1 : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.ref.define %bar_bar__a, %0 : !firrtl.probe<uint<1>>
    // CHECK-NEXT: firrtl.ref.define %bar__a, %bar_bar__a : !firrtl.probe<uint<1>>
    // CHECK-NEXT: %1 = firrtl.ref.resolve %bar_bar__a : !firrtl.probe<uint<1>>
    // CHECK-NEXT: %bar_a = firrtl.wire : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.matchingconnect %bar_a, %1 : !firrtl.uint<1>
    %foo_a = firrtl.instance foo sym @foo @Foo(out _a: !firrtl.probe<uint<1>>)
    // CHECK-NEXT: %foo__a = firrtl.wire : !firrtl.probe<uint<1>>
    // CHECK-NEXT: %foo_bar__a = firrtl.wire : !firrtl.probe<uint<1>>
    // CHECK-NEXT: %c0_ui1_0 = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK-NEXT: %2 = firrtl.ref.send %c0_ui1_0 : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.ref.define %foo_bar__a, %2 : !firrtl.probe<uint<1>>
    // CHECK-NEXT: firrtl.ref.define %foo__a, %foo_bar__a : !firrtl.probe<uint<1>>
    // CHECK-NEXT: %3 = firrtl.ref.resolve %foo_bar__a : !firrtl.probe<uint<1>>
    // CHECK-NEXT: %foo_a = firrtl.wire : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.matchingconnect %foo_a, %3 : !firrtl.uint<1>
    %xmr_a = firrtl.instance xmr sym @xmr @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    // CHECK-NEXT: %xmr__a = firrtl.wire : !firrtl.probe<uint<1>>
    // CHECK-NEXT: %c0_ui1_1 = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK-NEXT: %4 = firrtl.ref.send %c0_ui1_1 : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.ref.define %xmr__a, %4 : !firrtl.probe<uint<1>>
    // CHECK-NEXT: %a = firrtl.wire : !firrtl.uint<1>
    // CHECK-NEXT: %b = firrtl.wire : !firrtl.uint<1>
    // CHECK-NEXT: %c = firrtl.wire : !firrtl.uint<1>
    %a = firrtl.wire : !firrtl.uint<1>
    %b = firrtl.wire : !firrtl.uint<1>
    %c = firrtl.wire : !firrtl.uint<1>
    %0 = firrtl.ref.resolve %bar_a : !firrtl.probe<uint<1>>
    %1 = firrtl.ref.resolve %foo_a : !firrtl.probe<uint<1>>
    %2 = firrtl.ref.resolve %xmr_a : !firrtl.probe<uint<1>>
    // CHECK-NEXT: %5 = firrtl.ref.resolve %bar__a : !firrtl.probe<uint<1>>
    // CHECK-NEXT: %6 = firrtl.ref.resolve %foo__a : !firrtl.probe<uint<1>>
    // CHECK-NEXT: %7 = firrtl.ref.resolve %xmr__a : !firrtl.probe<uint<1>>
    firrtl.matchingconnect %a, %0 : !firrtl.uint<1>
    firrtl.matchingconnect %b, %1 : !firrtl.uint<1>
    firrtl.matchingconnect %c, %2 : !firrtl.uint<1>
    // CHECK:  firrtl.matchingconnect %a, %5 : !firrtl.uint<1>
    // CHECK:  firrtl.matchingconnect %b, %6 : !firrtl.uint<1>
    // CHECK:  firrtl.matchingconnect %c, %7 : !firrtl.uint<1>
  }
}

// -----

// PR #4882 fixes a bug, which was producing invalid NLAs.
// error: 'hw.hierpath' op  module: "instNameRename" does not contain any instance with symbol: "w"
// Due to coincidental name collisions, renaming was not updating the actual hierpath.
firrtl.circuit "Bug4882Rename"  {
  hw.hierpath private @nla_5560 [@Bug4882Rename::@w, @Bar2::@x]
  firrtl.module private @Bar2() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]}{
    %x = firrtl.wire sym @x  {annotations = [{circt.nonlocal = @nla_5560, class = "test0"}]} : !firrtl.uint<8>
  }
  firrtl.module private @Bar1() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]}{
    firrtl.instance bar3 sym @w  @Bar3()
  }
  firrtl.module private @Bar3()  {
    %w = firrtl.wire sym @w1   : !firrtl.uint<8>
  }
  firrtl.module @Bug4882Rename() {
  // CHECK-LABEL: firrtl.module @Bug4882Rename() {
    firrtl.instance no sym @no  @Bar1()
    // CHECK-NEXT: firrtl.instance no_bar3 sym @w_0 @Bar3()
    firrtl.instance bar2 sym @w  @Bar2()
    // CHECK-NEXT: %bar2_x = firrtl.wire sym @x {annotations = [{class = "test0"}]}
  }
}

// -----

// Issue #4920, the recursive inlining should consider the correct retop for NLAs.

firrtl.circuit "DidNotContainSymbol" {
  hw.hierpath private @path [@Bar1::@w, @Bar3]
  firrtl.module private @Bar2() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance no sym @no @Bar1()
  }
  firrtl.module private @Bar1() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance bar3 sym @w @Bar3()
  }
  firrtl.module private @Bar3() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %w = firrtl.wire sym @w {annotations = [{circt.nonlocal = @path , class = "test0"}]} : !firrtl.uint<8>
  }
  firrtl.module @DidNotContainSymbol() {
    firrtl.instance bar2 sym @w @Bar2()
  }
  // CHECK-LABEL: firrtl.module @DidNotContainSymbol() {
  // CHECK-NEXT:     %bar2_no_bar3_w = firrtl.wire sym @w_0 {annotations = [{class = "test0"}]} : !firrtl.uint<8>
  // CHECK-NEXT:  }
}

// -----

// Issue #4915, the NLAs should be updated with renamed extern module instance.

firrtl.circuit "SimTop" {
  hw.hierpath private @nla_61 [@Rob::@difftest_3, @DifftestLoadEvent]
  // CHECK: hw.hierpath private @nla_61 [@SimTop::@difftest_3_0, @DifftestLoadEvent]
  hw.hierpath private @nla_60 [@Rob::@difftest_2, @DifftestLoadEvent]
  // CHECK: hw.hierpath private @nla_60 [@SimTop::@difftest_2, @DifftestLoadEvent]
  firrtl.extmodule private @DifftestIntWriteback()
  firrtl.extmodule private @DifftestLoadEvent() attributes {annotations = [{circt.nonlocal = @nla_60, class = "B"}, {circt.nonlocal = @nla_61, class = "B"}]}
	// CHECK: firrtl.extmodule private @DifftestLoadEvent() attributes {annotations = [{circt.nonlocal = @nla_60, class = "B"}, {circt.nonlocal = @nla_61, class = "B"}]}
  firrtl.module private @Rob() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance difftest_2 sym @difftest_2 @DifftestLoadEvent()
    firrtl.instance difftest_3 sym @difftest_3 @DifftestLoadEvent()
  }
  firrtl.module private @CtrlBlock() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance rob @Rob()
  }
  firrtl.module @SimTop() {
    firrtl.instance difftest_3 sym @difftest_3 @DifftestIntWriteback()
    firrtl.instance ctrlBlock @CtrlBlock()
    // CHECK:  firrtl.instance difftest_3 sym @difftest_3 @DifftestIntWriteback()
    // CHECK:  firrtl.instance ctrlBlock_rob_difftest_2 sym @difftest_2 @DifftestLoadEvent()
    // CHECK:  firrtl.instance ctrlBlock_rob_difftest_3 sym @difftest_3_0 @DifftestLoadEvent()
  }
}

// -----

// Check updating of module-local inner symbol users, handle per-field inner symbols.


// CHECK-LABEL: circuit "RWProbePort"
firrtl.circuit "RWProbePort" {
  // CHECK-NOT: @Child
  firrtl.module private @Child(in %in: !firrtl.vector<uint<1>, 2>
                                 sym [<@sym,2,public>],
                               out %p: !firrtl.rwprobe<uint<1>>) attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %0 = firrtl.ref.rwprobe <@Child::@sym> : !firrtl.rwprobe<uint<1>>
    firrtl.ref.define %p, %0 : !firrtl.rwprobe<uint<1>>
  }
  // CHECK: module @RWProbePort
  firrtl.module @RWProbePort(in %in_0: !firrtl.uint<1>,
                             in %in_1: !firrtl.uint<1>,
                             out %p_0: !firrtl.rwprobe<uint<1>>,
                             out %p_1: !firrtl.rwprobe<uint<1>>) attributes {convention = #firrtl<convention scalarized>} {

    // CHECK-NEXT: %[[C1_IN:.+]] = firrtl.wire sym [<@[[C1_IN_SYM:.+]],2,public>]
    // CHECK-NEXT: %[[C1_P_WIRE:.+]] = firrtl.wire : !firrtl.rwprobe<uint<1>>
    // CHECK-NEXT: %[[C1_P:.+]] = firrtl.ref.rwprobe <@RWProbePort::@[[C1_IN_SYM]]>
    // CHECK-NEXT: firrtl.ref.define %c1_p, %0 : !firrtl.rwprobe<uint<1>>
    // CHECK-NEXT: %[[C2_IN:.+]] = firrtl.wire sym [<@[[C2_IN_SYM:.+]],2,public>]
    // CHECK-NEXT: %[[C2_P_WIRE:.+]] = firrtl.wire : !firrtl.rwprobe<uint<1>>
    // CHECK-NEXT: %[[C2_P:.+]] = firrtl.ref.rwprobe <@RWProbePort::@[[C2_IN_SYM]]>
    // CHECK-NEXT: firrtl.ref.define %c2_p, %1 : !firrtl.rwprobe<uint<1>>
    // CHECK-NOT: firrtl.instance
    %c1_in, %c1_p = firrtl.instance c1 @Child(in in: !firrtl.vector<uint<1>, 2>, out p: !firrtl.rwprobe<uint<1>>)
    %c2_in, %c2_p = firrtl.instance c2 @Child(in in: !firrtl.vector<uint<1>, 2>, out p: !firrtl.rwprobe<uint<1>>)
    %0 = firrtl.subindex %c1_in[0] : !firrtl.vector<uint<1>, 2>
    firrtl.matchingconnect %0, %in_0 : !firrtl.uint<1>
    %1 = firrtl.subindex %c1_in[1] : !firrtl.vector<uint<1>, 2>
    firrtl.matchingconnect %1, %in_1 : !firrtl.uint<1>
    %2 = firrtl.subindex %c2_in[0] : !firrtl.vector<uint<1>, 2>
    firrtl.matchingconnect %2, %in_0 : !firrtl.uint<1>
    %3 = firrtl.subindex %c2_in[1] : !firrtl.vector<uint<1>, 2>
    firrtl.matchingconnect %3, %in_1 : !firrtl.uint<1>
    // CHECK: firrtl.ref.define %p_0, %[[C1_P_WIRE]]
    firrtl.ref.define %p_0, %c1_p : !firrtl.rwprobe<uint<1>>
    // CHECK: firrtl.ref.define %p_1, %[[C2_P_WIRE]]
    firrtl.ref.define %p_1, %c2_p : !firrtl.rwprobe<uint<1>>
  }
}

// -----

// https://github.com/llvm/circt/issues/5598

// CHECK-LABEL: "CollidingSymbolsFields"
firrtl.circuit "CollidingSymbolsFields" {
  // CHECK-NEXT: hw.hierpath private @nla1 [@CollidingSymbolsFields::@[[FoobarSym:[_a-zA-Z0-9]+]], @Bar]
  hw.hierpath private @nla1 [@CollidingSymbolsFields::@foo, @Foo::@bar, @Bar]
  firrtl.module @Bar() attributes {annotations = [{circt.nonlocal = @nla1, class = "nla1"}]} {}
  firrtl.module @Foo(in %x : !firrtl.bundle<a: uint<1>, b: uint<1>> sym [<@b_0,1,public>,<@foo,2,public>]) attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %b = firrtl.wire sym [<@b,1,public>,<@bar_0,2,public>] : !firrtl.bundle<a: uint<1>, b: uint<1>>
    firrtl.instance bar sym @bar @Bar()
    %1 = firrtl.ref.rwprobe <@Foo::@b_0> : !firrtl.rwprobe<uint<1>>
    %2 = firrtl.ref.rwprobe <@Foo::@bar_0> : !firrtl.rwprobe<uint<1>>
    %3 = firrtl.ref.rwprobe <@Foo::@foo> : !firrtl.rwprobe<uint<1>>
  }
  // CHECK: module @CollidingSymbolsFields(
  // CHECK-SAME: sym [<@b_0
  firrtl.module @CollidingSymbolsFields(in %x : !firrtl.bundle<a: uint<1>> sym [<@b_0,1,public>]) {
    // CHECK-NEXT: firrtl.wire sym [<@[[FOO_X_A_SYM:.+]],1,public>, <@[[FOO_X_B_SYM:.+]],2,public>]
    // CHECK-NEXT: firrtl.wire sym [<@[[FOO_B_A_SYM:.+]],1,public>, <@[[FOO_B_B_SYM:.+]],2,public>]
    // CHECK-NEXT: firrtl.instance foo_bar sym @[[FoobarSym]] @Bar
    // CHECK-NEXT: firrtl.ref.rwprobe <@CollidingSymbolsFields::@[[FOO_X_A_SYM]]>
    // CHECK-NEXT: firrtl.ref.rwprobe <@CollidingSymbolsFields::@[[FOO_B_B_SYM]]>
    // CHECK-NEXT: firrtl.ref.rwprobe <@CollidingSymbolsFields::@[[FOO_X_B_SYM]]>
    // CHECK-NEXT: firrtl.wire sym @b
    // CHECK-NEXT: firrtl.wire sym [<@bar,1,public>, <@bar_0,2,public>]
    // CHECK-NEXT: }
    firrtl.instance foo sym @foo @Foo(in x : !firrtl.bundle<a: uint<1>, b: uint<1>>)
    %collision_b = firrtl.wire sym @b : !firrtl.uint<1>
    %collision_bar = firrtl.wire sym [<@bar,1,public>,<@bar_0,2,public>] : !firrtl.bundle<a: uint<1>, b: uint<1>>
  }
}

// -----
// Test that unused classes are NOT deleted.

firrtl.circuit "Top" {
  firrtl.module @Top () {}
  // CHECK: firrtl.class private @MyClass()
  firrtl.class private @MyClass() {}
}

// -----
// Test this doesn't crash.

firrtl.circuit "InlinerRefs" {
  firrtl.module private @ChildOut(in %in: !firrtl.bundle<a: uint<1>, b: uint<2>>, out %out: !firrtl.probe<bundle<a: uint<1>, b: uint<2>>>) attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %0 = firrtl.subfield %in[a] : !firrtl.bundle<a: uint<1>, b: uint<2>>
    firrtl.when %0 : !firrtl.uint<1> {
      %1 = firrtl.ref.send %in : !firrtl.bundle<a: uint<1>, b: uint<2>>
      firrtl.ref.define %out, %1 : !firrtl.probe<bundle<a: uint<1>, b: uint<2>>>
    }
  }
  firrtl.module @InlinerRefs(in %in: !firrtl.bundle<a: uint<1>, b: uint<2>>, out %out: !firrtl.uint<1>) {
    %0 = firrtl.subfield %in[a] : !firrtl.bundle<a: uint<1>, b: uint<2>>
    %co_in, %co_out = firrtl.instance co interesting_name @ChildOut(in in: !firrtl.bundle<a: uint<1>, b: uint<2>>, out out: !firrtl.probe<bundle<a: uint<1>, b: uint<2>>>)
    %1 = firrtl.ref.sub %co_out[0] : !firrtl.probe<bundle<a: uint<1>, b: uint<2>>>
    firrtl.matchingconnect %co_in, %in : !firrtl.bundle<a: uint<1>, b: uint<2>>
    firrtl.when %0 : !firrtl.uint<1> {
      %2 = firrtl.ref.resolve %1 : !firrtl.probe<uint<1>>
      firrtl.matchingconnect %out, %2 : !firrtl.uint<1>
    }
  }
}

// -----
// Issue #5941.

// Test for U-Turn in property ports.
// The inlined module propassign's and uses the property.
// CHECK-LABEL: firrtl.circuit "PropertyUTurn"
firrtl.circuit "PropertyUTurn" {
  // CHECK: module @PropertyUTurn
  firrtl.module @PropertyUTurn() {
    %c_in, %c_out = firrtl.instance child @Child(in in: !firrtl.string, out out: !firrtl.string)
    firrtl.propassign %c_in, %c_out : !firrtl.string
    // CHECK-NEXT: %child_in = firrtl.wire : !firrtl.string
    // CHECK-NEXT: %child_out = firrtl.wire : !firrtl.string
    // CHECK-NEXT: %child_s_out = firrtl.instance child_s sym @Out @OutStr(out out: !firrtl.string)
    // CHECK-NEXT: firrtl.propassign %child_out, %child_s_out : !firrtl.string
    // CHECK-NEXT: %child_c_in = firrtl.instance child_c sym @C @Consume(in in: !firrtl.string)
    // CHECK-NEXT: firrtl.propassign %child_c_in, %child_in : !firrtl.string
    // CHECK-NEXT: firrtl.propassign %child_in, %child_out : !firrtl.string
  }
  firrtl.module @Child(in %in: !firrtl.string, out %out: !firrtl.string) attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]}{
    %s_out = firrtl.instance s sym @Out @OutStr(out out: !firrtl.string)
    firrtl.propassign %out, %s_out : !firrtl.string

    %c_in = firrtl.instance c sym @C @Consume(in in : !firrtl.string)
    firrtl.propassign %c_in, %in : !firrtl.string
  }
  firrtl.module @OutStr(out %out : !firrtl.string) {
    %str = firrtl.string "hello"
    firrtl.propassign %out, %str : !firrtl.string
  }
  firrtl.extmodule @Consume(in in : !firrtl.string)
}

// -----

// Test that inlining and flattening compose with nla well.
firrtl.circuit "compose_nla" {
  hw.hierpath private @nla1 [@test1::@sym, @test2::@sym, @test3]
// Retention keeps the fully-collapsed path (annotation localizes onto the op).
// CHECK: hw.hierpath private @nla1 [@compose_nla]
firrtl.module @compose_nla() {
// CHECK-LABEL: firrtl.module @compose_nla() {
  firrtl.instance test1 @test1()
  firrtl.instance test2 @test2()
  firrtl.instance test3 @test3()
}
firrtl.module private @test1() attributes {annotations =
        [{class = "firrtl.transforms.FlattenAnnotation"},
         {class = "firrtl.passes.InlineAnnotation"}]} {
  %test_wire = firrtl.wire : !firrtl.uint<2>
  firrtl.instance test2 sym @sym @test2()
  firrtl.instance test3 @test3()
}
firrtl.module private @test2() attributes {annotations =
        [{class = "firrtl.passes.InlineAnnotation"}]} {
  %test_wire = firrtl.wire : !firrtl.uint<2>
  firrtl.instance test3 sym @sym @test3()
}
firrtl.module private @test3() {
  %test_wire = firrtl.wire : !firrtl.uint<2>
}
}

// -----

// Directly check simplest example of inlining a module containing a layerblock.

// CHECK-LABEL: firrtl.circuit "InlineLayerBlockSimple"
firrtl.circuit "InlineLayerBlockSimple" {
  firrtl.layer @I inline { }
  // CHECK-NOT: @Child
  firrtl.module private @Child() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.layerblock @I {
      %o = firrtl.wire interesting_name : !firrtl.uint<8>
    }
  }
  // CHECK: @InlineLayerBlockSimple
  firrtl.module @InlineLayerBlockSimple() {
    // Check inlined structure.
    // CHECK-NEXT: layerblock @I
    // CHECK-NEXT:   firrtl.wire
    // CHECK-NEXT: }
    firrtl.instance c @Child()
  }
}

// -----

// Check recurse into instances not at top-level.

// CHECK-LABEL: firrtl.circuit "WalkIntoInstancesUnderLayerBlock"
firrtl.circuit "WalkIntoInstancesUnderLayerBlock" {
  firrtl.layer @I inline { }
  // CHECK-NOT: @GChild
  firrtl.module private @GChild() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %o = firrtl.wire interesting_name : !firrtl.uint<8>
  }
  // CHECK: @Child
  firrtl.module private @Child() {
    // CHECK-NEXT: %gc_o = firrtl.wire
    firrtl.instance gc @GChild()
  }
  // CHECK: @WalkIntoInstancesUnderLayerBlock
  firrtl.module @WalkIntoInstancesUnderLayerBlock() {
    // CHECK-NEXT: layerblock @I
    // CHECK-NEXT:   firrtl.instance c @Child
    firrtl.layerblock @I {
      firrtl.instance c @Child()
    }
  }
}

// -----

// Test inlining into nested layer, and cloning operations with blocks + blockargs (match).

// CHECK-LABEL: firrtl.circuit "MatchInline"
firrtl.circuit "MatchInline" attributes {enable_layers = [@I]} {
  firrtl.layer @I inline {
    firrtl.layer @J inline { }
  }
  // CHECK-NOT: @MatchAgain
  firrtl.module private @MatchAgain(in %i: !firrtl.enum<Some: uint<8>, None: uint<0>>, out %o: !firrtl.uint<8>) attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.match %i : !firrtl.enum<Some: uint<8>, None: uint<0>> {
      case Some(%arg0) {
        %not = firrtl.not %arg0 : (!firrtl.uint<8>) -> !firrtl.uint<8>
        firrtl.matchingconnect %o, %not : !firrtl.uint<8>
      }
      case None(%arg0) {
        %invalid_ui8 = firrtl.invalidvalue : !firrtl.uint<8>
        firrtl.matchingconnect %o, %invalid_ui8 : !firrtl.uint<8>
      }
    }
  }
  // CHECK: @MatchInline
  firrtl.module @MatchInline(in %i: !firrtl.enum<Some: uint<8>, None: uint<0>>) {
    // CHECK-NEXT: layerblock @I
    firrtl.layerblock @I {
      // CHECK-NEXT: layerblock @I::@J
      firrtl.layerblock @I::@J {
        // CHECK-NOT: @MatchAgain
        // CHECK: firrtl.match
        // CHECK-NEXT: Some(%arg0)
        // CHECK-NEXT: firrtl.not %arg0
        // CHECK: None(%arg0)
        %c_i, %c_o = firrtl.instance c @MatchAgain(in i: !firrtl.enum<Some: uint<8>, None: uint<0>>, out o: !firrtl.uint<8>)
        firrtl.matchingconnect %c_i, %i : !firrtl.enum<Some: uint<8>, None: uint<0>>
      }
    }
  }
}

// -----

// Test inlining module containing various operations with blocks.
// Include operations before/after regions as well as populating block bodies
// and using results to check inlining actually does work here and the
// management of the insertion points throughout.

firrtl.circuit "InlineBlocks" {
  firrtl.layer @I inline {
    firrtl.layer @J inline { }
  }
  firrtl.module private @HasBlocks(in %i: !firrtl.enum<Some: uint<8>, None: uint<0>>,
                                   in %cond: !firrtl.uint<1>,
                                   out %p: !firrtl.probe<uint<8>, @I::@J>) attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.layerblock @I {
      firrtl.when %cond : !firrtl.uint<1> {
        firrtl.layerblock @I::@J {
          %o = firrtl.wire interesting_name : !firrtl.uint<8>
          %0 = firrtl.ref.send %o : !firrtl.uint<8>
          %1 = firrtl.ref.cast %0 : (!firrtl.probe<uint<8>>) -> !firrtl.probe<uint<8>, @I::@J>
          firrtl.ref.define %p, %1 : !firrtl.probe<uint<8>, @I::@J>
          firrtl.match %i : !firrtl.enum<Some: uint<8>, None: uint<0>> {
            case Some(%arg0) {
              firrtl.matchingconnect %o, %arg0 : !firrtl.uint<8>
            }
            case None(%arg0) {
              %invalid_ui8 = firrtl.invalidvalue : !firrtl.uint<8>
              firrtl.matchingconnect %o, %invalid_ui8 : !firrtl.uint<8>
            }
          }
          %unused = firrtl.node %o : !firrtl.uint<8>
        }
        %unused = firrtl.node %cond : !firrtl.uint<1>
      }
    }
  }
  // CHECK: @InlineBlocks
  firrtl.module @InlineBlocks(in %i: !firrtl.enum<Some: uint<8>, None: uint<0>>, in %cond: !firrtl.uint<1>, out %o: !firrtl.probe<uint<8>, @I::@J>) attributes {convention = #firrtl<convention scalarized>} {
    // Check inlined structure.
    // CHECK:      layerblock @I
    // CHECK-NEXT:   firrtl.when
    // CHECK-NEXT:     firrtl.layerblock @I::@J
    // CHECK-NEXT:       firrtl.wire
    // CHECK:            firrtl.match
    // CHECK:              Some(
    // CHECK:              None(
    // CHECK:              }
    // CHECK-NEXT:       }
    // CHECK-NEXT:       firrtl.node {{.*}}
    // CHECK-NEXT:     }
    // CHECK-NEXT:     firrtl.node {{.*}}
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    %c_i, %c_cond, %c_p = firrtl.instance c interesting_name @HasBlocks(in i: !firrtl.enum<Some: uint<8>, None: uint<0>>, in cond: !firrtl.uint<1>, out p: !firrtl.probe<uint<8>, @I::@J>)
    firrtl.matchingconnect %c_i, %i : !firrtl.enum<Some: uint<8>, None: uint<0>>
    firrtl.matchingconnect %c_cond, %cond : !firrtl.uint<1>
    firrtl.ref.define %o, %c_p : !firrtl.probe<uint<8>, @I::@J>
  }
}

// -----

// The inliner must not delete modules which are still referenced, even in unknown ops.
firrtl.circuit "FormalMarkerIsUse" {
  firrtl.extmodule @FormalMarkerIsUse()
  firrtl.formal @Test, @Foo {}
  "some_unknown_dialect.op"() { magic = @Bar } : () -> ()
  firrtl.module private @Foo() {}
  firrtl.module private @Bar() {}
  // CHECK: firrtl.module private @Foo
  // CHECK: firrtl.module private @Bar
}


// -----

firrtl.circuit "RemoveNonLocalFromLocal" {
  // Retention keeps @dutNLA; the tracker annotation still localizes off it.
  // CHECK: hw.hierpath private @dutNLA [@RemoveNonLocalFromLocal::@sym]
  hw.hierpath private @dutNLA [@RemoveNonLocalFromLocal::@sym]
  firrtl.module @Bar() {}
  // CHECK-LABEL: firrtl.module @RemoveNonLocalFromLocal
  firrtl.module @RemoveNonLocalFromLocal() {
    // CHECK: firrtl.instance bar sym @sym {annotations = [{class = "circt.tracker", id = distinct[0]<>}]} @Bar()
    firrtl.instance bar sym @sym {annotations = [{circt.nonlocal = @dutNLA, class = "circt.tracker", id = distinct[0]<>}]} @Bar()
  }
}

// -----
firrtl.circuit "Object" {
  // CHECK: firrtl.class @MyClass
  firrtl.class @MyClass() {}

  firrtl.module private @Child() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    // Both object and instance_choice in the same module
    %obj = firrtl.object @MyClass()
  }

  // CHECK-LABEL: firrtl.module @Object
  firrtl.module @Object() {
    // CHECK: firrtl.object @MyClass
    firrtl.instance child @Child()
  }
}
// -----

// Test that both firrtl.object and instance_choice work together during inlining.
// This ensures both FInstanceLike operations are handled correctly in the same module.
firrtl.circuit "InstanceChoice" {
  firrtl.option @Platform {
    firrtl.option_case @FPGA
  }

  // CHECK: firrtl.module private @ImplA
  firrtl.module private @ImplA() {}

  firrtl.module private @Child(in %x: !firrtl.uint<8>, out %y: !firrtl.uint<8>)
    attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    // Both object and instance_choice in the same module
    firrtl.instance_choice inst @ImplA alternatives @Platform {
      @FPGA -> @ImplA
    } ()
  }

  // CHECK-LABEL: firrtl.module @InstanceChoice
  firrtl.module @InstanceChoice() {
    // After inlining, both object and instance_choice should be present
    // CHECK: firrtl.instance_choice child_inst @ImplA
    firrtl.instance child @Child(in x: !firrtl.uint<8>, out y: !firrtl.uint<8>)
  }
}

// -----

// Test that children of modules referenced by instance_choice can still be inlined.
// This ensures that marking modules as live doesn't prevent their children from being inlined.
firrtl.circuit "InstanceChoiceChildrenInlineable" {
  firrtl.option @Platform {
    firrtl.option_case @FPGA
    firrtl.option_case @ASIC
  }

  // This should be inlined into FPGAImpl and then deleted
  // CHECK-NOT: firrtl.module private @InlineableChild
  firrtl.module private @InlineableChild(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>)
    attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %c1_ui8 = firrtl.constant 1 : !firrtl.uint<8>
    firrtl.connect %out, %c1_ui8 : !firrtl.uint<8>, !firrtl.uint<8>
  }

  // This module is referenced by instance_choice and has an inlineable child
  // CHECK: firrtl.module private @FPGAImpl
  firrtl.module private @FPGAImpl(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    // CHECK-NOT: firrtl.instance child @InlineableChild
    // CHECK: %child_in = firrtl.wire
    // CHECK: %child_out = firrtl.wire
    // CHECK: firrtl.constant 1
    %child_in, %child_out = firrtl.instance child @InlineableChild(in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.connect %child_in, %in : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %out, %child_out : !firrtl.uint<8>, !firrtl.uint<8>
  }

  // CHECK: firrtl.module private @ASICImpl
  firrtl.module private @ASICImpl(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    firrtl.connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }

  // CHECK-LABEL: firrtl.module @InstanceChoiceChildrenInlineable
  firrtl.module @InstanceChoiceChildrenInlineable(in %a: !firrtl.uint<8>, out %b: !firrtl.uint<8>) {
    // CHECK: firrtl.instance_choice
    // CHECK-SAME: @FPGAImpl
    // CHECK-SAME: @ASICImpl
    %inst_in, %inst_out = firrtl.instance_choice inst @FPGAImpl alternatives @Platform {
      @FPGA -> @FPGAImpl,
      @ASIC -> @ASICImpl
    } (in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.connect %inst_in, %a : !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.connect %b, %inst_out : !firrtl.uint<8>, !firrtl.uint<8>
  }
}

// -----

// Test that instance_choice works correctly during flattening.
// Flattening should stop at instance_choice boundaries - the instance_choice
// itself gets inlined, but modules referenced by it are not flattened.
firrtl.circuit "InstanceChoiceWithFlattening" {
  firrtl.option @Platform {
    firrtl.option_case @FPGA
  }

  // CHECK-LABEL: firrtl.module private @ChildInsideChoice
  firrtl.module private @ChildInsideChoice() {}

  // This module is referenced by instance_choice and should be kept
  // CHECK-LABEL: firrtl.module private @ImplA
  firrtl.module private @ImplA() {
    // CHECK: firrtl.instance child @ChildInsideChoice
    firrtl.instance child @ChildInsideChoice()
  }

  // This module contains an instance_choice
  firrtl.module private @Level2() {
    firrtl.instance_choice inst @ImplA alternatives @Platform {
      @FPGA -> @ImplA
    } ()
  }

  firrtl.module private @Level1() {
    firrtl.instance level2 @Level2()
  }

  // CHECK-LABEL: firrtl.module @InstanceChoiceWithFlattening
  firrtl.module @InstanceChoiceWithFlattening()
    attributes {annotations = [{class = "firrtl.transforms.FlattenAnnotation"}]} {
    // After flattening, instance_choice should be inlined but still reference @ImplA
    // CHECK: firrtl.instance_choice level1_level2_inst @ImplA
    firrtl.instance level1 @Level1()
  }
}

// -----

// Test that NLAs are correctly updated when flattening.
//
// CHECK-LABEL: firrtl.circuit "FlattenAtRoot"
firrtl.circuit "FlattenAtRoot" {
  // Retention keeps @nla (flattened to a one-hop path; annotation localizes).
  // CHECK: hw.hierpath private @nla [@Foo::@b]
  hw.hierpath private @nla [@Foo::@bar, @Bar::@b]
  // CHECK: firrtl.module @Bar
  firrtl.module @Bar() {
    // CHECK: %b = firrtl.wire sym @b
    // CHECK-NOT: annotations
    %b = firrtl.wire sym @b {annotations = [{circt.nonlocal = @nla, class = "nla"}]} : !firrtl.uint<1>
  }
  // CHECK: firrtl.module @Foo
  firrtl.module @Foo() attributes {annotations = [{class = "firrtl.transforms.FlattenAnnotation"}]} {
    // CHECK: %bar_b = firrtl.wire sym @b {annotations = [{class = "nla"}]}
    firrtl.instance bar sym @bar @Bar()
  }
  // CHECK: firrtl.module @FlattenAtRoot
  firrtl.module @FlattenAtRoot() {
    firrtl.instance foo sym @foo @Foo()
  }
}

// -----

// Minimal testcase for: https://github.com/llvm/circt/issues/3374
// Modified so as to not expose a different unrelated bug.

// Verify each top gets its own isolated annotation entry.
// (Don't generate invalid DictionaryAttr's with many circt.local/class entries!)
// CHECK-LABEL: "Issue3374Derived"
firrtl.circuit "Issue3374Derived" {
  // CHECK-NEXT: hw.hierpath private @nla1 [@Issue3374Derived::@baz, @Baz]
  // CHECK-NEXT: hw.hierpath private @nla1_0 [@Quux::@baz, @Baz]
  // CHECK-NEXT: hw.hierpath private @nla1_1 [@Qux::@baz, @Baz]
  // CHECK-NEXT: hw.hierpath private @nla1_2 [@Foo::@baz, @Baz]
  // CHECK-NEXT: @Baz() {
  hw.hierpath private @nla1 [@Bar::@baz, @Baz]

  firrtl.module @Baz() {
  // CHECK-NEXT:   wire sym @a
  // CHECK-SAME:     {annotations = [{circt.nonlocal = @nla1, class = "hello"},
  // CHECK-SAME:                     {circt.nonlocal = @nla1_0, class = "hello"},
  // CHECK-SAME:                     {circt.nonlocal = @nla1_1, class = "hello"},
  // CHECK-SAME:                     {circt.nonlocal = @nla1_2, class = "hello"}]}
    %a = firrtl.wire sym @a {annotations = [{circt.nonlocal = @nla1, class = "hello"}]} : !firrtl.uint<1>
  }
  firrtl.module private @Bar() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance baz sym @baz @Baz()
  }
  firrtl.module private @Foo() {
    firrtl.instance bar @Bar()
  }
  firrtl.module private @Qux() {
    firrtl.instance bar @Bar()
  }
  firrtl.module private @Quux() {
    firrtl.instance bar @Bar()
  }
  firrtl.module @Issue3374Derived() {
    firrtl.instance b @Bar()
    firrtl.instance c @Foo()
    firrtl.instance d @Qux()
    firrtl.instance e @Quux()
  }
}

// -----
// https://github.com/llvm/circt/issues/10674
//
// Idempotent renames of inner symbols.
// CHECK-LABEL: "Issue10674"
firrtl.circuit "Issue10674" {
  hw.hierpath private @nla [@Parent::@inst, @Child::@w]
  firrtl.module private @Child() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    // @Child persists, nothing annotated here (@nla is rooted at @Parent).
    %w = firrtl.wire sym @w {annotations = [
      {circt.nonlocal = @nla, class = "anno1"},
      {circt.nonlocal = @nla, class = "anno2"}
    ]} : !firrtl.uint<1>
  }
  // CHECK: firrtl.module @Parent
  firrtl.module @Parent() {
    // CHECK: %existing = firrtl.wire sym @w
    %existing = firrtl.wire sym @w : !firrtl.uint<1>
    // both annotations localized onto the inlined copy:
    // CHECK: firrtl.wire sym @w_0 {annotations = [{class = "anno1"}, {class = "anno2"}]}
    firrtl.instance inst sym @inst @Child()
  }
  firrtl.module @Issue10674() {
    firrtl.instance p @Parent()
  }
}

// -----
// https://github.com/llvm/circt/issues/10682
//
// Single-element hierpath.
// CHECK-LABEL: "Issue10682"
firrtl.circuit "Issue10682" {
  hw.hierpath private @nla [@M::@w]
  // CHECK: firrtl.module @Issue10682
  firrtl.module @Issue10682() {
    firrtl.instance m @M()
  }
  // After inlining @M, the wire is moved into @Issue10682 and localized:
  // CHECK-NEXT: firrtl.wire sym @w {annotations = [{class = "test"}]}
  // CHECK-NOT: circt.nonlocal
  firrtl.module private @M() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %w = firrtl.wire sym @w {annotations = [{circt.nonlocal = @nla, class = "test"}]} : !firrtl.uint<1>
  }
}

// -----

// An inline-marked private module kept live only by a non-instance symbol
// user (a verbatim substitution here): its plain instances are inlined, the
// module is retained for the symbol user, and the annotation is consumed.
// Used to trip an assert that only expected public retained modules.
// CHECK-LABEL: firrtl.circuit "RetainedBySymbolUse"
firrtl.circuit "RetainedBySymbolUse" {
  sv.verbatim "`define REF {{0}}" {symbols = [@M]}
  // CHECK: firrtl.module private @M() {
  firrtl.module private @M() attributes {annotations = [
      {class = "firrtl.passes.InlineAnnotation"}]} {
    %w = firrtl.wire : !firrtl.uint<1>
  }
  // CHECK: firrtl.module @RetainedBySymbolUse()
  // CHECK-NEXT: %m_w = firrtl.wire
  firrtl.module @RetainedBySymbolUse() {
    firrtl.instance m @M()
  }
}

// -----
//===----------------------------------------------------------------------===//
// Hierarchical-path (NLA) behavior suite
//===----------------------------------------------------------------------===//

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
// The companion `FlattenFromAbove` in the issue-shapes section below exercises
// the same shape; both collapse to a single hierpath.
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

// -----
//===----------------------------------------------------------------------===//
// Issue-paired regressions and reduced crash shapes
//===----------------------------------------------------------------------===//

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
