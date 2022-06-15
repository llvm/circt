// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-inliner)' -allow-unregistered-dialect %s | FileCheck %s

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

// This is testing that connects are properly replaced when inlining. This is
// also testing that the deep clone and remapping values is working correctly.
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
// CHECK-NEXT:   firrtl.connect %b_a_out0, %0 : !firrtl.uint<4>, !firrtl.uint<4>
// CHECK-NEXT:   %1 = firrtl.and %b_a_in0, %b_a_in1 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
// CHECK-NEXT:   firrtl.connect %b_a_out1, %1 : !firrtl.uint<4>, !firrtl.uint<4>
// CHECK-NEXT:   firrtl.connect %b_a_in0, %b_in0 : !firrtl.uint<4>, !firrtl.uint<4>
// CHECK-NEXT:   firrtl.connect %b_a_in1, %b_in1 : !firrtl.uint<4>, !firrtl.uint<4>
// CHECK-NEXT:   firrtl.connect %b_out0, %b_a_out0 : !firrtl.uint<4>, !firrtl.uint<4>
// CHECK-NEXT:   firrtl.connect %b_out1, %b_a_out1 : !firrtl.uint<4>, !firrtl.uint<4>
// CHECK-NEXT:   firrtl.connect %b_in0, %in0 : !firrtl.uint<4>, !firrtl.uint<4>
// CHECK-NEXT:   firrtl.connect %b_in1, %in1 : !firrtl.uint<4>, !firrtl.uint<4>
// CHECK-NEXT:   firrtl.connect %out0, %b_out0 : !firrtl.uint<4>, !firrtl.uint<4>
// CHECK-NEXT:   firrtl.connect %out1, %b_out1 : !firrtl.uint<4>, !firrtl.uint<4>
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
// CHECK: firrtl.connect %i_out0, %i_in0 : !firrtl.bundle<a: uint<4>, b flip: uint<4>>, !firrtl.bundle<a: uint<4>, b flip: uint<4>>
// CHECK: firrtl.connect %i_in0, %in0 : !firrtl.bundle<a: uint<4>, b flip: uint<4>>, !firrtl.bundle<a: uint<4>, b flip: uint<4>>
// CHECK: firrtl.connect %out0, %i_out0 : !firrtl.bundle<a: uint<4>, b flip: uint<4>>, !firrtl.bundle<a: uint<4>, b flip: uint<4>>
}
}

// Test that all operations with names are renamed.
firrtl.circuit "renaming" {
firrtl.module @renaming() {
  %0, %1, %2 = firrtl.instance myinst @declarations(in clock : !firrtl.clock, in u8 : !firrtl.uint<8>, in reset : !firrtl.asyncreset)
}
firrtl.module @declarations(in %clock : !firrtl.clock, in %u8 : !firrtl.uint<8>, in %reset : !firrtl.asyncreset) attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
  %c0_ui8 = firrtl.constant 0 : !firrtl.uint<8>
  // CHECK: %cmem = chirrtl.combmem : !chirrtl.cmemory<uint<8>, 8>
  %cmem = chirrtl.combmem : !chirrtl.cmemory<uint<8>, 8>
  // CHECK: %mem_read = firrtl.mem Undefined {depth = 1 : i64, name = "mem", portNames = ["read"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: sint<42>>
  %mem_read = firrtl.mem Undefined {depth = 1 : i64, name = "mem", portNames = ["read"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: sint<42>>
  // CHECK: %memoryport_data, %memoryport_port = chirrtl.memoryport Read %cmem {name = "memoryport"} : (!chirrtl.cmemory<uint<8>, 8>) -> (!firrtl.uint<8>, !chirrtl.cmemoryport)
  %memoryport_data, %memoryport_port = chirrtl.memoryport Read %cmem {name = "memoryport"} : (!chirrtl.cmemory<uint<8>, 8>) -> (!firrtl.uint<8>, !chirrtl.cmemoryport)
  chirrtl.memoryport.access %memoryport_port[%u8], %clock : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
  // CHECK: %myinst_node = firrtl.node %myinst_u8  : !firrtl.uint<8>
  %node = firrtl.node %u8 {name = "node"} : !firrtl.uint<8>
  // CHECK: %myinst_reg = firrtl.reg %myinst_clock : !firrtl.uint<8>
  %reg = firrtl.reg %clock {name = "reg"} : !firrtl.uint<8>
  // CHECK: %myinst_regreset = firrtl.regreset %myinst_clock, %myinst_reset, %c0_ui8 : !firrtl.asyncreset, !firrtl.uint<8>, !firrtl.uint<8>
  %regreset = firrtl.regreset %clock, %reset, %c0_ui8 : !firrtl.asyncreset, !firrtl.uint<8>, !firrtl.uint<8>
  // CHECK: %smem = chirrtl.seqmem Undefined : !chirrtl.cmemory<uint<8>, 8>
  %smem = chirrtl.seqmem Undefined : !chirrtl.cmemory<uint<8>, 8>
  // CHECK: %myinst_wire = firrtl.wire  : !firrtl.uint<1>
  %wire = firrtl.wire : !firrtl.uint<1>
  firrtl.when %wire {
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
// begins at the main module.  There are four behaviors being tested:
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
  // CHECK-NEXT: firrtl.hierpath @nla1 [@NLAInlining::@bar, @Bar]
  // CHECK-NEXT: firrtl.hierpath @nla2 [@NLAInlining::@bar, @Bar::@a]
  // CHECK-NEXT: firrtl.hierpath @nla3 [@NLAInlining::@bar, @Bar::@port]
  // CHECK-NOT:  firrtl.hierpath @nla4
  // CHECK-NOT:  firrtl.hierpath @nla5
  firrtl.hierpath @nla1 [@NLAInlining::@foo, @Foo::@bar, @Bar]
  firrtl.hierpath @nla2 [@NLAInlining::@foo, @Foo::@bar, @Bar::@a]
  firrtl.hierpath @nla3 [@NLAInlining::@foo, @Foo::@bar, @Bar::@port]
  firrtl.hierpath @nla4 [@NLAInlining::@foo, @Foo]
  firrtl.hierpath @nla5 [@NLAInlining::@foo, @Foo::@b]
  firrtl.hierpath @nla6 [@NLAInlining::@foo, @Foo::@port]
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

// Test NLA handling during inlining for situations where the NLA does NOT start
// at the root.  This checks that the NLA, on either a component or a port, is
// properly copied for each new instantiation.
//
// CHECK-LABEL: firrtl.circuit "NLAInliningNotMainRoot"
firrtl.circuit "NLAInliningNotMainRoot" {
  // CHECK-NEXT: firrtl.hierpath @nla1 [@NLAInliningNotMainRoot::@baz, @Baz::@a]
  // CHECK-NEXT: firrtl.hierpath @nla1_0 [@Foo::@baz, @Baz::@a]
  // CHECK-NEXT: firrtl.hierpath @nla2 [@NLAInliningNotMainRoot::@baz, @Baz::@port]
  // CHECK-NEXT: firrtl.hierpath @nla2_0 [@Foo::@baz, @Baz::@port]
  firrtl.hierpath @nla1 [@Bar::@baz, @Baz::@a]
  firrtl.hierpath @nla2 [@Bar::@baz, @Baz::@port]
  // CHECK: firrtl.module private @Baz
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
// is the flattened module or an ancestor of the flattened module.  This is
// testing the following conditions:
//
//   1) @nla1: Targeting a reference should be updated.
//   2) @nla1: Targeting a port should be updated.
//   3) @nla3: Targeting a module should be dropped.
//   4) @nla4: Targeting a reference should be promoted to local.
//
// CHECK-LABEL: firrtl.circuit "NLAFlattening"
firrtl.circuit "NLAFlattening" {
  // CHECK-NEXT: firrtl.hierpath @nla1 [@NLAFlattening::@foo, @Foo::@a]
  // CHECK-NEXT: firrtl.hierpath @nla2 [@NLAFlattening::@foo, @Foo::@port]
  // CHECK-NOT:  firrtl.hierpath @nla3
  // CHECK-NOT:  firrtl.hierpath @nla4
  firrtl.hierpath @nla1 [@NLAFlattening::@foo, @Foo::@bar, @Bar::@baz, @Baz::@a]
  firrtl.hierpath @nla2 [@NLAFlattening::@foo, @Foo::@bar, @Bar::@baz, @Baz::@port]
  firrtl.hierpath @nla3 [@NLAFlattening::@foo, @Foo::@bar, @Bar::@baz, @Baz]
  firrtl.hierpath @nla4 [@Foo::@bar, @Bar::@b]
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

// Test NLA handling during flattening for situations where the NLA root is a
// child of the flattened module.  This is testing the following situations:
//
//   1) @nla1: NLA component is made local and garbage collected.
//   2) @nla2: NLA port is made local and garbage collected.
//   3) @nla3: NLA component is made local, but not garbage collected.
//   4) @nla4: NLA port is made local, but not garbage collected.
//
// CHECK-LABEL: firrtl.circuit "NLAFlatteningChildRoot"
firrtl.circuit "NLAFlatteningChildRoot" {
  // CHECK-NOT:  firrtl.hierpath @nla1
  // CHECK-NOT:  firrtl.hierpath @nla2
  // CHECK-NEXT: firrtl.hierpath @nla3 [@Baz::@quz, @Quz::@b]
  // CHECK-NEXT: firrtl.hierpath @nla4 [@Baz::@quz, @Quz::@Quz_port]
  firrtl.hierpath @nla1 [@Bar::@qux, @Qux::@a]
  firrtl.hierpath @nla2 [@Bar::@qux, @Qux::@Qux_port]
  firrtl.hierpath @nla3 [@Baz::@quz, @Quz::@b]
  firrtl.hierpath @nla4 [@Baz::@quz, @Quz::@Quz_port]
  // CHECK: firrtl.module private @Quz
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
  // CHECK-NEXT: firrtl.hierpath @nla1 [@CollidingSymbols::@[[FoobarSym:[_a-zA-Z0-9]+]], @Bar]
  firrtl.hierpath @nla1 [@CollidingSymbols::@foo, @Foo::@bar, @Bar]
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
  // CHECK-NEXT: firrtl.hierpath @nla1 [@CollidingSymbolsPort::@foo, @Foo::@[[BarbSym:[_a-zA-Z0-9]+]]]
  firrtl.hierpath @nla1 [@CollidingSymbolsPort::@foo, @Foo::@bar, @Bar::@b]
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
  // CHECK-NOT:  #hw.innerNameRef<@CollidingSymbolsReTop::@baz>
  // CHECK-NOT:  #hw.innerNameRef<@Foo::@baz>
  // CHECK-NEXT: firrtl.hierpath @nla1 [@CollidingSymbolsReTop::@[[TopbazSym:[_a-zA-Z0-9]+]], @Baz::@a]
  // CHECK-NEXT: firrtl.hierpath @nla1_0 [@Foo::@[[FoobazSym:[_a-zA-Z0-9]+]], @Baz::@a]
  firrtl.hierpath @nla1 [@Bar::@baz, @Baz::@a]
  // CHECK: firrtl.module @Baz
  firrtl.module @Baz() {
    // CHECK-NEXT: firrtl.wire {{.+}} [{circt.nonlocal = @nla1, class = "hello"}, {circt.nonlocal = @nla1_0, class = "hello"}]
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
// that the NLA is properly updated.  Specifically in this test case, the second
// instance inlined should be renamed, and it should *not* update the NLA.
// CHECK-LABEL: firrtl.circuit "CollidingSymbolsNLAFixup"
firrtl.circuit "CollidingSymbolsNLAFixup" {
  // CHECK: firrtl.hierpath @nla0 [@Foo::@bar, @Bar::@io]
  firrtl.hierpath @nla0 [@Foo::@bar, @Bar::@baz0, @Baz::@io]

  // CHECK: firrtl.hierpath @nla1 [@Foo::@bar, @Bar::@w]
  firrtl.hierpath @nla1 [@Foo::@bar, @Bar::@baz0, @Baz::@w]

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
firrtl.hierpath @nla_5560 [@Bar0::@leaf, @Leaf::@w]
firrtl.hierpath @nla_5561 [@Bar1::@leaf, @Leaf::@w]
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
// on it should only be copied to a specific clone. This differs from the test
// above in that the annotation does not become a regular local annotation.
// CHECK-LABEL: firrtl.circuit "AnnotationSplit1"
firrtl.circuit "AnnotationSplit1" {
firrtl.hierpath @nla_5560 [@AnnotationSplit1::@bar0, @Bar0::@leaf, @Leaf::@w]
firrtl.hierpath @nla_5561 [@AnnotationSplit1::@bar1, @Bar1::@leaf, @Leaf::@w]
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
  firrtl.hierpath @nla_2 [@Inline::@bar, @Bar::@i]
  firrtl.hierpath @nla_1 [@Inline::@foo, @Foo::@bar, @Bar::@i]
  // CHECK:   firrtl.hierpath @nla_2 [@Inline::@bar, @Bar::@i]
  // CHECK:   firrtl.hierpath @nla_1 [@Inline::@[[bar_0:.+]], @Bar::@i]
  firrtl.module @Inline(in %i: !firrtl.uint<1>, out %o: !firrtl.uint<1>) {
    %foo_i, %foo_o = firrtl.instance foo sym @foo  @Foo(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    // CHECK:  = firrtl.instance foo_bar sym @[[bar_0]]  @Bar(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    %bar_i, %bar_o = firrtl.instance bar sym @bar  @Bar(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    // CHECK:  = firrtl.instance bar sym @bar  @Bar(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    firrtl.strictconnect %foo_i, %bar_i : !firrtl.uint<1>
    firrtl.strictconnect %bar_i, %i : !firrtl.uint<1>
    firrtl.strictconnect %o, %foo_o : !firrtl.uint<1>
  }
  firrtl.module private @Bar(in %i: !firrtl.uint<1> sym @i [{circt.nonlocal = @nla_1, class = "test_1"}, {circt.nonlocal = @nla_2, class = "test_2"}], out %o: !firrtl.uint<1>) {
    firrtl.strictconnect %o, %i : !firrtl.uint<1>
  }
  firrtl.module private @Foo(in %i: !firrtl.uint<1>, out %o: !firrtl.uint<1>) attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %bar_i, %bar_o = firrtl.instance bar sym @bar  @Bar(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    firrtl.strictconnect %bar_i, %i : !firrtl.uint<1>
    firrtl.strictconnect %o, %bar_o : !firrtl.uint<1>
  }
}

firrtl.circuit "Inline2"  {
  // CHECK-LABEL firrtl.circuit "Inline2"
  firrtl.hierpath @nla_1 [@Inline2::@foo, @Foo::@bar, @Bar::@i]
  // CHECK:  firrtl.hierpath @nla_1 [@Inline2::@[[bar_0:.+]], @Bar::@i]
  firrtl.module @Inline2(in %i: !firrtl.uint<1>, out %o: !firrtl.uint<1>) {
    %foo_i, %foo_o = firrtl.instance foo sym @foo  @Foo(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    %bar = firrtl.wire sym @bar  : !firrtl.uint<1>
    firrtl.strictconnect %foo_i, %bar : !firrtl.uint<1>
    firrtl.strictconnect %bar, %i : !firrtl.uint<1>
    firrtl.strictconnect %o, %foo_o : !firrtl.uint<1>
  }
  firrtl.module private @Bar(in %i: !firrtl.uint<1> sym @i [{circt.nonlocal = @nla_1, class = "testing"}], out %o: !firrtl.uint<1>) {
    firrtl.strictconnect %o, %i : !firrtl.uint<1>
  }
  firrtl.module private @Foo(in %i: !firrtl.uint<1>, out %o: !firrtl.uint<1>) attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %bar_i, %bar_o = firrtl.instance bar sym @bar  @Bar(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    // CHECK:  = firrtl.instance foo_bar sym @[[bar_0]]  @Bar(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    firrtl.strictconnect %bar_i, %i : !firrtl.uint<1>
    firrtl.strictconnect %o, %bar_o : !firrtl.uint<1>
  }
}
