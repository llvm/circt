// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-inliner))' -allow-unregistered-dialect %s | FileCheck %s

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
  // CHECK-NEXT: hw.hierpath private @nla1 [@NLAInlining::@bar, @Bar]
  // CHECK-NEXT: hw.hierpath private @nla2 [@NLAInlining::@bar, @Bar::@a]
  // CHECK-NEXT: hw.hierpath private @nla3 [@NLAInlining::@bar, @Bar::@port]
  // CHECK-NOT:  hw.hierpath private @nla4
  // CHECK-NOT:  hw.hierpath private @nla5
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

// Test NLA handling during inlining for situations where the NLA does NOT start
// at the root.  This checks that the NLA, on either a component or a port, is
// properly copied for each new instantiation.
//
// CHECK-LABEL: firrtl.circuit "NLAInliningNotMainRoot"
firrtl.circuit "NLAInliningNotMainRoot" {
  // CHECK-NEXT: hw.hierpath private @nla1 [@NLAInliningNotMainRoot::@baz, @Baz::@a]
  // CHECK-NEXT: hw.hierpath private @nla1_0 [@Foo::@baz, @Baz::@a]
  // CHECK-NEXT: hw.hierpath private @nla2 [@NLAInliningNotMainRoot::@baz, @Baz::@port]
  // CHECK-NEXT: hw.hierpath private @nla2_0 [@Foo::@baz, @Baz::@port]
  hw.hierpath private @nla1 [@Bar::@baz, @Baz::@a]
  hw.hierpath private @nla2 [@Bar::@baz, @Baz::@port]
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
  // CHECK-NEXT: hw.hierpath private @nla1 [@NLAFlattening::@foo, @Foo::@a]
  // CHECK-NEXT: hw.hierpath private @nla2 [@NLAFlattening::@foo, @Foo::@port]
  // CHECK-NOT:  hw.hierpath private @nla3
  // CHECK-NOT:  hw.hierpath private @nla4
  hw.hierpath private @nla1 [@NLAFlattening::@foo, @Foo::@bar, @Bar::@baz, @Baz::@a]
  hw.hierpath private @nla2 [@NLAFlattening::@foo, @Foo::@bar, @Bar::@baz, @Baz::@port]
  hw.hierpath private @nla3 [@NLAFlattening::@foo, @Foo::@bar, @Bar::@baz, @Baz]
  hw.hierpath private @nla4 [@Foo::@bar, @Bar::@b]
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
  // CHECK-NOT:  hw.hierpath private @nla1
  // CHECK-NOT:  hw.hierpath private @nla2
  // CHECK-NEXT: hw.hierpath private @nla3 [@Baz::@quz, @Quz::@b]
  // CHECK-NEXT: hw.hierpath private @nla4 [@Baz::@quz, @Quz::@Quz_port]
  hw.hierpath private @nla1 [@Bar::@qux, @Qux::@a]
  hw.hierpath private @nla2 [@Bar::@qux, @Qux::@Qux_port]
  hw.hierpath private @nla3 [@Baz::@quz, @Quz::@b]
  hw.hierpath private @nla4 [@Baz::@quz, @Quz::@Quz_port]
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
  // CHECK-NOT:  #hw.innerNameRef<@CollidingSymbolsReTop::@baz>
  // CHECK-NOT:  #hw.innerNameRef<@Foo::@baz>
  // CHECK-NEXT: hw.hierpath private @nla1 [@CollidingSymbolsReTop::@[[TopbazSym:[_a-zA-Z0-9]+]], @Baz::@a]
  // CHECK-NEXT: hw.hierpath private @nla1_0 [@Foo::@[[FoobazSym:[_a-zA-Z0-9]+]], @Baz::@a]
  hw.hierpath private @nla1 [@Bar::@baz, @Baz::@a]
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
// on it should only be copied to a specific clone. This differs from the test
// above in that the annotation does not become a regular local annotation.
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

// Test for U-Turn in property ports. The inlined module propassign's and uses the property.
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
// CHECK-NOT:  hw.hierpath private @nla1
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
  // CHECK-NOT: @dutNLA
  hw.hierpath private @dutNLA [@RemoveNonLocalFromLocal::@sym]
  firrtl.module @Bar() {}
  // CHECK-LABEL: firrtl.module @RemoveNonLocalFromLocal
  firrtl.module @RemoveNonLocalFromLocal() {
    // CHECK: firrtl.instance bar sym @sym {annotations = [{class = "circt.tracker", id = distinct[0]<>}]} @Bar()
    firrtl.instance bar sym @sym {annotations = [{circt.nonlocal = @dutNLA, class = "circt.tracker", id = distinct[0]<>}]} @Bar()
  }
}
