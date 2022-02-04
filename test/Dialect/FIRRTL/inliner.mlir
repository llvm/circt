// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-inliner)' -allow-unregistered-dialect %s | FileCheck %s

// Test that an external module as the main module works.
firrtl.circuit "main_extmodule" {
  firrtl.extmodule @main_extmodule()
  firrtl.module @unused () { }
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
firrtl.module @unused () { }
firrtl.module @used () { }
firrtl.extmodule @unused_ext ()
firrtl.extmodule @used_ext ()
}
// CHECK-LABEL: firrtl.circuit "delete_dead_modules" {
// CHECK-NEXT:   firrtl.module @delete_dead_modules() {
// CHECK-NEXT:     firrtl.instance used @used()
// CHECK-NEXT:     firrtl.instance used @used_ext
// CHECK-NEXT:   }
// CHECK-NEXT:   firrtl.module @used() {
// CHECK-NEXT:   }
// CHECK-NEXT:   firrtl.extmodule @used_ext()
// CHECK-NEXT: }


// Test basic inlining
firrtl.circuit "inlining" {
firrtl.module @inlining() {
  firrtl.instance test1 @test1()
}
firrtl.module @test1()
  attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
  %test_wire = firrtl.wire : !firrtl.uint<2>
  firrtl.instance test2 @test2()
}
firrtl.module @test2()
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
firrtl.module @test1() {
  %test_wire = firrtl.wire : !firrtl.uint<2>
  firrtl.instance test2 @test2()
}
firrtl.module @test2() {
  %test_wire = firrtl.wire : !firrtl.uint<2>
}
}
// CHECK-LABEL: firrtl.circuit "flattening"
// CHECK-NEXT:   firrtl.module @flattening()
// CHECK-NOT:      annotations
// CHECK-NEXT:     %test1_test_wire = firrtl.wire : !firrtl.uint<2>
// CHECK-NEXT:     %test1_test2_test_wire = firrtl.wire : !firrtl.uint<2>
// CHECK-NOT:    firrtl.module @test1
// CHECK-NOT:    firrtl.module @test2


// Test that inlining and flattening compose well.
firrtl.circuit "compose" {
firrtl.module @compose() {
  firrtl.instance test1 @test1()
  firrtl.instance test2 @test2()
  firrtl.instance test3 @test3()
}
firrtl.module @test1() attributes {annotations =
        [{class = "firrtl.transforms.FlattenAnnotation"},
         {class = "firrtl.passes.InlineAnnotation"}]} {
  %test_wire = firrtl.wire : !firrtl.uint<2>
  firrtl.instance test2 @test2()
  firrtl.instance test3 @test3()
}
firrtl.module @test2() attributes {annotations =
        [{class = "firrtl.passes.InlineAnnotation"}]} {
  %test_wire = firrtl.wire : !firrtl.uint<2>
  firrtl.instance test3 @test3()
}
firrtl.module @test3() {
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
// CHECK-NEXT:   firrtl.module @test3() {
// CHECK-NEXT:     %test_wire = firrtl.wire  : !firrtl.uint<2>
// CHECK-NEXT:   }
// CHECK-NEXT: }


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
  // CHECK: %myinst_cmem = chirrtl.combmem : !chirrtl.cmemory<uint<8>, 8>
  %cmem = chirrtl.combmem : !chirrtl.cmemory<uint<8>, 8>
  // CHECK: %myinst_mem_read = firrtl.mem Undefined {depth = 1 : i64, name = "myinst_mem", portNames = ["read"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: sint<42>>
  %mem_read = firrtl.mem Undefined {depth = 1 : i64, name = "mem", portNames = ["read"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: sint<42>>
  // CHECK: %myinst_memoryport_data, %myinst_memoryport_port = chirrtl.memoryport Read %myinst_cmem {name = "myinst_memoryport"} : (!chirrtl.cmemory<uint<8>, 8>) -> (!firrtl.uint<8>, !chirrtl.cmemoryport)
  %memoryport_data, %memoryport_port = chirrtl.memoryport Read %cmem {name = "memoryport"} : (!chirrtl.cmemory<uint<8>, 8>) -> (!firrtl.uint<8>, !chirrtl.cmemoryport)
  chirrtl.memoryport.access %memoryport_port[%u8], %clock : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
  // CHECK: %myinst_node = firrtl.node %myinst_u8  : !firrtl.uint<8>
  %node = firrtl.node %u8 {name = "node"} : !firrtl.uint<8>
  // CHECK: %myinst_reg = firrtl.reg %myinst_clock : !firrtl.uint<8>
  %reg = firrtl.reg %clock {name = "reg"} : !firrtl.uint<8>
  // CHECK: %myinst_regreset = firrtl.regreset %myinst_clock, %myinst_reset, %c0_ui8 : !firrtl.asyncreset, !firrtl.uint<8>, !firrtl.uint<8>
  %regreset = firrtl.regreset %clock, %reset, %c0_ui8 : !firrtl.asyncreset, !firrtl.uint<8>, !firrtl.uint<8>
  // CHECK: %myinst_smem = chirrtl.seqmem Undefined : !chirrtl.cmemory<uint<8>, 8>
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
  // CHECK-NEXT: firrtl.nla @nla1 [#hw.innerNameRef<@NLAInlining::@bar>, @Bar]
  // CHECK-NEXT: firrtl.nla @nla2 [#hw.innerNameRef<@NLAInlining::@bar>, #hw.innerNameRef<@Bar::@a>]
  // CHECK-NEXT: firrtl.nla @nla3 [#hw.innerNameRef<@NLAInlining::@bar>, #hw.innerNameRef<@Bar::@port>]
  // CHECK-NOT:  firrtl.nla @nla4
  // CHECK-NOT:  firrtl.nla @nla5
  firrtl.nla @nla1 [#hw.innerNameRef<@NLAInlining::@foo>, #hw.innerNameRef<@Foo::@bar>, @Bar]
  firrtl.nla @nla2 [#hw.innerNameRef<@NLAInlining::@foo>, #hw.innerNameRef<@Foo::@bar>, #hw.innerNameRef<@Bar::@a>]
  firrtl.nla @nla3 [#hw.innerNameRef<@NLAInlining::@foo>, #hw.innerNameRef<@Foo::@bar>, #hw.innerNameRef<@Bar::@port>]
  firrtl.nla @nla4 [#hw.innerNameRef<@NLAInlining::@foo>, @Foo]
  firrtl.nla @nla5 [#hw.innerNameRef<@NLAInlining::@foo>, #hw.innerNameRef<@Foo::@b>]
  firrtl.nla @nla6 [#hw.innerNameRef<@NLAInlining::@foo>, #hw.innerNameRef<@Foo::@port>]
  // CHECK-NEXT: firrtl.module @Bar
  // CHECK-SAME: %port: {{.+}} sym @port [{circt.nonlocal = @nla3, class = "nla3"}]
  // CHECK-SAME: [{circt.nonlocal = @nla1, class = "nla1"}]
  firrtl.module @Bar(
    in %port: !firrtl.uint<1> sym @port [{circt.nonlocal = @nla3, class = "nla3"}]
  ) attributes {annotations = [{circt.nonlocal = @nla1, class = "nla1"}]} {
    %a = firrtl.wire sym @a {annotations = [{circt.nonlocal = @nla2, class = "nla2"}]} : !firrtl.uint<1>
  }
  firrtl.module @Foo(in %port: !firrtl.uint<1> sym @port [{circt.nonlocal = @nla6, class = "nla6"}]) attributes {annotations = [
  {class = "firrtl.passes.InlineAnnotation"}, {circt.nonlocal = @nla4, class = "nla4"}]} {
    %bar_port = firrtl.instance bar sym @bar {annotations = [
      {circt.nonlocal = @nla1, class = "circt.nonlocal"},
      {circt.nonlocal = @nla2, class = "circt.nonlocal"},
      {circt.nonlocal = @nla3, class = "circt.nonlocal"}]} @Bar(in port: !firrtl.uint<1>)
    %b = firrtl.wire sym @b {annotations = [{circt.nonlocal = @nla5, class = "nla5"}]} : !firrtl.uint<1>
  }
  // CHECK: firrtl.module @NLAInlining
  firrtl.module @NLAInlining() {
    %foo_port = firrtl.instance foo sym @foo {annotations = [
      {circt.nonlocal = @nla1, class = "circt.nonlocal"},
      {circt.nonlocal = @nla2, class = "circt.nonlocal"},
      {circt.nonlocal = @nla3, class = "circt.nonlocal"},
      {circt.nonlocal = @nla4, class = "circt.nonlocal"},
      {circt.nonlocal = @nla5, class = "circt.nonlocal"},
      {circt.nonlocal = @nla6, class = "circt.nonlocal"}]} @Foo(in port: !firrtl.uint<1>)
    // CHECK-NEXT: %foo_port = firrtl.wire {{.+}} [{class = "nla6"}]
    // CHECK-NEXT: firrtl.instance foo_bar {{.+}}annotations = [
    // CHECK-NOT: @nla4
    // CHECK-NOT: @nla5
    // CHECK-NOT: @nla6
    // CHECK-SAME: {circt.nonlocal = @nla1, class = "circt.nonlocal"}
    // CHECK-SAME: {circt.nonlocal = @nla2, class = "circt.nonlocal"}
    // CHECK-SAME: {circt.nonlocal = @nla3, class = "circt.nonlocal"}
    // CHECK-NEXT: %foo_b = firrtl.wire {{.+}} [{class = "nla5"}]
  }
}

// Test NLA handling during inlining for situations where the NLA does NOT start
// at the root.  This checks that the NLA, on either a component or a port, is
// properly copied for each new instantiation.
//
// CHECK-LABEL: firrtl.circuit "NLAInliningNotMainRoot"
firrtl.circuit "NLAInliningNotMainRoot" {
  // CHECK-NEXT: firrtl.nla @nla1 [#hw.innerNameRef<@NLAInliningNotMainRoot::@baz>, #hw.innerNameRef<@Baz::@a>]
  // CHECK-NEXT: firrtl.nla @nla1_0 [#hw.innerNameRef<@Foo::@baz>, #hw.innerNameRef<@Baz::@a>]
  // CHECK-NEXT: firrtl.nla @nla2 [#hw.innerNameRef<@NLAInliningNotMainRoot::@baz>, #hw.innerNameRef<@Baz::@port>]
  // CHECK-NEXT: firrtl.nla @nla2_0 [#hw.innerNameRef<@Foo::@baz>, #hw.innerNameRef<@Baz::@port>]
  firrtl.nla @nla1 [#hw.innerNameRef<@Bar::@baz>, #hw.innerNameRef<@Baz::@a>]
  firrtl.nla @nla2 [#hw.innerNameRef<@Bar::@baz>, #hw.innerNameRef<@Baz::@port>]
  // CHECK: firrtl.module @Baz
  // CHECK-SAME: %port: {{.+}} [{circt.nonlocal = @nla2, class = "nla2"}, {circt.nonlocal = @nla2_0, class = "nla2"}]
  firrtl.module @Baz(
    in %port: !firrtl.uint<1> sym @port [{circt.nonlocal = @nla2, class = "nla2"}]
  ) {
    // CHECK-NEXT: firrtl.wire {{.+}} [{circt.nonlocal = @nla1, class = "hello"}, {circt.nonlocal = @nla1_0, class = "hello"}]
    %a = firrtl.wire sym @a {annotations = [{circt.nonlocal = @nla1, class = "hello"}]} : !firrtl.uint<1>
  }
  firrtl.module @Bar() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %baz_port = firrtl.instance baz sym @baz {annotations = [
      {circt.nonlocal = @nla1, class = "circt.nonlocal"},
      {circt.nonlocal = @nla2, class = "circt.nonlocal"}
    ]} @Baz(in port: !firrtl.uint<1>)
  }
  // CHECK: firrtl.module @Foo
  firrtl.module @Foo() {
    // CHECK-NEXT: firrtl.instance bar_baz {{.+}} [{circt.nonlocal = @nla1_0, class = "circt.nonlocal"}, {circt.nonlocal = @nla2_0, class = "circt.nonlocal"}]
    firrtl.instance bar @Bar()
  }
  // CHECK: firrtl.module @NLAInliningNotMainRoot
  firrtl.module @NLAInliningNotMainRoot() {
    firrtl.instance foo @Foo()
    // CHECK: firrtl.instance bar_baz {{.+}} [{circt.nonlocal = @nla1, class = "circt.nonlocal"}, {circt.nonlocal = @nla2, class = "circt.nonlocal"}]
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
  // CHECK-NEXT: firrtl.nla @nla1 [#hw.innerNameRef<@NLAFlattening::@foo>, #hw.innerNameRef<@Foo::@a>]
  // CHECK-NEXT: firrtl.nla @nla2 [#hw.innerNameRef<@NLAFlattening::@foo>, #hw.innerNameRef<@Foo::@port>]
  // CHECK-NOT:  firrtl.nla @nla3
  // CHECK-NOT:  firrtl.nla @nla4
  firrtl.nla @nla1 [#hw.innerNameRef<@NLAFlattening::@foo>, #hw.innerNameRef<@Foo::@bar>, #hw.innerNameRef<@Bar::@baz>, #hw.innerNameRef<@Baz::@a>]
  firrtl.nla @nla2 [#hw.innerNameRef<@NLAFlattening::@foo>, #hw.innerNameRef<@Foo::@bar>, #hw.innerNameRef<@Bar::@baz>, #hw.innerNameRef<@Baz::@port>]
  firrtl.nla @nla3 [#hw.innerNameRef<@NLAFlattening::@foo>, #hw.innerNameRef<@Foo::@bar>, #hw.innerNameRef<@Bar::@baz>, @Baz]
  firrtl.nla @nla4 [#hw.innerNameRef<@Foo::@bar>, #hw.innerNameRef<@Bar::@b>]
  firrtl.module @Baz(
    in %port: !firrtl.uint<1> sym @port [{circt.nonlocal = @nla2, class = "nla2"}]
  ) attributes {annotations = [{circt.nonlocal = @nla3, class = "nla3"}]} {
    %a = firrtl.wire sym @a {annotations = [{circt.nonlocal = @nla1, class = "nla1"}]} : !firrtl.uint<1>
  }
  firrtl.module @Bar() {
    firrtl.instance baz sym @baz {annotations = [
      {circt.nonlocal = @nla1, class = "circt.nonlocal"},
      {circt.nonlocal = @nla2, class = "circt.nonlocal"},
      {circt.nonlocal = @nla3, class = "circt.nonlocal"}
    ]} @Baz(in port: !firrtl.uint<1>)
    %b = firrtl.wire sym @b {annotations = [{circt.nonlocal = @nla4, class = "nla4"}]} : !firrtl.uint<1>
  }
  // CHECK: firrtl.module @Foo
  firrtl.module @Foo() attributes {annotations = [{class = "firrtl.transforms.FlattenAnnotation"}]} {
    firrtl.instance bar sym @bar {annotations = [
      {circt.nonlocal = @nla1, class = "circt.nonlocal"},
      {circt.nonlocal = @nla2, class = "circt.nonlocal"},
      {circt.nonlocal = @nla3, class = "circt.nonlocal"},
      {circt.nonlocal = @nla4, class = "circt.nonlocal"}
    ]} @Bar()
    // CHECK-NEXT: %bar_baz_port = firrtl.wire sym @port {{.+}} [{circt.nonlocal = @nla2, class = "nla2"}]
    // CHECK-NEXT: %bar_baz_a = firrtl.wire {{.+}} [{circt.nonlocal = @nla1, class = "nla1"}]
    // CHECK-NEXT: %bar_b = firrtl.wire {{.+}} [{class = "nla4"}]
  }
  // CHECK: firrtl.module @NLAFlattening
  firrtl.module @NLAFlattening() {
    // CHECK-NEXT: firrtl.instance foo {{.+}} [{circt.nonlocal = @nla1, class = "circt.nonlocal"}, {circt.nonlocal = @nla2, class = "circt.nonlocal"}]
    firrtl.instance foo sym @foo {annotations = [
      {circt.nonlocal = @nla1, class = "circt.nonlocal"},
      {circt.nonlocal = @nla2, class = "circt.nonlocal"},
      {circt.nonlocal = @nla3, class = "circt.nonlocal"}
    ]} @Foo()
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
  // CHECK-NOT:  firrtl.nla @nla1
  // CHECK-NOT:  firrtl.nla @nla2
  // CHECK-NEXT: firrtl.nla @nla3 [#hw.innerNameRef<@Baz::@quz>, #hw.innerNameRef<@Quz::@b>]
  // CHECK-NEXT: firrtl.nla @nla4 [#hw.innerNameRef<@Baz::@quz>, #hw.innerNameRef<@Quz::@Quz_port>]
  firrtl.nla @nla1 [#hw.innerNameRef<@Bar::@qux>, #hw.innerNameRef<@Qux::@a>]
  firrtl.nla @nla2 [#hw.innerNameRef<@Bar::@qux>, #hw.innerNameRef<@Qux::@Qux_port>]
  firrtl.nla @nla3 [#hw.innerNameRef<@Baz::@quz>, #hw.innerNameRef<@Quz::@b>]
  firrtl.nla @nla4 [#hw.innerNameRef<@Baz::@quz>, #hw.innerNameRef<@Quz::@Quz_port>]
  // CHECK: firrtl.module @Quz
  // CHECK-SAME: in %port: {{.+}} [{circt.nonlocal = @nla4, class = "nla4"}]
  firrtl.module @Quz(
    in %port: !firrtl.uint<1> sym @Quz_port [{circt.nonlocal = @nla4, class = "nla4"}]
  ) {
    // CHECK-NEXT: firrtl.wire {{.+}} [{circt.nonlocal = @nla3, class = "nla3"}]
    %b = firrtl.wire sym @b {annotations = [{circt.nonlocal = @nla3, class = "nla3"}]} : !firrtl.uint<1>
  }
  firrtl.module @Qux(
    in %port: !firrtl.uint<1> sym @Qux_port [{circt.nonlocal = @nla2, class = "nla2"}]
  ) {
    %a = firrtl.wire sym @a {annotations = [{circt.nonlocal = @nla1, class = "nla1"}]} : !firrtl.uint<1>
  }
  // CHECK: firrtl.module @Baz
  firrtl.module @Baz() {
    // CHECK-NEXT: firrtl.instance {{.+}} [{circt.nonlocal = @nla3, class = "circt.nonlocal"}, {circt.nonlocal = @nla4, class = "circt.nonlocal"}]
    firrtl.instance quz sym @quz {annotations = [
      {circt.nonlocal = @nla3, class = "circt.nonlocal"},
      {circt.nonlocal = @nla4, class = "circt.nonlocal"}
    ]} @Quz(in port: !firrtl.uint<1>)
  }
  firrtl.module @Bar() {
    firrtl.instance qux sym @qux {annotations = [
      {circt.nonlocal = @nla1, class = "circt.nonlocal"},
      {circt.nonlocal = @nla2, class = "circt.nonlocal"}
    ]} @Qux(in port: !firrtl.uint<1>)
  }
  // CHECK: firrtl.module @Foo
  firrtl.module @Foo() attributes {annotations = [{class = "firrtl.transforms.FlattenAnnotation"}]} {
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
  // CHECK-NEXT: firrtl.nla @nla1 [#hw.innerNameRef<@CollidingSymbols::@[[FoobarSym:[_a-zA-Z0-9]+]]>, @Bar]
  firrtl.nla @nla1 [#hw.innerNameRef<@CollidingSymbols::@foo>, #hw.innerNameRef<@Foo::@bar>, @Bar]
  firrtl.module @Bar() attributes {annotations = [{circt.nonlocal = @nla1, class = "nla1"}]} {}
  firrtl.module @Foo() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %b = firrtl.wire sym @b : !firrtl.uint<1>
    firrtl.instance bar sym @bar {annotations = [
      {circt.nonlocal = @nla1, class = "circt.nonlocal"}
    ]} @Bar()
  }
  // CHECK:      firrtl.module @CollidingSymbols
  // CHECK-NEXT:   firrtl.wire sym @[[inlinedSymbol:[_a-zA-Z0-9]+]]
  // CHECK-NEXT:   firrtl.instance foo_bar sym @[[FoobarSym]]
  // CHECK-SAME:     {circt.nonlocal = @nla1, class = "circt.nonlocal"}
  // CHECK-NOT:    firrtl.wire sym @[[inlinedSymbol]]
  // CHECK-NOT:    firrtl.wire sym @[[FoobarSym]]
  firrtl.module @CollidingSymbols() {
    firrtl.instance foo sym @foo {annotations = [
      {circt.nonlocal = @nla1, class = "circt.nonlocal"}
    ]} @Foo()
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
  // CHECK-NEXT: firrtl.nla @nla1 [#hw.innerNameRef<@CollidingSymbolsPort::@foo>, #hw.innerNameRef<@Foo::@[[BarbSym:[_a-zA-Z0-9]+]]>]
  firrtl.nla @nla1 [#hw.innerNameRef<@CollidingSymbolsPort::@foo>, #hw.innerNameRef<@Foo::@bar>, #hw.innerNameRef<@Bar::@b>]
  // CHECK-NOT: firrtl.module @Bar
  firrtl.module @Bar(
    in %b: !firrtl.uint<1> sym @b [{circt.nonlocal = @nla1, class = "nla1"}]
  ) attributes {annotations = [
    {class = "firrtl.passes.InlineAnnotation"}
  ]} {}
  // CHECK-NEXT: firrtl.module @Foo
  firrtl.module @Foo() {
    // CHECK-NEXT: firrtl.wire sym @[[BarbSym]] {annotations = [{circt.nonlocal = @nla1, class = "nla1"}]}
    firrtl.instance bar sym @bar {annotations = [
      {circt.nonlocal = @nla1, class = "circt.nonlocal"}
    ]} @Bar(in b: !firrtl.uint<1>)
    // CHECK-NEXT: firrtl.wire sym @b
    %colliding_b = firrtl.wire sym @b : !firrtl.uint<1>
  }
  firrtl.module @CollidingSymbolsPort() {
    firrtl.instance foo sym @foo {annotations = [
      {circt.nonlocal = @nla1, class = "circt.nonlocal"}
    ]} @Foo()
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
  // CHECK-NEXT: firrtl.nla @nla1 [#hw.innerNameRef<@CollidingSymbolsReTop::@[[TopbazSym:[_a-zA-Z0-9]+]]>, #hw.innerNameRef<@Baz::@a>]
  // CHECK-NEXT: firrtl.nla @nla1_0 [#hw.innerNameRef<@Foo::@[[FoobazSym:[_a-zA-Z0-9]+]]>, #hw.innerNameRef<@Baz::@a>]
  firrtl.nla @nla1 [#hw.innerNameRef<@Bar::@baz>, #hw.innerNameRef<@Baz::@a>]
  // CHECK: firrtl.module @Baz
  firrtl.module @Baz() {
    // CHECK-NEXT: firrtl.wire {{.+}} [{circt.nonlocal = @nla1, class = "hello"}, {circt.nonlocal = @nla1_0, class = "hello"}]
    %a = firrtl.wire sym @a {annotations = [{circt.nonlocal = @nla1, class = "hello"}]} : !firrtl.uint<1>
  }
  firrtl.module @Bar() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    firrtl.instance baz sym @baz {annotations = [
      {circt.nonlocal = @nla1, class = "circt.nonlocal"}
    ]} @Baz()
  }
  // CHECK: firrtl.module @Foo
  firrtl.module @Foo() {
    // CHECK-NEXT: firrtl.instance bar_baz sym @[[FoobazSym]] {{.+}} [{circt.nonlocal = @nla1_0, class = "circt.nonlocal"}]
    firrtl.instance bar @Bar()
    %colliding_baz = firrtl.wire sym @baz : !firrtl.uint<1>
    %colliding_baz_0 = firrtl.wire sym @baz_0 : !firrtl.uint<1>
  }
  // CHECK: firrtl.module @CollidingSymbolsReTop
  firrtl.module @CollidingSymbolsReTop() {
    firrtl.instance foo @Foo()
    // CHECK: firrtl.instance bar_baz sym @[[TopbazSym]]{{.+}} [{circt.nonlocal = @nla1, class = "circt.nonlocal"}]
    firrtl.instance bar @Bar()
    firrtl.instance baz @Baz()
    %colliding_baz = firrtl.wire sym @baz : !firrtl.uint<1>
  }
}

// Test that anything with a "name" will be renamed, even things that FIRRTL
// Dialect doesn't understand.
//
// CHECK-LABEL: firrtl.circuit "RenameAnything"
firrtl.circuit "RenameAnything" {
  firrtl.module @Foo() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    "some_unknown_dialect.op"() { name = "world" } : () -> ()
  }
  // CHECK-NEXT: firrtl.module @RenameAnything
  firrtl.module @RenameAnything() {
    // CHECK-NEXT: "some_unknown_dialect.op"(){{.+}}name = "hello_world"
    firrtl.instance hello @Foo()
  }
}
