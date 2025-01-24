// RUN: circt-opt -pass-pipeline='builtin.module(firrtl-imdeadcodeelim)' --split-input-file --verify-diagnostics --allow-unregistered-dialect %s | FileCheck %s
firrtl.circuit "top" {
  // In `dead_module`, %source is connected to %dest through several dead operations such as
  // node, wire, reg or rgereset. %dest is also dead at any instantiation, so check that
  // all operations are removed by IMDeadCodeElim pass.
  // CHECK-NOT: @dead_module
  firrtl.module private @dead_module(in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>,
                                     in %clock:!firrtl.clock, in %reset:!firrtl.uint<1>) {
    %dead_node = firrtl.node %source: !firrtl.uint<1>

    %dead_wire = firrtl.wire : !firrtl.uint<1>
    firrtl.matchingconnect %dead_wire, %dead_node : !firrtl.uint<1>

    %dead_reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<1>
    firrtl.matchingconnect %dead_reg, %dead_wire : !firrtl.uint<1>

    %dead_reg_reset = firrtl.regreset %clock, %reset, %dead_reg  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.matchingconnect %dead_reg_reset, %dead_reg : !firrtl.uint<1>

    %not = firrtl.not %dead_reg_reset : (!firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.matchingconnect %dest, %not : !firrtl.uint<1>
  }

  // `%dontTouch` port has a symbol so it shouldn't be removed. `%sym_wire` also has a
  // symbol so check  that `%source` is preserved too.
  // CHECK-LABEL: firrtl.module private @dontTouch(in %dontTouch: !firrtl.uint<1> sym @sym, in %source: !firrtl.uint<1>) {
  firrtl.module private @dontTouch(in %dontTouch: !firrtl.uint<1> sym @sym, in %source: !firrtl.uint<1>, in %dead: !firrtl.uint<1>) {
    // CHECK-NEXT: %sym_wire = firrtl.wire sym @sym2   : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.matchingconnect %sym_wire, %source : !firrtl.uint<1>
    // CHECK-NEXT: }
    %sym_wire = firrtl.wire sym @sym2 : !firrtl.uint<1>
    firrtl.matchingconnect %sym_wire, %source : !firrtl.uint<1>

  }

  // CHECK-LABEL: firrtl.module private @mem(in %source: !firrtl.uint<1>) {
  firrtl.module private @mem(in %source: !firrtl.uint<1>) {
    // CHECK-NEXT: %ReadMemory_read0 = firrtl.mem Undefined {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}], depth = 16 : i64, name = "ReadMemory", portNames = ["read0"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>
    %mem = firrtl.mem Undefined {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}], depth = 16 : i64, name = "ReadMemory", portNames = ["read0"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>
    // CHECK-NEXT: %0 = firrtl.subfield %ReadMemory_read0[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>
    // CHECK-NEXT: firrtl.connect %0, %source : !firrtl.uint<4>, !firrtl.uint<1>
    // CHECK-NEXT: }
    %0 = firrtl.subfield %mem[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>
    firrtl.connect %0, %source : !firrtl.uint<4>, !firrtl.uint<1>
  }

  // Ports of public modules should not be modified.
  // CHECK-LABEL: firrtl.module @top(in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>, in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
  firrtl.module @top(in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>,
                     in %clock:!firrtl.clock, in %reset:!firrtl.uint<1>) {
    // CHECK-NEXT: %tmp = firrtl.node %source
    // CHECK-NEXT: firrtl.matchingconnect %dest, %tmp
    %tmp = firrtl.node %source: !firrtl.uint<1>
    firrtl.matchingconnect %dest, %tmp : !firrtl.uint<1>

    // CHECK-NOT: @dead_module
    %source1, %dest1, %clock1, %reset1  = firrtl.instance dead_module @dead_module(in source: !firrtl.uint<1>, out dest: !firrtl.uint<1>, in clock:!firrtl.clock, in reset:!firrtl.uint<1>)
    firrtl.matchingconnect %source1, %source : !firrtl.uint<1>
    firrtl.matchingconnect %clock1, %clock : !firrtl.clock
    firrtl.matchingconnect %reset1, %reset : !firrtl.uint<1>

    // Check that ports with dontTouch are not removed.
    // CHECK-NEXT: %testDontTouch_dontTouch, %testDontTouch_source = firrtl.instance testDontTouch @dontTouch(in dontTouch: !firrtl.uint<1>, in source: !firrtl.uint<1>)
    // CHECK-NEXT: firrtl.matchingconnect %testDontTouch_dontTouch, %source
    // CHECK-NEXT: firrtl.matchingconnect %testDontTouch_source, %source
    %testDontTouch_dontTouch, %testDontTouch_source,  %dead = firrtl.instance testDontTouch @dontTouch(in dontTouch: !firrtl.uint<1>, in source: !firrtl.uint<1>, in dead:!firrtl.uint<1>)
    firrtl.matchingconnect %testDontTouch_dontTouch, %source : !firrtl.uint<1>
    firrtl.matchingconnect %testDontTouch_source, %source : !firrtl.uint<1>
    firrtl.matchingconnect %dead, %source : !firrtl.uint<1>

    // CHECK-NEXT: %mem_source = firrtl.instance mem @mem(in source: !firrtl.uint<1>)
    // CHECK-NEXT: firrtl.matchingconnect %mem_source, %source : !firrtl.uint<1>
    %mem_source  = firrtl.instance mem @mem(in source: !firrtl.uint<1>)
    firrtl.matchingconnect %mem_source, %source : !firrtl.uint<1>
    // CHECK-NEXT: }
  }
}

// -----

// Check that it's possible to analyze complex dependency across different modules.
firrtl.circuit "top"  {
  // CHECK-NOT: @Child1
  firrtl.module private @Child1(in %input: !firrtl.uint<1>, out %output: !firrtl.uint<1>) {
    firrtl.matchingconnect %output, %input : !firrtl.uint<1>
  }
  // CHECK-NOT: @Child2
  firrtl.module private @Child2(in %input: !firrtl.uint<1>, in %clock: !firrtl.clock, out %output: !firrtl.uint<1>) {
    %r = firrtl.reg %clock  : !firrtl.clock, !firrtl.uint<1>
    firrtl.matchingconnect %r, %input : !firrtl.uint<1>
    firrtl.matchingconnect %output, %r : !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module @top(in %clock: !firrtl.clock, in %input: !firrtl.uint<1>) {
  // CHECK-NEXT:  }
  // expected-warning @below {{module `top` is empty but cannot be removed because the module is public}}
  firrtl.module @top(in %clock: !firrtl.clock, in %input: !firrtl.uint<1>) {
    %tile_input, %tile_output = firrtl.instance tile  @Child1(in input: !firrtl.uint<1>, out output: !firrtl.uint<1>)
    firrtl.matchingconnect %tile_input, %input : !firrtl.uint<1>
    %named = firrtl.node  %tile_output  : !firrtl.uint<1>
    %bar_input, %bar_clock, %bar_output = firrtl.instance bar  @Child2(in input: !firrtl.uint<1>, in clock: !firrtl.clock, out output: !firrtl.uint<1>)
    firrtl.matchingconnect %bar_clock, %clock : !firrtl.clock
    firrtl.matchingconnect %bar_input, %named : !firrtl.uint<1>
  }
}

// -----

// CHECK-LABEL: "UnusedOutput"
firrtl.circuit "UnusedOutput"  {
  // CHECK: firrtl.module {{.+}}@SingleDriver
  // CHECK-NOT:     out %c
  firrtl.module private @SingleDriver(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>) {
    // CHECK-NEXT: %[[c_wire:.+]] = firrtl.wire
    // CHECK-NEXT: firrtl.matchingconnect %b, %[[c_wire]]
    firrtl.matchingconnect %b, %c : !firrtl.uint<1>
    // CHECK-NEXT: %[[not_a:.+]] = firrtl.not %a
    %0 = firrtl.not %a : (!firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK-NEXT: firrtl.matchingconnect %[[c_wire]], %[[not_a]]
    firrtl.matchingconnect %c, %0 : !firrtl.uint<1>
  }
  // CHECK-LABEL: @UnusedOutput
  firrtl.module @UnusedOutput(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    // CHECK: %singleDriver_a, %singleDriver_b = firrtl.instance singleDriver
    %singleDriver_a, %singleDriver_b, %singleDriver_c = firrtl.instance singleDriver @SingleDriver(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>, out c: !firrtl.uint<1>)
    firrtl.matchingconnect %singleDriver_a, %a : !firrtl.uint<1>
    firrtl.matchingconnect %b, %singleDriver_b : !firrtl.uint<1>
  }
}

// -----

// Ensure that the "output_file" attribute isn't destroyed by IMDeadCodeElim.
// This matters for interactions between Grand Central (which sets these) and
// IMDeadCodeElim which may clone modules with stripped ports.
//
// CHECK-LABEL: "PreserveOutputFile"
firrtl.circuit "PreserveOutputFile" {
  // CHECK-NEXT: firrtl.module {{.+}}@Sub
  // CHECK-NOT:    %a
  // CHECK-SAME:   output_file
  // expected-warning @+1{{module `Sub` is empty but cannot be removed because the module has ports "b" are referenced by name or dontTouched}}
  firrtl.module private @Sub(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1> sym @sym) attributes {output_file = #hw.output_file<"hello">} {}
  // CHECK: firrtl.module @PreserveOutputFile
  firrtl.module @PreserveOutputFile() {
    // CHECK-NEXT: firrtl.instance sub
    // CHECK-SAME: output_file
    firrtl.instance sub {output_file = #hw.output_file<"hello">} @Sub(in a: !firrtl.uint<1>, in b: !firrtl.uint<1>)
  }
}

// -----

// CHECK-LABEL: "DeleteEmptyModule"
firrtl.circuit "DeleteEmptyModule" {
  // CHECK: firrtl.module private @empty
  // expected-warning @+1{{module `empty` is empty but cannot be removed because the module has annotations [{class = "foo"}]}}
  firrtl.module private @empty() attributes {annotations = [{class = "foo"}]}  {}
  // CHECK-NOT: firrtl.module private @Sub
  firrtl.module private @Sub(in %a: !firrtl.uint<1>)  {}
  // CHECK: firrtl.module @DeleteEmptyModule
  firrtl.module @DeleteEmptyModule() {
    // CHECK-NOT: firrtl.instance sub1
    firrtl.instance sub1 sym @Foo @Sub(in a: !firrtl.uint<1>)
    // CHECK-NOT: sub2
    firrtl.instance sub2 @Sub(in a: !firrtl.uint<1>)
    // CHECK: empty
    firrtl.instance empty @empty()
  }
}

// -----

// CHECK-LABEL: "ForwardConstant"
firrtl.circuit "ForwardConstant" {
  // CHECK-NOT: Zero
  firrtl.module private @Zero(out %zero: !firrtl.uint<1>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    firrtl.matchingconnect %zero, %c0_ui1 : !firrtl.uint<1>
  }
  // CHECK-LABEL: @ForwardConstant
  firrtl.module @ForwardConstant(out %zero: !firrtl.uint<1>) {
    // CHECK: %c0_ui1 = firrtl.constant 0
    %sub_zero = firrtl.instance sub @Zero(out zero: !firrtl.uint<1>)
    // CHECK-NEXT: firrtl.matchingconnect %zero, %c0_ui1
    firrtl.matchingconnect %zero, %sub_zero : !firrtl.uint<1>
  }
}

// -----

// Test handling of ref ports and ops.

// CHECK-LABEL: "RefPorts"
firrtl.circuit "RefPorts" {
  // CHECK-NOT: @dead_ref_send
  firrtl.module private @dead_ref_send(in %source: !firrtl.uint<1>, out %dest: !firrtl.probe<uint<1>>) {
    %ref = firrtl.ref.send %source: !firrtl.uint<1>
    firrtl.ref.define %dest, %ref : !firrtl.probe<uint<1>>
  }

  // CHECK-LABEL: @dead_ref_port
  // CHECK-NOT: firrtl.ref
  firrtl.module private @dead_ref_port(in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>, out %ref_dest: !firrtl.probe<uint<1>>) {
    %ref_not = firrtl.ref.send %source: !firrtl.uint<1>
    firrtl.ref.define %ref_dest, %ref_not : !firrtl.probe<uint<1>>
    firrtl.matchingconnect %dest, %source : !firrtl.uint<1>
  }

  // CHECK: @live_ref
  firrtl.module private @live_ref(in %source: !firrtl.uint<1>, out %dest: !firrtl.probe<uint<1>>) {
    %ref_source = firrtl.ref.send %source: !firrtl.uint<1>
    firrtl.ref.define %dest, %ref_source : !firrtl.probe<uint<1>>
  }

  // CHECK-LABEL: @RefPorts
  firrtl.module @RefPorts(in %source : !firrtl.uint<1>, out %dest : !firrtl.uint<1>) {
    // Delete send's that aren't resolved, and check deletion of modules with ref ops + ports.
    // CHECK-NOT: @dead_ref_send
    %source1, %dest1 = firrtl.instance dead_ref_send @dead_ref_send(in source: !firrtl.uint<1>, out dest: !firrtl.probe<uint<1>>)
    firrtl.matchingconnect %source1, %source : !firrtl.uint<1>

    // Check that an unused resolve doesn't keep send alive, and test ref port removal.
    // CHECK: @dead_ref_port
    // CHECK-NOT: firrtl.ref
    %source2, %dest2, %ref_dest2 = firrtl.instance dead_ref_port @dead_ref_port(in source: !firrtl.uint<1>, out dest: !firrtl.uint<1>, out ref_dest: !firrtl.probe<uint<1>>)
    firrtl.matchingconnect %source2, %source : !firrtl.uint<1>
    %unused = firrtl.ref.resolve %ref_dest2 : !firrtl.probe<uint<1>>
    firrtl.matchingconnect %dest, %dest2 : !firrtl.uint<1>

    // Check not deleted if live.
    // CHECK: @live_ref
    %source3, %dest3 = firrtl.instance live_ref @live_ref(in source: !firrtl.uint<1>, out dest: !firrtl.probe<uint<1>>)
    firrtl.matchingconnect %source3, %source : !firrtl.uint<1>
    // CHECK: firrtl.ref.resolve
    %dest3_resolved = firrtl.ref.resolve %dest3 : !firrtl.probe<uint<1>>
    firrtl.matchingconnect %dest, %dest3_resolved : !firrtl.uint<1>

    // Check dead resolve is deleted.
    // CHECK-NOT: dead_instance
    %source4, %dest4 = firrtl.instance dead_instance @live_ref(in source: !firrtl.uint<1>, out dest: !firrtl.probe<uint<1>>)
    firrtl.matchingconnect %source4, %source : !firrtl.uint<1>
    // CHECK-NOT: firrtl.ref.resolve
    %unused5 = firrtl.ref.resolve %dest4 : !firrtl.probe<uint<1>>
  }
}

// -----

// Test the removal of memories in dead cycles

firrtl.circuit "MemoryInDeadCycle" {
  // CHECK-LABEL: firrtl.module public @MemoryInDeadCycle
  // expected-warning @below {{module `MemoryInDeadCycle` is empty but cannot be removed because the module is public}}
  firrtl.module public @MemoryInDeadCycle(in %clock: !firrtl.clock, in %addr: !firrtl.uint<4>) {

    // CHECK-NOT: firrtl.mem
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %Memory_r = firrtl.mem Undefined
      {
        depth = 12 : i64,
        name = "Memory",
        portNames = ["read"],
        readLatency = 0 : i32,
        writeLatency = 1 : i32
      } : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>

    %r_addr = firrtl.subfield %Memory_r[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    firrtl.connect %r_addr, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %r_en = firrtl.subfield %Memory_r[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    firrtl.connect %r_en, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %r_clk = firrtl.subfield %Memory_r[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    firrtl.connect %r_clk, %clock : !firrtl.clock, !firrtl.clock

    // CHECK-NOT: firrtl.mem
    %Memory_w = firrtl.mem Undefined
      {
        depth = 12 : i64,
        name = "Memory",
        portNames = ["w"],
        readLatency = 0 : i32,
        writeLatency = 1 : i32
      } : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>

    %w_addr = firrtl.subfield %Memory_w[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    firrtl.connect %w_addr, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %w_en = firrtl.subfield %Memory_w[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    firrtl.connect %w_en, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %w_clk = firrtl.subfield %Memory_w[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    firrtl.connect %w_clk, %clock : !firrtl.clock, !firrtl.clock
    %w_mask = firrtl.subfield %Memory_w[mask] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    firrtl.connect %w_mask, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>

    %w_data = firrtl.subfield %Memory_w[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>
    %r_data = firrtl.subfield %Memory_r[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>
    firrtl.connect %w_data, %r_data : !firrtl.uint<42>, !firrtl.uint<42>
  }
}

// -----
// CHECK-LABEL: firrtl.circuit "DeadInputPort"
firrtl.circuit "DeadInputPort"  {
  // CHECK-NOT: firrtl.module private @Bar
  firrtl.module private @Bar(in %a: !firrtl.uint<1>) {
  }

  // CHECK-LABEL: firrtl.module @DeadInputPort
  firrtl.module @DeadInputPort(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    // CHECK-NEXT: %0 = firrtl.wire
    // CHECK-NEXT: firrtl.matchingconnect %0, %a
    // CHECK-NEXT: firrtl.matchingconnect %b, %0
    %bar_a = firrtl.instance bar  @Bar(in a: !firrtl.uint<1>)
    firrtl.matchingconnect %bar_a, %a : !firrtl.uint<1>
    firrtl.matchingconnect %b, %bar_a : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "DeleteInstance" {
  // CHECK-NOT: @InvalidValue
  firrtl.module private @InvalidValue() {
      %invalid_ui289 = firrtl.invalidvalue : !firrtl.uint<289>
  }
  firrtl.module private @SideEffect1(in %a: !firrtl.uint<1>, in %clock: !firrtl.clock) {
    firrtl.printf %clock, %a, "foo"  : !firrtl.clock, !firrtl.uint<1>
  }
  firrtl.module private @SideEffect2(in %a: !firrtl.uint<1>, in %clock: !firrtl.clock) {
    %s1_a, %s1_clock = firrtl.instance s1 @SideEffect1(in a: !firrtl.uint<1>, in clock: !firrtl.clock)
    firrtl.matchingconnect %s1_a, %a : !firrtl.uint<1>
    firrtl.matchingconnect %s1_clock, %clock : !firrtl.clock
  }
  firrtl.module private @PassThrough(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }
  // CHECK-LABEL: DeleteInstance
  firrtl.module @DeleteInstance(in %a: !firrtl.uint<1>, in %clock: !firrtl.clock, out %b: !firrtl.uint<1>) {
    // CHECK-NOT: inv
    firrtl.instance inv @InvalidValue()
    // CHECK-NOT: p1
    // CHECK: instance p2 @PassThrough
    // CHECK-NEXT: instance s @SideEffect2
    %p1_a, %p1_b = firrtl.instance p1 @PassThrough(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>)
    %p2_a, %p2_b = firrtl.instance p2 @PassThrough(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>)
    %s_a, %s_clock = firrtl.instance s @SideEffect2(in a: !firrtl.uint<1>, in clock: !firrtl.clock)
    // CHECK-NEXT: firrtl.matchingconnect %s_a, %a : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.matchingconnect %s_clock, %clock : !firrtl.clock
    // CHECK-NEXT: firrtl.matchingconnect %p2_a, %a : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.matchingconnect %b, %p2_b : !firrtl.uint<1>
    firrtl.matchingconnect %s_a, %a : !firrtl.uint<1>
    firrtl.matchingconnect %s_clock, %clock : !firrtl.clock
    firrtl.matchingconnect %p1_a, %a : !firrtl.uint<1>
    firrtl.matchingconnect %p2_a, %a : !firrtl.uint<1>
    firrtl.matchingconnect %b, %p2_b : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "Top" {
  // CHECK: hw.hierpath private @nla_1
  hw.hierpath private @nla_1 [@Top::@foo1, @Bar::@w]
  // CHECK-NEXT: hw.hierpath private @nla_2
  hw.hierpath private @nla_2 [@Top::@foo1, @Bar]
  // CHECK-NEXT: sv.verbatim "foo" {some = [@nla_2]}
  sv.verbatim "foo" {some = [@nla_2]}
  // CHECK-LABEL: firrtl.module private @Bar
  // CHECK: %in1{{.*}}sym @w
  // CHECK-SAME: %in2
  // CHECK-NOT: %in3
  // expected-warning @+1 {{module `Bar` is empty but cannot be removed because the module has ports "in1", "in2" are referenced by name or dontTouched}}
  firrtl.module private @Bar(in %in1 : !firrtl.uint<1> sym @w, in %in2: !firrtl.uint<1> [{class = "foo"}], in %in3: !firrtl.uint<1>) {}
  // CHECK-LABEL: firrtl.module private @Baz
  // expected-warning @+1 {{module `Baz` is empty but cannot be removed because an instance is referenced by nam}}
  firrtl.module private @Baz() {}

  // CHECK-LABEL: firrtl.module @Top
  firrtl.module @Top(in %in: !firrtl.uint<1>) {
    %c_in1, %c_in2, %c_in3 = firrtl.instance c sym @foo1 @Bar(in in1: !firrtl.uint<1>, in in2: !firrtl.uint<1>, in in3: !firrtl.uint<1>)
    firrtl.matchingconnect %c_in1, %in : !firrtl.uint<1>
    firrtl.matchingconnect %c_in2, %in : !firrtl.uint<1>
    firrtl.matchingconnect %c_in3, %in : !firrtl.uint<1>
    // CHECK: sv.verbatim "foo" {some = #hw.innerNameRef<@Top::@baz1>}
    sv.verbatim "foo" {some = #hw.innerNameRef<@Top::@baz1>}
    // Don't remove the instance if there is an unknown use of inner reference.
    // CHECK: baz1
    // expected-note @+1 {{these are instances with symbols}}
    firrtl.instance baz1 sym @baz1 @Baz()
    // Remove a dead instance otherwise.
    // CHECK-NOT: baz2
    firrtl.instance baz2 sym @baz2 @Baz()
  }
}

// -----

// This tests that dead modules do not keep dead instance output ports alive.

firrtl.circuit "Test" {
  firrtl.extmodule @ExtModule(out out : !firrtl.uint<1>)

  // %out is only used by a dead module. It should be removed from the port list.
  // CHECK: firrtl.module private @Blah() {
  firrtl.module private @Blah(out %out : !firrtl.uint<1>) {
    %extmodule_out = firrtl.instance extmodule @ExtModule(out out : !firrtl.uint<1>)
    firrtl.matchingconnect %out, %extmodule_out : !firrtl.uint<1>
  }
  firrtl.module @Test() attributes {convention = #firrtl<convention scalarized>} {
    // CHECK: firrtl.instance blah interesting_name @Blah()
    %blah_out = firrtl.instance blah interesting_name @Blah(out out : !firrtl.uint<1>)
  }
  // This module is dead (unreachable from the toplevel) so the module should be removed.
  // CHECK-NOT: firrtl.module private @Other
  firrtl.module private @Other(out %out : !firrtl.uint<1>) {
    %blah_out = firrtl.instance blah interesting_name @Blah(out out : !firrtl.uint<1>)
    firrtl.matchingconnect %out, %blah_out : !firrtl.uint<1>
  }
}

// -----
// Test that empty classes and objects are kept alive.

firrtl.circuit "Test" {
  // CHECK: firrtl.class private @Empty()
  firrtl.class private @Empty() {}

  // CHECK: firrtl.class private @UnusedAndEmpty()
  firrtl.class private @UnusedAndEmpty() {}

  // CHECK: firrtl.module @Test()
  firrtl.module @Test() {
    // CHECK: %obj = firrtl.object @Empty()
    %obj = firrtl.object @Empty()
  }
}

// -----
// Test that instances of classes are kept alive.

firrtl.circuit "Test" {
  // Both the input and the output of this class are ignored, but preserved by
  // IMDCE.
  // CHECK: firrtl.class private @Class(in %in: !firrtl.integer, out %out: !firrtl.integer)
  firrtl.class private @Class(in %in: !firrtl.integer, out %out: !firrtl.integer) {
    // CHECK:   %0 = firrtl.integer 123
    // CHECK:   firrtl.propassign %out, %0 : !firrtl.integer
    %0 = firrtl.integer 123
    firrtl.propassign %out, %0 : !firrtl.integer
  }

  // The write to %o's "in" port is preserved by IMDCE, even though the input
  // is unused by the class.
  // CHECK: firrtl.module @Test() attributes {convention = #firrtl<convention scalarized>}
  firrtl.module @Test() attributes {convention = #firrtl<convention scalarized>} {
    // CHECK: %0 = firrtl.integer 456
    // CHECK: %o = firrtl.object @Class(in in: !firrtl.integer, out out: !firrtl.integer)
    // CHECK: %1 = firrtl.object.subfield %o[in] : !firrtl.class<@Class(in in: !firrtl.integer, out out: !firrtl.integer)>
    // CHECK: firrtl.propassign %1, %0 : !firrtl.integer
    %0 = firrtl.integer 456
    %o = firrtl.object @Class(in in: !firrtl.integer, out out: !firrtl.integer)
    %1 = firrtl.object.subfield %o[in] : !firrtl.class<@Class(in in: !firrtl.integer, out out: !firrtl.integer)>
    firrtl.propassign %1, %0 : !firrtl.integer
  }
}

// -----
// Test that instances of extclasses are kept alive.

module {
  firrtl.circuit "Test" {
    // CHECK: firrtl.extclass private @Class(out out_str: !firrtl.string, in in_str: !firrtl.string)
    firrtl.extclass private @Class(out out_str: !firrtl.string, in in_str: !firrtl.string)

    // CHECK: firrtl.module @Test(out %out_str: !firrtl.string) attributes {convention = #firrtl<convention scalarized>}
    firrtl.module @Test(out %out_str: !firrtl.string) attributes {convention = #firrtl<convention scalarized>} {
      // CHECK: %0 = firrtl.string "whatever"
      // CHECK: %obj = firrtl.object @Class(out out_str: !firrtl.string, in in_str: !firrtl.string)
      // CHECK: %1 = firrtl.object.subfield %obj[out_str] : !firrtl.class<@Class(out out_str: !firrtl.string, in in_str: !firrtl.string)>
      // CHECK: %2 = firrtl.object.subfield %obj[in_str] : !firrtl.class<@Class(out out_str: !firrtl.string, in in_str: !firrtl.string)>
      // CHECK: firrtl.propassign %2, %0 : !firrtl.string
      // CHECK: firrtl.propassign %out_str, %1 : !firrtl.string
      %0 = firrtl.string "whatever"
      %obj = firrtl.object @Class(out out_str: !firrtl.string, in in_str: !firrtl.string)
      %1 = firrtl.object.subfield %obj[out_str] : !firrtl.class<@Class(out out_str: !firrtl.string, in in_str: !firrtl.string)>
      %2 = firrtl.object.subfield %obj[in_str] : !firrtl.class<@Class(out out_str: !firrtl.string, in in_str: !firrtl.string)>
      firrtl.propassign %2, %0 : !firrtl.string
      firrtl.propassign %out_str, %1 : !firrtl.string
    }
  }
}

// -----
// Test that a live use of a forceable declaration keeps it alive.
// https://github.com/llvm/circt/issues/5898

// CHECK-LABEL: circuit "Issue5898"
firrtl.circuit "Issue5898" {
  firrtl.module @Issue5898(in %x: !firrtl.uint<5>, out %p: !firrtl.rwprobe<uint<5>>) {
    // CHECK: connect
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<5>, !firrtl.rwprobe<uint<5>>
    firrtl.matchingconnect %w, %x : !firrtl.uint<5>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<uint<5>>
  }
}

// -----
// Test that annotations keep declarations alive.
// CHECK-LABEL: "AnnoAlive"
firrtl.circuit "AnnoAlive" {
  firrtl.module @AnnoAlive() {
     // CHECK: firrtl.wire
     firrtl.wire {annotations = [{class = "circt.test"}]} : !firrtl.uint
  }
}

// -----
// Test warning about not being able to remove dead public modules.

// CHECK-LABEL: "DeadPublic"
firrtl.circuit "DeadPublic" {
  // CHECK: module @PublicDeadChild
  // expected-warning @below {{module `PublicDeadChild` is empty but cannot be removed because the module is public}}
  firrtl.module @PublicDeadChild() {}
  // CHECK: module @DeadPublic
  firrtl.module @DeadPublic() {
     firrtl.instance pdc @PublicDeadChild()
  }
}

// -----

// Test that an operation with a nested block user will be removed (and not
// crash).  This should work for both FIRRTL operations and non-FIRRTL
// operations.
//
// CHECK-LABEL: "Foo"
firrtl.circuit "Foo" {
  firrtl.layer @A bind {}
  sv.macro.decl @B["B"]
  // CHECK-NOT: @Bar
  firrtl.module private @Bar() {}
  firrtl.module @Foo() {
    // CHECK-LABEL: firrtl.layerblock @A
    firrtl.layerblock @A {
      // CHECK-NOT: firrtl.instance
      firrtl.instance bar @Bar()
      // CHECK-LABEL: sv.ifdef @B
      sv.ifdef @B {
        // CHECK-NOT: firrtl.instance
        firrtl.instance bar2 @Bar()
      }
    }
  }
}

// -----

// CHECK-LABEL: "Foo"
firrtl.circuit "Foo" {
  firrtl.layer @A bind {}
  // CHECK-LABEL: @Bar
  // CHECK-NOT:     out %probe
  firrtl.module private @Bar(
    in %a: !firrtl.uint<1>,
    out %b: !firrtl.uint<1>,
    out %probe: !firrtl.probe<uint<1>, @A>
  ) {
    // CHECK:      firrtl.layerblock @A {
    // CHECK-NEXT: }
    firrtl.layerblock @A {
      %1 = firrtl.not %a : (!firrtl.uint<1>) -> !firrtl.uint<1>
      %a_not = firrtl.node %1 : !firrtl.uint<1>
      %2 = firrtl.ref.send %a_not : !firrtl.uint<1>
      %3 = firrtl.ref.cast %2 : (!firrtl.probe<uint<1>>) -> !firrtl.probe<uint<1>, @A>
      firrtl.ref.define %probe, %3 : !firrtl.probe<uint<1>, @A>
    }
    // CHECK-NEXT: firrtl.matchingconnect %b, %a
    firrtl.matchingconnect %b, %a : !firrtl.uint<1>
  }
  firrtl.module @Foo(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    %bar_a, %bar_b, %bar_probe = firrtl.instance bar @Bar(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>, out probe: !firrtl.probe<uint<1>, @A>)
    firrtl.matchingconnect %bar_a, %a : !firrtl.uint<1>
    firrtl.matchingconnect %b, %bar_b : !firrtl.uint<1>
  }
}

// -----

// Modules referenced from unknown ops cannot be removed, even if they are empty
// and all their instances have been deleted.
// CHECK-LABEL: firrtl.circuit "FormalMarkerIsUse"
firrtl.circuit "FormalMarkerIsUse" {
  // expected-warning @below {{module `FormalMarkerIsUse` is empty but cannot be removed because the module is public}}
  firrtl.module @FormalMarkerIsUse() {
    // CHECK-NOT: firrtl.instance foo @Foo
    // CHECK-NOT: firrtl.instance bar @Bar
    firrtl.instance foo @Foo()
    firrtl.instance bar @Bar()
  }
  // CHECK: firrtl.module private @Foo
  // CHECK: firrtl.module private @Bar
  // CHECK: firrtl.module private @Uninstantiated
  firrtl.module private @Foo() {}
  firrtl.module private @Bar() {}
  firrtl.module private @Uninstantiated() {}
  firrtl.formal @Test, @Foo {}
  "some_unknown_dialect.op"() { magic = @Bar, other = @Uninstantiated } : () -> ()
}
