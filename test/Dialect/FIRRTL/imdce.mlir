// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-imdeadcodeelim)' --split-input-file  %s | FileCheck %s
firrtl.circuit "top" {
  // In `dead_module`, %source is connected to %dest through several dead operations such as
  // node, wire, reg or rgereset. %dest is also dead at any instantiation, so check that
  // all operations are removed by IMDeadCodeElim pass.
  // CHECK-NOT: @dead_module
  firrtl.module private @dead_module(in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>,
                                     in %clock:!firrtl.clock, in %reset:!firrtl.uint<1>) {
    %dead_node = firrtl.node %source: !firrtl.uint<1>

    %dead_wire = firrtl.wire : !firrtl.uint<1>
    firrtl.strictconnect %dead_wire, %dead_node : !firrtl.uint<1>

    %dead_reg = firrtl.reg %clock : !firrtl.uint<1>
    firrtl.strictconnect %dead_reg, %dead_wire : !firrtl.uint<1>

    %dead_reg_reset = firrtl.regreset %clock, %reset, %dead_reg  : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.strictconnect %dead_reg_reset, %dead_reg : !firrtl.uint<1>

    %not = firrtl.not %dead_reg_reset : (!firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %dest, %not : !firrtl.uint<1>
  }

  // `%dontTouch` port has a symbol so it shouldn't be removed. `%sym_wire` also has a
  // symbol so check  that `%source` is preserved too.
  // CHECK-LABEL: firrtl.module private @dontTouch(in %dontTouch: !firrtl.uint<1> sym @sym, in %source: !firrtl.uint<1>) {
  firrtl.module private @dontTouch(in %dontTouch: !firrtl.uint<1> sym @sym, in %source: !firrtl.uint<1>, in %dead: !firrtl.uint<1>) {
    // CHECK-NEXT: %sym_wire = firrtl.wire sym @sym2   : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.strictconnect %sym_wire, %source : !firrtl.uint<1>
    // CHECK-NEXT: }
    %sym_wire = firrtl.wire sym @sym2 : !firrtl.uint<1>
    firrtl.strictconnect %sym_wire, %source : !firrtl.uint<1>

  }

  // CHECK-LABEL: firrtl.module private @mem(in %source: !firrtl.uint<1>) {
  firrtl.module private @mem(in %source: !firrtl.uint<1>) {
    // CHECK-NEXT: %ReadMemory_read0 = firrtl.mem Undefined {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}], depth = 16 : i64, name = "ReadMemory", portNames = ["read0"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>
    %mem = firrtl.mem Undefined {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}], depth = 16 : i64, name = "ReadMemory", portNames = ["read0"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>
    // CHECK-NEXT: %0 = firrtl.subfield %ReadMemory_read0(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>) -> !firrtl.uint<4>
    // CHECK-NEXT: firrtl.connect %0, %source : !firrtl.uint<4>, !firrtl.uint<1>
    // CHECK-NEXT: }
    %0 = firrtl.subfield %mem(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>) -> !firrtl.uint<4>
    firrtl.connect %0, %source : !firrtl.uint<4>, !firrtl.uint<1>
  }

  // Ports of public modules should not be modified.
  // CHECK-LABEL: firrtl.module @top(in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>, in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
  firrtl.module @top(in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>,
                     in %clock:!firrtl.clock, in %reset:!firrtl.uint<1>) {
    // CHECK-NEXT: %tmp = firrtl.node %source
    // CHECK-NEXT: firrtl.strictconnect %dest, %tmp
    %tmp = firrtl.node %source: !firrtl.uint<1>
    firrtl.strictconnect %dest, %tmp : !firrtl.uint<1>

    // CHECK-NOT: @dead_module
    %source1, %dest1, %clock1, %reset1  = firrtl.instance dead_module @dead_module(in source: !firrtl.uint<1>, out dest: !firrtl.uint<1>, in clock:!firrtl.clock, in reset:!firrtl.uint<1>)
    firrtl.strictconnect %source1, %source : !firrtl.uint<1>
    firrtl.strictconnect %clock1, %clock : !firrtl.clock
    firrtl.strictconnect %reset1, %reset : !firrtl.uint<1>

    // Check that ports with dontTouch are not removed.
    // CHECK-NEXT: %testDontTouch_dontTouch, %testDontTouch_source = firrtl.instance testDontTouch @dontTouch(in dontTouch: !firrtl.uint<1>, in source: !firrtl.uint<1>)
    // CHECK-NEXT: firrtl.strictconnect %testDontTouch_dontTouch, %source
    // CHECK-NEXT: firrtl.strictconnect %testDontTouch_source, %source
    %testDontTouch_dontTouch, %testDontTouch_source,  %dead = firrtl.instance testDontTouch @dontTouch(in dontTouch: !firrtl.uint<1>, in source: !firrtl.uint<1>, in dead:!firrtl.uint<1>)
    firrtl.strictconnect %testDontTouch_dontTouch, %source : !firrtl.uint<1>
    firrtl.strictconnect %testDontTouch_source, %source : !firrtl.uint<1>
    firrtl.strictconnect %dead, %source : !firrtl.uint<1>

    // CHECK-NEXT: %mem_source = firrtl.instance mem @mem(in source: !firrtl.uint<1>)
    // CHECK-NEXT: firrtl.strictconnect %mem_source, %source : !firrtl.uint<1>
    %mem_source  = firrtl.instance mem @mem(in source: !firrtl.uint<1>)
    firrtl.strictconnect %mem_source, %source : !firrtl.uint<1>
    // CHECK-NEXT: }
  }
}

// -----

// Check that it's possible to analyze complex dependency across different modules.
firrtl.circuit "top"  {
  // CHECK-NOT: @Child1
  firrtl.module private @Child1(in %input: !firrtl.uint<1>, out %output: !firrtl.uint<1>) {
    firrtl.strictconnect %output, %input : !firrtl.uint<1>
  }
  // CHECK-NOT: @Child2
  firrtl.module private @Child2(in %input: !firrtl.uint<1>, in %clock: !firrtl.clock, out %output: !firrtl.uint<1>) {
    %r = firrtl.reg %clock  : !firrtl.uint<1>
    firrtl.strictconnect %r, %input : !firrtl.uint<1>
    firrtl.strictconnect %output, %r : !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module @top(in %clock: !firrtl.clock, in %input: !firrtl.uint<1>) {
  // CHECK-NEXT:  }
  firrtl.module @top(in %clock: !firrtl.clock, in %input: !firrtl.uint<1>) {
    %tile_input, %tile_output = firrtl.instance tile  @Child1(in input: !firrtl.uint<1>, out output: !firrtl.uint<1>)
    firrtl.strictconnect %tile_input, %input : !firrtl.uint<1>
    %named = firrtl.node  %tile_output  : !firrtl.uint<1>
    %bar_input, %bar_clock, %bar_output = firrtl.instance bar  @Child2(in input: !firrtl.uint<1>, in clock: !firrtl.clock, out output: !firrtl.uint<1>)
    firrtl.strictconnect %bar_clock, %clock : !firrtl.clock
    firrtl.strictconnect %bar_input, %named : !firrtl.uint<1>
  }
}

// -----

// CHECK-LABEL: "UnusedOutput"
firrtl.circuit "UnusedOutput"  {
  // CHECK: firrtl.module {{.+}}@SingleDriver
  // CHECK-NOT:     out %c
  firrtl.module private @SingleDriver(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>) {
    // CHECK-NEXT: %[[c_wire:.+]] = firrtl.wire
    // CHECK-NEXT: firrtl.strictconnect %b, %[[c_wire]]
    firrtl.strictconnect %b, %c : !firrtl.uint<1>
    // CHECK-NEXT: %[[not_a:.+]] = firrtl.not %a
    %0 = firrtl.not %a : (!firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK-NEXT: firrtl.strictconnect %[[c_wire]], %[[not_a]]
    firrtl.strictconnect %c, %0 : !firrtl.uint<1>
  }
  // CHECK-LABEL: @UnusedOutput
  firrtl.module @UnusedOutput(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    // CHECK: %singleDriver_a, %singleDriver_b = firrtl.instance singleDriver
    %singleDriver_a, %singleDriver_b, %singleDriver_c = firrtl.instance singleDriver @SingleDriver(in a: !firrtl.uint<1>, out b: !firrtl.uint<1>, out c: !firrtl.uint<1>)
    firrtl.strictconnect %singleDriver_a, %a : !firrtl.uint<1>
    firrtl.strictconnect %b, %singleDriver_b : !firrtl.uint<1>
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
  // Don't delete @Sub because instance `sub1` has a symbol.
  // CHECK: firrtl.module private @Sub
  firrtl.module private @Sub(in %a: !firrtl.uint<1>)  {}
  // CHECK: firrtl.module @DeleteEmptyModule
  firrtl.module @DeleteEmptyModule() {
    // CHECK-NEXT: firrtl.instance sub1 sym @Foo @Sub()
    firrtl.instance sub1 sym @Foo @Sub(in a: !firrtl.uint<1>)
    // CHECK-NOT: sub2
    firrtl.instance sub2 @Sub(in a: !firrtl.uint<1>)
  }
}

// -----

// CHECK-LABEL: "ForwardConstant"
firrtl.circuit "ForwardConstant" {
  // CHECK-NOT: Zero
  firrtl.module private @Zero(out %zero: !firrtl.uint<1>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    firrtl.strictconnect %zero, %c0_ui1 : !firrtl.uint<1>
  }
  // CHECK-LABEL: @ForwardConstant
  firrtl.module @ForwardConstant(out %zero: !firrtl.uint<1>) {
    // CHECK: %c0_ui1 = firrtl.constant 0
    %sub_zero = firrtl.instance sub @Zero(out zero: !firrtl.uint<1>)
    // CHECK-NEXT: firrtl.strictconnect %zero, %c0_ui1
    firrtl.strictconnect %zero, %sub_zero : !firrtl.uint<1>
  }
}

// -----

// Test handling of ref ports and ops.

// CHECK-LABEL: "RefPorts"
firrtl.circuit "RefPorts" {
  // CHECK-NOT: @dead_ref_send
  firrtl.module private @dead_ref_send(in %source: !firrtl.uint<1>, out %dest: !firrtl.ref<uint<1>>) {
    %ref = firrtl.ref.send %source: !firrtl.uint<1>
    firrtl.strictconnect %dest, %ref : !firrtl.ref<uint<1>>
  }

  // CHECK-LABEL: @dead_ref_port
  // CHECK-NOT: firrtl.ref
  firrtl.module private @dead_ref_port(in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>, out %ref_dest: !firrtl.ref<uint<1>>) {
    %ref_not = firrtl.ref.send %source: !firrtl.uint<1>
    firrtl.strictconnect %ref_dest, %ref_not : !firrtl.ref<uint<1>>
    firrtl.strictconnect %dest, %source : !firrtl.uint<1>
  }

  // CHECK: @live_ref
  firrtl.module private @live_ref(in %source: !firrtl.uint<1>, out %dest: !firrtl.ref<uint<1>>) {
    %ref_source = firrtl.ref.send %source: !firrtl.uint<1>
    firrtl.strictconnect %dest, %ref_source : !firrtl.ref<uint<1>>
  }

  // CHECK-LABEL: @RefPorts
  firrtl.module @RefPorts(in %source : !firrtl.uint<1>, out %dest : !firrtl.uint<1>) {
    // Delete send's that aren't resolved, and check deletion of modules with ref ops + ports.
    // CHECK-NOT: @dead_ref_send
    %source1, %dest1 = firrtl.instance dead_ref_send @dead_ref_send(in source: !firrtl.uint<1>, out dest: !firrtl.ref<uint<1>>)
    firrtl.strictconnect %source1, %source : !firrtl.uint<1>

    // Check that an unused resolve doesn't keep send alive, and test ref port removal.
    // CHECK: @dead_ref_port
    // CHECK-NOT: firrtl.ref
    %source2, %dest2, %ref_dest2 = firrtl.instance dead_ref_port @dead_ref_port(in source: !firrtl.uint<1>, out dest: !firrtl.uint<1>, out ref_dest: !firrtl.ref<uint<1>>)
    firrtl.strictconnect %source2, %source : !firrtl.uint<1>
    %unused = firrtl.ref.resolve %ref_dest2 : !firrtl.ref<uint<1>>
    firrtl.strictconnect %dest, %dest2 : !firrtl.uint<1>

    // Check not deleted if live.
    // CHECK: @live_ref
    %source3, %dest3 = firrtl.instance live_ref @live_ref(in source: !firrtl.uint<1>, out dest: !firrtl.ref<uint<1>>)
    firrtl.strictconnect %source3, %source : !firrtl.uint<1>
    // CHECK: firrtl.ref.resolve
    %dest3_resolved = firrtl.ref.resolve %dest3 : !firrtl.ref<uint<1>>
    firrtl.strictconnect %dest, %dest3_resolved : !firrtl.uint<1>

    // Check dead resolve is deleted, even if send isn't.
    // (Instance is dead too but need context-sensitive analysis to show that.)
    // CHECK: @live_ref
    %source4, %dest4 = firrtl.instance live_ref @live_ref(in source: !firrtl.uint<1>, out dest: !firrtl.ref<uint<1>>)
    firrtl.strictconnect %source4, %source : !firrtl.uint<1>
    // CHECK-NOT: firrtl.ref.resolve
    %unused5 = firrtl.ref.resolve %dest4 : !firrtl.ref<uint<1>>
  }
}

// -----

// Test the removal of memories in dead cycles

firrtl.circuit "MemoryInDeadCycle" {
  // CHECK-LABEL: firrtl.module public @MemoryInDeadCycle
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

    %r_addr = firrtl.subfield %Memory_r(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>) -> !firrtl.uint<4>
    firrtl.connect %r_addr, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %r_en = firrtl.subfield %Memory_r(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>) -> !firrtl.uint<1>
    firrtl.connect %r_en, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %r_clk = firrtl.subfield %Memory_r(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>) -> !firrtl.clock
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

    %w_addr = firrtl.subfield %Memory_w(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>) -> !firrtl.uint<4>
    firrtl.connect %w_addr, %addr : !firrtl.uint<4>, !firrtl.uint<4>
    %w_en = firrtl.subfield %Memory_w(1) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %w_en, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %w_clk = firrtl.subfield %Memory_w(2) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>) -> !firrtl.clock
    firrtl.connect %w_clk, %clock : !firrtl.clock, !firrtl.clock
    %w_mask = firrtl.subfield %Memory_w(4) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %w_mask, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>

    %w_data = firrtl.subfield %Memory_w(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<42>, mask: uint<1>>) -> !firrtl.uint<42>
    %r_data = firrtl.subfield %Memory_r(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<42>>) -> !firrtl.uint<42>
    firrtl.connect %w_data, %r_data : !firrtl.uint<42>, !firrtl.uint<42>
  }
}
