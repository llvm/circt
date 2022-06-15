// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-imdeadcodeelim)' --split-input-file  %s | FileCheck %s
firrtl.circuit "top" {
  // In `dead_module`, %source is connected to %dest through several dead operations such as
  // node, wire, reg or rgereset. %dest is also dead at any instantiation, so check that
  // all operations are removed by IMDeadCodeElim pass.
  // CHECK-LABEL: private @dead_module() {
  // CHECK-NEXT:  }
  firrtl.module private @dead_module(in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>,
                                     in %clock:!firrtl.clock, in %reset:!firrtl.uint<1>) {
    %dead_node = firrtl.node droppable_name %source: !firrtl.uint<1>

    %dead_wire = firrtl.wire droppable_name : !firrtl.uint<1>
    firrtl.strictconnect %dead_wire, %dead_node : !firrtl.uint<1>

    %dead_reg = firrtl.reg droppable_name %clock  : !firrtl.uint<1>
    firrtl.strictconnect %dead_reg, %dead_wire : !firrtl.uint<1>

    %dead_reg_reset = firrtl.regreset droppable_name %clock, %reset, %dead_reg  : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.strictconnect %dead_reg_reset, %dead_reg : !firrtl.uint<1>

    %not = firrtl.not %dead_reg_reset : (!firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %dest, %not : !firrtl.uint<1>
  }

  // `%dontTouch` port has a symbol so it shouldn't be removed. `%sym_wire` also has a
  // symbol so check  that `%source` is preserved too.
  // CHECK-LABEL: firrtl.module private @dontTouch(in %dontTouch: !firrtl.uint<1> sym @sym, in %source: !firrtl.uint<1>) {
  firrtl.module private @dontTouch(in %dontTouch: !firrtl.uint<1> sym @sym, in %source: !firrtl.uint<1>, in %dead: !firrtl.uint<1>) {
    // CHECK-NEXT: %sym_wire = firrtl.wire sym @sym   : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.strictconnect %sym_wire, %source : !firrtl.uint<1>
    // CHECK-NEXT: }
    %sym_wire = firrtl.wire sym @sym : !firrtl.uint<1>
    firrtl.strictconnect %sym_wire, %source : !firrtl.uint<1>

  }

  // CHECK-LABEL: firrtl.module private @mem(in %source: !firrtl.uint<1>, out %dest: !firrtl.sint<8>) {
  firrtl.module private @mem(in %source: !firrtl.uint<1>, out %dest: !firrtl.sint<8>) {
    // CHECK-NEXT: %ReadMemory_read0 = firrtl.mem Undefined {depth = 16 : i64, name = "ReadMemory", portNames = ["read0"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>
    %mem = firrtl.mem Undefined {depth = 16 : i64, name = "ReadMemory", portNames = ["read0"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>
    // CHECK-NEXT: %0 = firrtl.subfield %ReadMemory_read0(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>) -> !firrtl.uint<4>
    // CHECK-NEXT: firrtl.connect %0, %source : !firrtl.uint<4>, !firrtl.uint<1>
    // CHECK: }
    %0 = firrtl.subfield %mem(0) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>) -> !firrtl.uint<4>
    firrtl.connect %0, %source : !firrtl.uint<4>, !firrtl.uint<1>

    %read = firrtl.subfield %mem(3) : (!firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>) -> !firrtl.sint<8>
    firrtl.connect %dest, %read : !firrtl.sint<8>, !firrtl.sint<8>
  }

  // Ports of public modules should not be modified.
  // CHECK-LABEL: firrtl.module @top(in %source: !firrtl.uint<1>, out %dest1: !firrtl.uint<1>, out %dest2: !firrtl.sint<8>, in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
  firrtl.module @top(in %source: !firrtl.uint<1>, out %dest1: !firrtl.uint<1>, out %dest2: !firrtl.sint<8>,
                     in %clock:!firrtl.clock, in %reset:!firrtl.uint<1>) {
    // CHECK-NEXT: %tmp = firrtl.node droppable_name %source
    // CHECK-NEXT: firrtl.strictconnect %dest1, %tmp
    %tmp = firrtl.node droppable_name %source: !firrtl.uint<1>
    firrtl.strictconnect %dest1, %tmp : !firrtl.uint<1>

    // TODO: Remove instances of empty modules.
    // CHECK-NEXT: firrtl.instance dead_module  @dead_module()
    %source1, %dummy, %clock1, %reset1  = firrtl.instance dead_module @dead_module(in source: !firrtl.uint<1>, out dest: !firrtl.uint<1>, in clock:!firrtl.clock, in reset:!firrtl.uint<1>)
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

    // CHECK-NEXT: %mem_source, %mem_dest = firrtl.instance mem @mem(in source: !firrtl.uint<1>, out dest: !firrtl.sint<8>)
    // CHECK-NEXT: firrtl.strictconnect %mem_source, %source
    // CHECK-NEXT: firrtl.strictconnect %dest2, %mem_dest
    %mem_source, %mem_dest  = firrtl.instance mem @mem(in source: !firrtl.uint<1>, out dest: !firrtl.sint<8>)
    firrtl.strictconnect %mem_source, %source : !firrtl.uint<1>
    firrtl.strictconnect %dest2, %mem_dest : !firrtl.sint<8>
    // CHECK-NEXT: }
  }
}

// -----

// Check that it's possible to analyze complex dependency across different modules.
firrtl.circuit "top"  {
  // CHECK-LABEL: firrtl.module private @Child1() {
  // CHECK-NEXT:  }
  firrtl.module private @Child1(in %input: !firrtl.uint<1>, out %output: !firrtl.uint<1>) {
    firrtl.strictconnect %output, %input : !firrtl.uint<1>
  }
  // CHECK-LABEL: firrtl.module private @Child2() {
  // CHECK-NEXT:  }
  firrtl.module private @Child2(in %input: !firrtl.uint<1>, in %clock: !firrtl.clock, out %output: !firrtl.uint<1>) {
    %r = firrtl.reg droppable_name %clock  : !firrtl.uint<1>
    firrtl.strictconnect %r, %input : !firrtl.uint<1>
    firrtl.strictconnect %output, %r : !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module @top(in %clock: !firrtl.clock, in %input: !firrtl.uint<1>) {
  // CHECK-NEXT:    firrtl.instance tile  @Child1()
  // CHECK-NEXT:    firrtl.instance bar  @Child2()
  // CHECK-NEXT:  }
  firrtl.module @top(in %clock: !firrtl.clock, in %input: !firrtl.uint<1>) {
    %tile_input, %tile_output = firrtl.instance tile  @Child1(in input: !firrtl.uint<1>, out output: !firrtl.uint<1>)
    firrtl.strictconnect %tile_input, %input : !firrtl.uint<1>
    %named = firrtl.node droppable_name  %tile_output  : !firrtl.uint<1>
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
  // CHECK-SAME:   output_file
  firrtl.module private @Sub(in %a: !firrtl.uint<1>) attributes {output_file = #hw.output_file<"hello">} {}
  // CHECK: firrtl.module @PreserveOutputFile
  firrtl.module @PreserveOutputFile() {
    // CHECK-NEXT: firrtl.instance sub
    // CHECK-SAME: output_file
    firrtl.instance sub {output_file = #hw.output_file<"hello">} @Sub(in a: !firrtl.uint<1>)
  }
}
