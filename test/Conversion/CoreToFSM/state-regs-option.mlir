// RUN: circt-opt -convert-core-to-fsm="state-regs=my_fsm_reg" %s | FileCheck %s

// Test that the --state-regs option correctly identifies a non-default state register.
// The register "my_fsm_reg" does not contain "state" in its name, so it would not
// be detected by default. Using --state-regs=my_fsm_reg specifies it manually.

// CHECK-LABEL: fsm.machine @traffic_light
// CHECK-SAME: (%[[INP:.*]]: i1) -> i1
// CHECK-SAME: attributes {initialState = "state_0"}
hw.module @traffic_light(in %clk : !seq.clock, in %rst : i1, in %inp : i1, out output : i1) {
    %c0_i2 = hw.constant 0 : i2
    %c0_i1 = hw.constant 0 : i1
    %c2_i2 = hw.constant 2 : i2
    %true = hw.constant true
    // Register named "my_fsm_reg" - does not contain "state" but is specified via --state-regs
    %my_fsm_reg = seq.compreg name "my_fsm_reg" %next_val, %clk reset %rst, %c0_i2 : i2
    %add = comb.concat %c0_i1, %inp : i1, i1
    %next_val = comb.add %my_fsm_reg, %add : i2
    %is_2 = comb.icmp eq %my_fsm_reg, %c2_i2 : i2
    hw.output %is_2 : i1
}

// Verify that states are created correctly with proper transitions
// CHECK: fsm.state @state_0 output {
// CHECK:   %[[FALSE:.*]] = hw.constant false
// CHECK:   fsm.output %[[FALSE]] : i1
// CHECK: } transitions {
// From state_0: inp=1 -> state_1, inp=0 -> state_0
// CHECK:   fsm.transition @state_1 guard {
// CHECK:     fsm.return %[[INP]]
// CHECK:   }
// CHECK:   fsm.transition @state_0 guard {

// CHECK: fsm.state @state_1 output {
// CHECK:   %[[FALSE1:.*]] = hw.constant false
// CHECK:   fsm.output %[[FALSE1]] : i1
// CHECK: } transitions {
// From state_1: transitions based on inp value
// CHECK:   fsm.transition @state_{{[0-3]}} guard {

// CHECK: fsm.state @state_2 output {
// CHECK:   %[[TRUE:.*]] = hw.constant true
// CHECK:   fsm.output %[[TRUE]] : i1
// CHECK: } transitions {
// From state_2: transitions based on inp value
// CHECK:   fsm.transition @state_{{[0-3]}} guard {
