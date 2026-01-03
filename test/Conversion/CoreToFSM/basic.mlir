// RUN: circt-opt -convert-core-to-fsm %s | FileCheck %s

// Test basic RTL-to-FSM conversion with a simple counter module.
// The counter increments by 1 when inp=1, stays the same when inp=0.
// State encoding: state_0=0, state_1=1, state_2=2, state_3=3

// CHECK-LABEL: fsm.machine @counter
// CHECK-SAME: (%[[INP:.*]]: i1) -> i1
// CHECK-SAME: attributes {initialState = "state_0"}
hw.module @counter(in %clk : !seq.clock, in %rst : i1, in %inp : i1, out output : i1) {
    %c0_i2 = hw.constant 0 : i2
    %c0_i1 = hw.constant 0 : i1
    %c2_i2 = hw.constant 2 : i2
    %true = hw.constant true
    %state = seq.compreg sym @state name "state" %next_state, %clk reset %rst, %c0_i2 : i2
    %add = comb.concat %c0_i1, %inp : i1, i1
    %next_state = comb.add %state, %add : i2
    %is_2 = comb.icmp eq %state, %c2_i2 : i2
    hw.output %is_2 : i1
}

// Verify states are created with correct outputs
// CHECK: fsm.state @state_0 output {
// CHECK:   %[[FALSE:.*]] = hw.constant false
// CHECK:   fsm.output %[[FALSE]] : i1
// CHECK: } transitions {
// Transition to state_1 when inp=1
// CHECK:   fsm.transition @state_1 guard {
// CHECK:     fsm.return %[[INP]]
// CHECK:   }
// Stay in state_0 when inp=0
// CHECK:   fsm.transition @state_0 guard {

// CHECK: fsm.state @state_1 output {
// CHECK:   %[[FALSE:.*]] = hw.constant false
// CHECK:   fsm.output %[[FALSE]] : i1
// CHECK: } transitions {
// Transitions from state_1: to state_2 (inp=1), stay at state_1 (inp=0)
// CHECK:   fsm.transition @state_{{[0-3]}} guard {

// CHECK: fsm.state @state_2 output {
// CHECK:   %[[TRUE:.*]] = hw.constant true
// CHECK:   fsm.output %[[TRUE]] : i1
// CHECK: } transitions {
// Transitions from state_2: to state_3 (inp=1), stay at state_2 (inp=0)
// CHECK:   fsm.transition @state_{{[0-3]}} guard {
