// RUN: circt-opt -convert-core-to-fsm %s | FileCheck %s

// Test RTL-to-FSM conversion with both a state register and a variable register.
// This test verifies that non-state registers become fsm.variable operations,
// and that transitions include fsm.update operations in their action blocks.

// The module has:
// - "state" register: determines FSM state (2-bit counter)
// - "counter" register: non-state variable that increments when in state 0

// CHECK-LABEL: fsm.machine @counter_with_variable
// CHECK-SAME: (%[[INP:.*]]: i1) -> i1
// CHECK-SAME: attributes {initialState = "state_0"}
// CHECK: %[[COUNTER:.*]] = fsm.variable "counter" {initValue = 0 : i4} : i4
hw.module @counter_with_variable(in %clk : !seq.clock, in %rst : i1, in %inp : i1, out output : i1) {
    %c0_i2 = hw.constant 0 : i2
    %c0_i4 = hw.constant 0 : i4
    %c0_i1 = hw.constant 0 : i1
    %c2_i2 = hw.constant 2 : i2
    %true = hw.constant true

    // State register - determines FSM state
    %state = seq.compreg name "state" %next_state, %clk reset %rst, %c0_i2 : i2

    // Variable register - counts how many times we've been in state 0
    %counter = seq.compreg name "counter" %next_counter, %clk reset %rst, %c0_i4 : i4

    // State logic: increment state by inp
    %add = comb.concat %c0_i1, %inp : i1, i1
    %next_state = comb.add %state, %add : i2

    // Counter logic: if we're in state 0, increment the counter
    %in_state_0 = comb.icmp eq %state, %c0_i2 : i2
    %one_i4 = hw.constant 1 : i4
    %counter_plus_1 = comb.add %counter, %one_i4 : i4
    %next_counter = comb.mux %in_state_0, %counter_plus_1, %counter : i4

    // Output: is state == 2?
    %is_2 = comb.icmp eq %state, %c2_i2 : i2
    hw.output %is_2 : i1
}

// Verify state_0 has transitions with action blocks containing fsm.update
// CHECK: fsm.state @state_0 output {
// CHECK:   %[[FALSE:.*]] = hw.constant false
// CHECK:   fsm.output %[[FALSE]] : i1
// CHECK: } transitions {
// CHECK:   fsm.transition @state_1 guard {
// CHECK:     fsm.return %[[INP]]
// CHECK:   } action {
// Transition state_0 -> state_1 should update counter
// CHECK:     %[[C1:.*]] = hw.constant 1 : i4
// CHECK:     %[[SUM:.*]] = comb.add %[[COUNTER]], %[[C1]] : i4
// CHECK:     fsm.update %[[COUNTER]], %[[SUM]] : i4
// CHECK:   }
// CHECK:   fsm.transition @state_0 guard {
// CHECK:   } action {
// Transition state_0 -> state_0 should also update counter
// CHECK:     fsm.update %[[COUNTER]],
// CHECK:   }

// Verify state_1 has transitions with action blocks
// CHECK: fsm.state @state_1 output {
// CHECK: } transitions {
// CHECK:   fsm.transition @state_{{[0-3]}} guard {
// CHECK:   } action {
// In state_1, counter keeps its value
// CHECK:     fsm.update %[[COUNTER]], %[[COUNTER]] : i4
// CHECK:   }

// Verify state_2 exists with output true
// CHECK: fsm.state @state_2 output {
// CHECK:   %[[TRUE:.*]] = hw.constant true
// CHECK:   fsm.output %[[TRUE]] : i1
// CHECK: } transitions {
// CHECK:   fsm.transition @state_{{[0-3]}} guard {
// CHECK:   } action {
// In state_2, counter keeps its value
// CHECK:     fsm.update %[[COUNTER]], %[[COUNTER]] : i4
// CHECK:   }
